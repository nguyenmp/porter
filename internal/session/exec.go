package session

import (
	"context"
	"errors"
	"fmt"
	"io"
	"strings"

	"porter/internal/api"
	"porter/internal/exec"
	"porter/internal/llm"
	"porter/internal/tools"
)

// execClient is one execution provider a session can run tools with: a
// connected remote client (e.g. the REPL on a laptop), or the implicit local
// provider (the server process, always selectable, never in the map). ch is
// the client's exec subscription — the channel tool calls are sent down; it
// stays live while the client is connected even when another provider is
// active, so a deselected client can be selected again without reconnecting.
// ctx is the environment the client reported (system, cwd, files, skills),
// which the active provider injects into the model and uses for load_skill.
type execClient struct {
	id        string
	name      string // human-readable label, e.g. hostname; "" for legacy clients
	kind      string // "remote" today; "cloud" and friends later
	ch        chan api.ExecRequest
	ctx       *api.ExecContext
	connected bool
}

// remoteProvider routes a session's tool calls to the active execution client
// instead of running them locally. It implements tools.Provider so
// agent.RunTurn is unchanged: Run blocks until the client streams the
// command's output back, returning that output as a stream the agent reads to
// completion. The tools it exposes (including load_skill) and the environment
// context it reports both come from the active client's registered context.
type remoteProvider struct {
	sess *Session
}

// activeClient returns the connected remote client the session's provider
// routes to — the active one — or nil when the active provider is local.
func (r *remoteProvider) activeClient() *execClient {
	s := r.sess
	s.mu.Lock()
	defer s.mu.Unlock()
	c, ok := s.execClients[s.activeExec]
	if !ok || !c.connected || c.kind == "local" {
		return nil
	}
	return c
}

// Defs returns the tools the active client can run: shell plus load_skill when
// the client reported skills. It is read from the client's registered context,
// so it stays in sync with what the provider actually loaded.
func (r *remoteProvider) Defs() []llm.Tool {
	c := r.activeClient()
	var skills []api.Skill
	if c != nil && c.ctx != nil {
		skills = c.ctx.Skills
	}
	return tools.DefsForSkills(skills)
}

// Environment returns the active client's reported environment context as a
// system message, or "" when none has been registered.
func (r *remoteProvider) Environment() string {
	c := r.activeClient()
	if c == nil || c.ctx == nil {
		return ""
	}
	return exec.SystemMessage(*c.ctx)
}

// Run registers a tool call with the active execution client and returns a
// stream of the client's output. The client picks the call up on its exec
// subscription and streams the result back via POST /exec/{call_id}.
func (r *remoteProvider) Run(ctx context.Context, name string, args []byte) (io.ReadCloser, error) {
	s := r.sess
	callID := s.newCallID()
	pr, pw := io.Pipe()

	call := &execCall{pw: pw}
	s.mu.Lock()
	if s.execCalls == nil {
		s.execCalls = map[string]*execCall{}
	}
	s.execCalls[callID] = call
	c, ok := s.execClients[s.activeExec]
	s.mu.Unlock()

	if !ok || c == nil || c.ch == nil {
		s.dropCall(callID, pw)
		return nil, errors.New("no execution client connected")
	}
	select {
	case c.ch <- api.ExecRequest{CallID: callID, Name: name, Arguments: string(args)}:
	case <-ctx.Done():
		s.dropCall(callID, pw)
		return nil, ctx.Err()
	}
	// If the run's context is cancelled (a user clicking Cancel in the UI), the
	// agent's Read on pr must unblock even if the client never sends a result:
	// close the write end so pr reads EOF. The session also pushes a
	// Cancel=true request down the exec channel (see CancelRun) so the client
	// actually stops its command; this close is what lets the agent stop
	// waiting promptly regardless of the client's behavior.
	go func() {
		<-ctx.Done()
		_ = pw.Close()
	}()
	return pr, nil
}

// localProvider runs tools in the server process and reports the server's own
// environment context (system, working directory, files, skills), so "local"
// is a first-class provider in the selector instead of an anonymous fallback:
// the model sees where local commands run and can load local skills.
type localProvider struct {
	d   *tools.Dispatcher
	ctx api.ExecContext
}

// Defs returns the tools the local provider can run: shell plus load_skill for
// the skills discovered in the server's own environment.
func (p *localProvider) Defs() []llm.Tool { return p.d.Defs() }

// Environment returns the server process's environment context as a system
// message, so the model knows where local commands run.
func (p *localProvider) Environment() string { return exec.SystemMessage(p.ctx) }

// Run executes the tool locally, streaming its output.
func (p *localProvider) Run(ctx context.Context, name string, args []byte) (io.ReadCloser, error) {
	return p.d.Run(ctx, name, args)
}

// execCall tracks one in-flight tool call awaiting the client's streamed result.
type execCall struct {
	pw *io.PipeWriter
}

// SetExecContext stores the environment context a connected execution client
// reported. The context is registered separately from the exec connection
// (the REPL posts it before opening the connection), so it is held in the
// pending map keyed by the client's id and attached when the client registers.
// A context without an id (legacy clients) is attached to the next client to
// register. An id that names an already-connected client updates it in place.
// The active provider exposes the context as the model's system message and
// the load_skill tool definitions.
func (s *Session) SetExecContext(ctx api.ExecContext) {
	s.mu.Lock()
	defer s.mu.Unlock()
	id := ctx.ID
	if c, ok := s.execClients[id]; ok {
		c.ctx = &ctx
		return
	}
	if s.pendingCtx == nil {
		s.pendingCtx = map[string]*api.ExecContext{}
	}
	s.pendingCtx[id] = &ctx
}

// ExecStatus returns the session's execution provider state: whether the
// active provider is a remote client, its kind and context, the active id, and
// the full registry (local first) the selector renders.
func (s *Session) ExecStatus() api.ExecStatus {
	s.mu.Lock()
	defer s.mu.Unlock()
	st := api.ExecStatus{ActiveID: s.activeExec}
	if c, ok := s.execClients[s.activeExec]; ok && c.connected && c.kind != "local" {
		st.Connected = true
		st.Kind = c.kind
		st.Context = c.ctx
	} else {
		st.Kind = "local"
		ctx := s.localContextLocked()
		st.Context = &ctx
	}
	st.Clients = make([]api.ExecClient, 0, len(s.execClients)+1)
	lctx := s.localContextLocked()
	st.Clients = append(st.Clients, api.ExecClient{
		ID:        "local",
		Name:      "local",
		Kind:      "local",
		Connected: true,
		Context:   &lctx,
	})
	for _, c := range s.execClients {
		st.Clients = append(st.Clients, api.ExecClient{
			ID:        c.id,
			Name:      c.name,
			Kind:      c.kind,
			Connected: c.connected,
			Context:   c.ctx,
		})
	}
	return st
}

// RegisterExec adds a connected execution client to the session's registry and
// makes it the active provider: a connecting client takes over, matching the
// original behavior (the REPL creates the session and runs its tools; the user
// can then pick another provider in the selector). It returns the client's
// effective id — the server generates one when the client didn't send its own
// (legacy binaries, tests) — so the caller's deferred UnregisterExec can name
// the same client. It commits a system notice so the model sees the
// environment change, and broadcasts the new status.
func (s *Session) RegisterExec(ch chan api.ExecRequest, id, name, kind string) string {
	s.mu.Lock()
	var ctx *api.ExecContext
	if id == "" || id == "local" { // "local" is reserved for the server process
		id = s.nextClientID()
		if c, ok := s.pendingCtx[""]; ok { // legacy context posted without an id
			ctx = c
			delete(s.pendingCtx, "")
		}
	} else if c, ok := s.pendingCtx[id]; ok {
		ctx = c
		delete(s.pendingCtx, id)
	}
	c, ok := s.execClients[id]
	if !ok {
		c = &execClient{id: id}
		s.execClients[id] = c
	}
	c.name = name
	if ctx != nil {
		c.ctx = ctx
		if c.name == "" {
			c.name = ctx.Name
		}
	}
	c.kind = kind
	if c.kind == "" {
		c.kind = "remote"
	}
	c.ch = ch
	c.connected = true
	s.activeExec = id
	s.mu.Unlock()

	s.commitProviderNotice(s.providerNotice("execution provider connected", c.name, c.ctx))
	s.publishStatus()
	return id
}

// UnregisterExec removes a connected execution client from the registry and,
// when it was the active provider, closes its in-flight calls (so the agent
// does not hang) and reverts to local execution with a system notice. A
// non-active client's disconnect only updates the registry — the model's
// environment didn't change — but the status is still broadcast so the
// selector drops the entry.
func (s *Session) UnregisterExec(id string) {
	s.mu.Lock()
	_, ok := s.execClients[id]
	if !ok {
		s.mu.Unlock()
		return
	}
	delete(s.execClients, id)
	reverted := s.activeExec == id
	if reverted {
		s.activeExec = "local"
		for callID, call := range s.execCalls {
			_ = call.pw.CloseWithError(errors.New("execution client disconnected"))
			delete(s.execCalls, callID)
		}
	}
	s.mu.Unlock()

	if reverted {
		s.commitProviderNotice("execution provider disconnected; reverting to local execution")
	}
	s.publishStatus()
}

// SelectExec switches the session's active execution provider to the client
// with the given id ("local" for the server process). The deselected client
// keeps its connection — it is simply no longer sent tool calls — so it can be
// selected again without reconnecting. Selecting the already-active provider
// is a no-op (no notice, no status broadcast). It returns an error when the id
// names no connected client.
func (s *Session) SelectExec(id string) error {
	s.mu.Lock()
	if id != "local" {
		c, ok := s.execClients[id]
		if !ok || !c.connected {
			s.mu.Unlock()
			return fmt.Errorf("unknown execution provider %q", id)
		}
	}
	if s.activeExec == id {
		s.mu.Unlock()
		return nil
	}
	var c *execClient
	var ctx *api.ExecContext
	if id == "local" {
		lctx := s.localContextLocked()
		ctx = &lctx
	} else {
		c = s.execClients[id]
		ctx = c.ctx
	}
	s.activeExec = id
	s.mu.Unlock()

	// The notice names the provider the user picked (not "disconnected", which
	// is a failure-mode revert) so the model sees the environment change.
	name := "local"
	if c != nil {
		name = c.name
	}
	s.commitProviderNotice(s.providerNotice("execution provider", name, ctx))
	s.publishStatus()
	return nil
}

// providerNotice builds the role-"system" notice committed when the active
// execution provider changes: a short historical marker naming where commands
// now run. The full environment (system details, files, skills) is injected
// fresh with each request via the provider's Environment(), so the notice
// stays concise.
func (s *Session) providerNotice(prefix, name string, ctx *api.ExecContext) string {
	label := execLabel(ctx)
	if name == "" && label == "" {
		return prefix
	}
	var b strings.Builder
	b.WriteString(prefix)
	b.WriteString(": ")
	if name != "" {
		b.WriteString(name)
		if label != "" {
			b.WriteString(" (")
			b.WriteString(label)
			b.WriteString(")")
		}
	} else {
		b.WriteString(label)
	}
	return b.String()
}

// execLabel renders a short "system @ cwd" label for a provider's reported
// context, with the OS detail (the parenthetical in exec.System()) trimmed,
// or "" when the context is empty. It is what the provider notices and the
// web picker show.
func execLabel(ctx *api.ExecContext) string {
	if ctx == nil {
		return ""
	}
	var b strings.Builder
	if ctx.System != "" {
		system := ctx.System
		if i := strings.Index(system, " ("); i > 0 {
			system = system[:i]
		}
		b.WriteString(system)
	}
	if ctx.CWD != "" {
		if b.Len() > 0 {
			b.WriteString(" @ ")
		}
		b.WriteString(ctx.CWD)
	}
	return b.String()
}

// nextClientID returns a fresh id for a client that didn't send its own.
func (s *Session) nextClientID() string {
	s.clientSeq++
	return fmt.Sprintf("client-%d", s.clientSeq)
}

// commitProviderNotice commits a role-"system" message so the model sees the
// execution environment change on its next turn. It is best-effort and silent:
// a persist failure during shutdown (the database is already closed) is
// expected, and the notice is informational rather than a turn, so it is not
// worth a log line.
func (s *Session) commitProviderNotice(content string) {
	m := llm.SystemMessage(content)
	_, _ = s.commitEnv(api.Envelope{Kind: api.KindMessage, Message: &m})
}

// publishStatus broadcasts the session's current execution provider status as a
// live envelope, so clients can show a real-time indicator without polling.
func (s *Session) publishStatus() {
	status := s.ExecStatus()
	s.publish(api.Envelope{Kind: api.KindExecStatus, ExecStatus: &status})
}

// ExecResult pipes a client's streamed output into the in-flight call's writer,
// completing the agent's ReadAll.
func (s *Session) ExecResult(callID string, body io.Reader) error {
	s.mu.Lock()
	call, ok := s.execCalls[callID]
	delete(s.execCalls, callID)
	s.mu.Unlock()
	if !ok {
		return fmt.Errorf("unknown exec call %q", callID)
	}
	_, err := io.Copy(call.pw, body)
	_ = call.pw.Close()
	return err
}

// dropCall removes an in-flight call and closes its pipe with err.
func (s *Session) dropCall(callID string, pw io.WriteCloser) {
	s.mu.Lock()
	delete(s.execCalls, callID)
	s.mu.Unlock()
	_ = pw.Close()
}

// newCallID returns a fresh call id for this session.
func (s *Session) newCallID() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.execSeq++
	return fmt.Sprintf("call-%d", s.execSeq)
}
