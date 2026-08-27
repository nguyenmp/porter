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

// remoteProvider routes a session's tool calls to a connected execution client
// (e.g. the REPL on the host) instead of running them locally. It implements
// tools.Provider so agent.RunTurn is unchanged: Run blocks until the client
// streams the command's output back, returning that output as a stream the
// agent reads to completion. The tools it exposes (including load_skill) and
// the environment context it reports both come from the client's registered
// context.
type remoteProvider struct {
	sess *Session
}

// Defs returns the tools the connected client can run: shell plus load_skill
// when the client reported skills. It is read from the client's registered
// context, so it stays in sync with what the provider actually loaded.
func (r *remoteProvider) Defs() []llm.Tool {
	s := r.sess
	s.mu.Lock()
	skills := make([]api.Skill, 0)
	if s.execCtx != nil {
		skills = append(skills, s.execCtx.Skills...)
	}
	s.mu.Unlock()
	return tools.DefsForSkills(skills)
}

// Environment returns the connected client's reported environment context as a
// system message, or "" when none has been registered.
func (r *remoteProvider) Environment() string {
	s := r.sess
	s.mu.Lock()
	ctx := s.execCtx
	s.mu.Unlock()
	if ctx == nil {
		return ""
	}
	return exec.SystemMessage(*ctx)
}

// Run registers a tool call with the connected execution client and returns a
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
	ch := s.execCh
	s.mu.Unlock()

	if ch == nil {
		s.dropCall(callID, pw)
		return nil, errors.New("no execution client connected")
	}
	select {
	case ch <- api.ExecRequest{CallID: callID, Name: name, Arguments: string(args)}:
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

// execCall tracks one in-flight tool call awaiting the client's streamed result.
type execCall struct {
	pw *io.PipeWriter
}

// SetExecContext stores the environment context the connected execution client
// reported. It is registered separately from the exec connection so the server
// can hold it even while the client's registration is in flight; the remote
// provider exposes it as the model context and the load_skill tool definitions.
func (s *Session) SetExecContext(ctx api.ExecContext) {
	s.mu.Lock()
	s.execCtx = &ctx
	s.mu.Unlock()
}

// ExecStatus returns the session's current execution provider status: whether
// a remote client is connected, its kind, and the context it reported.
func (s *Session) ExecStatus() api.ExecStatus {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.execCh == nil {
		return api.ExecStatus{Kind: "local"}
	}
	return api.ExecStatus{Connected: true, Kind: "remote", Context: s.execCtx}
}

// RegisterExec sets the session's execution client and points the session's
// provider at it. Called when a client connects to GET /api/sessions/{id}/exec.
// It commits a system notice so the model sees the environment change and
// broadcasts the new status.
func (s *Session) RegisterExec(ch chan api.ExecRequest) {
	s.mu.Lock()
	s.execCh = ch
	s.js = &remoteProvider{sess: s}
	ctx := s.execCtx
	s.mu.Unlock()

	notice := "execution provider connected"
	if ctx != nil && (ctx.System != "" || ctx.CWD != "") {
		// A short historical marker; the full environment (system details,
		// files, skills) is injected fresh with each request via the provider's
		// Environment(), so the notice stays concise.
		system := ctx.System
		if i := strings.Index(system, " ("); i > 0 {
			system = system[:i]
		}
		notice = fmt.Sprintf("execution provider connected: %s @ %s", system, ctx.CWD)
	}
	s.commitProviderNotice(notice)
	s.publishStatus()
}

// UnregisterExec clears the execution client, closes any in-flight calls so the
// agent does not hang, and reverts to local execution. It commits a system
// notice and broadcasts the new status only when a remote provider was actually
// registered, so repeated disconnects don't spam history.
func (s *Session) UnregisterExec() {
	s.mu.Lock()
	if s.execCh == nil {
		s.mu.Unlock()
		return
	}
	s.execCh = nil
	for id, c := range s.execCalls {
		_ = c.pw.CloseWithError(errors.New("execution client disconnected"))
		delete(s.execCalls, id)
	}
	s.js = tools.NewDispatcher()
	s.mu.Unlock()

	s.commitProviderNotice("execution provider disconnected; reverting to local execution")
	s.publishStatus()
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
