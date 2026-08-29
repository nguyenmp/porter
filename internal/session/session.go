// Package session owns conversation state. It is the single writer for a
// session's history and its event log: user messages, every message the agent
// finalizes, and turn-completion markers are committed here under one lock, so
// the committed history and the event bus share the same append order and the
// same monotonic sequence. Clients are stateless — they poll history, subscribe
// to the bus, and send commands; the session paces turn execution through a
// queue.
package session

import (
	"context"
	"errors"
	"fmt"
	"log"
	"sort"
	"strings"
	"sync"
	"time"

	"porter/internal/agent"
	"porter/internal/api"
	"porter/internal/db"
	"porter/internal/llm"
	"porter/internal/mcp"
	"porter/internal/render"
	"porter/internal/tools"
)

// logEventsMax bounds the ring buffer the bus replays to a resubscribing
// client. A subscriber older than this is told to resync from history instead.
const logEventsMax = 256

// subBuffer is the per-subscriber event buffer. It must comfortably fit a
// turn's envelopes as an HTTP handler drains it.
const subBuffer = 64

// Persister is the durable backing for a session's committed state. The store
// writes every commit here first (fail-fast, so the database can never fall
// behind what the bus has told subscribers) and reads history, the session
// list, and startup state back from it. In-memory state is limited to
// transient or rebuildable caches: the replay ring buffer, the queue, and
// in-flight tool runs. *db.DB implements this interface.
type Persister interface {
	// CreateSession inserts a session and returns its numeric row id, the
	// numeric part of the "session_<id>" identifier the store exposes.
	CreateSession(createdAt int64) (int64, error)
	// AppendMessage writes one committed message with its bus position (seq).
	AppendMessage(sessionID int64, m db.Message) error
	// AppendQuery writes one per-request usage/error record, independent of the
	// message stream (a failed request produces no message). It is what lets a
	// reload rebuild per-turn token totals and failed-turn errors.
	AppendQuery(sessionID int64, q db.Query) error
	// LoadSession returns a session's full persisted state: its creation time,
	// every committed message in seq order, every query record, and the highest
	// seq written.
	LoadSession(id int64) (db.Session, error)
	// ListSessions returns every session, newest first, with the raw content of
	// each session's first user message (the sidebar preview source).
	ListSessions() ([]db.Summary, error)
}

// Store owns the live set of sessions. Sessions are created by persisting a
// row (the database is the id allocator and the source of truth), and their
// committed history and bus position come from the persister. Each session
// serializes its own history writes. Sessions' schedules run on the store's
// context, which should outlive any single request (the session server's
// lifetime).
type Store struct {
	ctx      context.Context
	cancel   context.CancelFunc
	mu       sync.Mutex
	persist  Persister
	// hub serves the session's MCP tools (FindMCP, CallMCP). It is shared
	// across sessions: the registry is loaded once at server startup from
	// porter.mcp.json. Nil when no MCP hub was configured.
	hub      *mcp.Hub
	sessions map[string]*Session
}

// NewStore returns an empty store backed by persist, serving MCP tools from
// hub (nil for none). Pass a context to bind the session schedules' lifetime
// to it; otherwise they live for the process lifetime.
func NewStore(persist Persister, hub *mcp.Hub, ctxs ...context.Context) *Store {
	ctx := context.Background()
	if len(ctxs) > 0 {
		ctx = ctxs[0]
	}
	ctx, cancel := context.WithCancel(ctx)
	return &Store{ctx: ctx, cancel: cancel, persist: persist, hub: hub, sessions: map[string]*Session{}}
}

// Create makes a new session and starts its turn scheduler. The session is
// persisted before it exists in memory: the row id becomes the numeric part of
// the public "session_<id>" identifier, and an insert failure is returned so
// the caller can surface it as a server fault.
func (st *Store) Create(client *llm.Client) (*Session, error) {
	now := time.Now().UnixMilli()
	dbID, err := st.persist.CreateSession(now)
	if err != nil {
		return nil, fmt.Errorf("create session: %w", err)
	}
	id := fmt.Sprintf("session_%d", dbID)
	s := newSession(id, client, nil, st.persist, dbID, now, st.hub)
	st.mu.Lock()
	st.sessions[id] = s
	st.mu.Unlock()
	go s.loop(st.ctx)
	return s, nil
}

// Get returns the session with the given id and whether it exists.
func (st *Store) Get(id string) (*Session, bool) {
	st.mu.Lock()
	defer st.mu.Unlock()
	ses, ok := st.sessions[id]
	return ses, ok
}

// List returns a summary of every session, newest first, read from the
// persister. It is what the web sidebar renders; because every session lives in
// the persister, the list survives restarts unchanged. The preview is derived
// from the persisted first user message.
func (st *Store) List() []api.SessionSummary {
	summaries, err := st.persist.ListSessions()
	if err != nil {
		log.Printf("list sessions: %v", err)
		return []api.SessionSummary{}
	}
	out := make([]api.SessionSummary, 0, len(summaries))
	for _, s := range summaries {
		out = append(out, api.SessionSummary{
			ID:        fmt.Sprintf("session_%d", s.ID),
			CreatedAt: s.CreatedAt,
			Preview:   previewOf(s.FirstUser),
		})
	}
	return out
}

// Load rebuilds the live session set from the persister: one Session per row,
// reconstructing each session's history, replay buffer, and bus position. It is
// called once at startup, before the server starts serving.
func (st *Store) Load(client *llm.Client) error {
	summaries, err := st.persist.ListSessions()
	if err != nil {
		return fmt.Errorf("load sessions: %w", err)
	}
	for _, sm := range summaries {
		ps, err := st.persist.LoadSession(sm.ID)
		if err != nil {
			return fmt.Errorf("load session %d: %w", sm.ID, err)
		}
		id := fmt.Sprintf("session_%d", ps.ID)
		s := newSession(id, client, nil, st.persist, ps.ID, ps.CreatedAt, st.hub)
		s.rebuildFromPersisted(ps)
		st.mu.Lock()
		st.sessions[id] = s
		st.mu.Unlock()
		go s.loop(st.ctx)
	}
	return nil
}

// Close stops every session scheduler and closes the persister. The server
// calls it on shutdown; a restarted server reopens the same database.
func (st *Store) Close() {
	st.cancel()
	if c, ok := st.persist.(interface{ Close() error }); ok {
		_ = c.Close()
	}
}

// previewOf shortens the first user message to a single-line sidebar label:
// the first line, trimmed, truncated to previewMax runes with an ellipsis.
func previewOf(s string) string {
	const previewMax = 80
	s = strings.TrimSpace(s)
	if i := strings.IndexByte(s, '\n'); i >= 0 {
		s = s[:i]
	}
	if r := []rune(s); len(r) > previewMax {
		return string(r[:previewMax]) + "…"
	}
	return s
}

// Session is one conversation. All state is guarded by mu; the scheduler
// goroutine owns turn execution.
//
// Committed state lives in the persister (the source of truth): history,
// creation time, and bus position are read back from it, so a restart loses
// nothing. What stays in memory is transient or rebuildable: the replay ring
// buffer (a cache rebuilt from the persister at startup), the turn/queue/exec
// plumbing, and in-flight tool runs. The session is the single writer for its
// persister rows, so memory and the database cannot diverge.
type Session struct {
	id        string
	dbID      int64 // numeric row id in the persister (the "n" of session_<n>)
	client    *llm.Client
	createdAt int64 // creation time from the persister; orders the session list

	mu      sync.Mutex
	persist Persister
	js      tools.Provider
	hub     *mcp.Hub
	logSeq  uint64
	turn    int64
	running bool // a turn has started but its completion marker is not yet committed
	// totalCached/totalUncached/totalOutput are the session's accumulated
	// token usage across completed turns, kept in memory and rebuilt from
	// the persister at startup (a restarted server has no in-flight turns,
	// so every persisted query came from a completed turn). This is what
	// the web UI shows as the session total below the input box; a running
	// turn's usage joins only when its completion marker commits, on the
	// same write that broadcasts the marker's authoritative session total.
	totalCached   int
	totalUncached int
	totalOutput   int
	log           []api.Envelope
	subs          []chan api.Envelope

	// Execution provider connection state.
	execCh    chan api.ExecRequest
	execCalls map[string]*execCall
	execSeq   int
	// execCtx is the environment context the connected execution client
	// reported (system, working directory, files, skills). It drives both the
	// model context the remote provider injects and the load_skill tool
	// definitions it exposes. Nil until a client registers one.
	execCtx *api.ExecContext

	// In-flight tool runs keyed by call_id. The agent's live envelopes are
	// recorded here (started/delta/result) so a client that connects or
	// reconnects mid-run can reconstruct running blocks from the authoritative
	// server state instead of missing everything that streamed while it was
	// away. Runs are removed when their terminal result arrives.
	runs map[string]*toolRun

	// turnCancel aborts the currently running turn's context. It is set when a
	// turn starts (runTurn) and cleared when it ends; Stop() calls it to stop
	// the whole loop. Nil when the session is idle.
	turnCancel context.CancelFunc

	queue chan string
}

// toolRun is the accumulated state of one in-flight tool execution: what was
// called, when it started (server clock), the partial output produced so far,
// and the cancel function that aborts it (wired to the agent's per-run context
// when the run starts, so a Cancel in the UI can stop the running command).
type toolRun struct {
	callID    string
	name      string
	args      string
	startedAt int64
	output    strings.Builder
	cancel    func()
}

func newSession(id string, client *llm.Client, js tools.Provider, persist Persister, dbID int64, createdAt int64, hub *mcp.Hub) *Session {
	if js == nil {
		js = tools.NewDispatcher()
	}
	return &Session{
		id:        id,
		dbID:      dbID,
		client:    client,
		js:        js,
		persist:   persist,
		createdAt: createdAt,
		hub:       hub,
		queue:     make(chan string, 16),
	}
}

// SetProvider sets the execution provider this session runs tools with. It is
// guarded by mu so a connected client can take over execution mid-session
// without racing a running turn.
func (s *Session) SetProvider(js tools.Provider) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.js = js
}

// provider returns the current execution provider, defaulting to local
// execution when none has been registered. When the session has an MCP hub,
// the execution provider is wrapped in a composite that also exposes the hub
// tools (FindMCP, CallMCP); hub calls are served on the server and never
// cross the exec channel.
func (s *Session) provider() tools.Provider {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.js == nil {
		s.js = tools.NewDispatcher()
	}
	if s.hub == nil {
		return s.js
	}
	return &mcp.Composite{Exec: s.js, Hub: s.hub}
}

// loop is the turn scheduler. It consumes queued user messages one at a time,
// so only one turn runs per session and history writes are strictly ordered.
func (s *Session) loop(ctx context.Context) {
	for {
		select {
		case <-ctx.Done():
			return
		case content := <-s.queue:
			s.runTurn(ctx, content)
		}
	}
}

func (s *Session) runTurn(ctx context.Context, content string) {
	turnID := s.nextTurn()
	// The turn runs under its own cancellable context so the user can stop the
	// whole loop (the Stop button): cancelling it aborts the model stream
	// (committing any partial reply, marked interrupted) and any running tool
	// (whose per-call context derives from it). The cancel func is cleared when
	// the turn ends, so Stop() only reaches a live turn.
	turnCtx, turnCancel := context.WithCancel(ctx)
	s.mu.Lock()
	s.turnCancel = turnCancel
	s.mu.Unlock()
	defer func() {
		turnCancel()
		s.mu.Lock()
		s.turnCancel = nil
		s.mu.Unlock()
	}()
	// Committing the user message marks the start of this turn. Carry the
	// remaining queue depth so subscribers can show how many turns are still
	// waiting behind this one (the loop already pulled this message out of the
	// queue, so QueueDepth is exactly the backlog).
	msg := llm.UserMessage(content)
	env := api.Envelope{Kind: api.KindMessage, Message: &msg, Queue: s.QueueDepth()}
	turnSeq, err := s.commitEnv(env)
	if err != nil {
		// The turn never started: its user message could not be persisted.
		// Report the failure on the bus and move on to the next queued message.
		s.endTurn(api.Envelope{Kind: api.KindTurnDone, TurnID: turnID, Error: err.Error()})
		return
	}
	// The user message's bus seq is the turn's identity: every query of this
	// turn is persisted under it, and turn_completed carries it so a live
	// client can dedup the turn's outcome against what /view already rendered.

	done := api.Envelope{Kind: api.KindTurnDone, TurnID: turnID, TurnSeq: turnSeq}
	// Persist each request's usage/error as the agent produces it (the query's
	// origin), so turns are rebuildable from the database on a reload.
	onQuery := func(q agent.Query) error { return s.commitQuery(turnSeq, q) }
	res, err := agent.RunTurn(turnCtx, s.client, s.snapshot(), s.provider(), s.emitLive, func(m llm.ChatMessage) error {
		return s.commit(m)
	}, agent.RunHooks{OnRunStarted: s.onRunStarted, OnQuery: onQuery})
	if err != nil {
		switch {
		case errors.Is(err, agent.ErrTurnStopped):
			// The user stopped the turn: end it with a stopped marker (not an
			// error). Any partial reply is already committed, marked
			// interrupted, so history is transparent.
			done.Stopped = true
		case errors.Is(err, agent.ErrToolCancelled):
			// A tool cancelled by the user is a clean stop, not a failure: the
			// partial result is already committed and the tool_cancelled
			// envelope went out, so end the turn without an error marker.
		default:
			done.Error = err.Error()
		}
	}
	// Usage rides on every completion path (a stopped turn carries its partial
	// usage), so the live marker matches what the persisted footer derives on
	// reload.
	done.CachedInput = res.Usage.CachedInput
	done.UncachedInput = res.Usage.UncachedInput
	done.Output = res.Usage.Output
	s.endTurn(done)
}

// Enqueue adds a user message for the scheduler to process. It never blocks a
// single message; the scheduler paces turns serially.
func (s *Session) Enqueue(content string) {
	s.queue <- content
}

// ID returns the session's id.
func (s *Session) ID() string { return s.id }

// QueueDepth returns how many user messages are still waiting in the queue
// behind the turn currently running (0 when idle). The server is the single
// writer, so this is the authoritative backlog; it is what the web client shows
// as a "N queued" indicator.
func (s *Session) QueueDepth() int {
	return len(s.queue)
}

// Totals returns the session's accumulated token usage across completed turns
// (cached/uncached input and output). It seeds the web UI's session-total line
// at page render; live turn_completed markers carry the authoritative updated
// totals, so this is only the starting point, not the live source of truth.
func (s *Session) Totals() (cached, uncached, output int) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.totalCached, s.totalUncached, s.totalOutput
}

// Running reports whether a turn is currently in progress: one has started (its
// user message committed) but its completion marker has not been committed yet.
func (s *Session) Running() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.running
}

// Snapshot returns the authoritative committed history and the bus position to
// resume from. Both come from the persister in a single read: History is every
// committed message and Seq is the highest committed message seq, so a client
// that replays the bus with since=seq gets exactly the messages newer than the
// returned history — no gap, no overlap.
func (s *Session) Snapshot() api.SessionHistory {
	msgs, maxSeq := s.loadMessages()
	return api.SessionHistory{History: msgs, Seq: maxSeq}
}

// snapshot returns the committed history as a slice, seeding the agent's view
// for one turn. The agent loop reads history exactly once per turn (then
// extends its own copy), and the persister is the source of truth, so this
// reads the database rather than a mirror.
func (s *Session) snapshot() []llm.ChatMessage {
	msgs, _ := s.loadMessages()
	return msgs
}

// loadMessages reads the session's committed history and its highest message
// seq from the persister in one call.
func (s *Session) loadMessages() ([]llm.ChatMessage, uint64) {
	ps, err := s.persist.LoadSession(s.dbID)
	if err != nil {
		log.Printf("load session %s: %v", s.id, err)
		return nil, 0
	}
	msgs := make([]llm.ChatMessage, 0, len(ps.Messages))
	for _, m := range ps.Messages {
		msgs = append(msgs, m.ChatMessage)
	}
	return msgs, ps.MaxSeq
}

// Turn is a derived partition of a session's message stream: one user message
// plus every message and query that follow it until the next user message (or
// the end of history). Turns are not stored — they are computed from the
// persisted messages and queries. Input/Output are the sum of the turn's
// queries' usage; Error is set when any of its queries failed (a failed query
// ends the turn, so at most one error per turn, on its final query).
type Turn struct {
	UserSeq       uint64 // bus seq of the user message that started the turn
	CachedInput   int    // prompt tokens served from cache, summed across the turn's queries
	UncachedInput int    // prompt tokens read fresh (cache misses), summed across the turn's queries
	Output        int    // completion tokens, summed across the turn's queries
	Error         string
	// Stopped reports that the user aborted the turn (the Stop button): any of
	// its queries is marked stopped. A stopped turn is not an error.
	Stopped bool
}

// Input returns the turn's total input tokens (cached + uncached).
func (t Turn) Input() int { return t.CachedInput + t.UncachedInput }

// Persisted returns the session's full persisted state (messages with their
// bus seqs, query records, max seq) in a single read. It backs the /view
// render, which needs per-message seqs to place turn footers.
func (s *Session) Persisted() (db.Session, error) {
	return s.persist.LoadSession(s.dbID)
}

// DeriveTurns partitions a persisted session into its turns, in stream order.
// Each turn begins at a user message and runs to the next user message;
// usage is summed across the turn's queries and the first failed query's error
// marks the turn. A user message with no queries still opens a turn (e.g. one
// that failed before any request ran), so callers can place a footer — the
// turn simply has nothing to show.
func DeriveTurns(ps db.Session) []Turn {
	byUser := make(map[uint64]*Turn, len(ps.Queries))
	for _, q := range ps.Queries {
		t := byUser[q.TurnSeq]
		if t == nil {
			t = &Turn{UserSeq: q.TurnSeq}
			byUser[q.TurnSeq] = t
		}
		t.CachedInput += q.CachedInput
		t.UncachedInput += q.UncachedInput
		t.Output += q.Output
		if q.Stopped {
			t.Stopped = true
		}
		if t.Error == "" && q.Error != "" {
			t.Error = q.Error
		}
	}
	var out []Turn
	for _, m := range ps.Messages {
		if m.Role != "user" {
			continue
		}
		t := byUser[m.Seq]
		if t == nil {
			t = &Turn{UserSeq: m.Seq}
		}
		out = append(out, *t)
	}
	return out
}

// commit writes m to the persister first (fail-fast: a message that cannot be
// persisted aborts the turn), then stamps it on the bus log with the next
// position and publishes it to every subscriber. For assistant messages it also
// pre-renders the content to HTML (markdown) and carries it on the envelope, so
// the SSE client renders the committed copy identically to the /view endpoint
// rather than approximating it client-side.
func (s *Session) commit(m llm.ChatMessage) error {
	env := api.Envelope{Kind: api.KindMessage, Message: &m}
	if m.Role == "assistant" && m.Content != "" {
		env.MessageHTML = render.Markdown(m.Content)
	}
	_, err := s.commitEnv(env)
	return err
}

// commitEnv is the tail of commit: stamp the envelope with the next bus
// position, write its message to the persister, log it for replay, and
// broadcast it to every subscriber. It returns the assigned bus position,
// which is what identifies the committed message (and, for a turn's opening
// user message, the turn itself). Splitting it out lets turn start commit a
// user message with extra metadata (the queue depth) without duplicating the
// plumbing.
//
// The persister write happens before the in-memory and bus updates so a crash
// never loses a committed message; on error the caller aborts the turn (the
// reserved seq is skipped, which is harmless since seqs are monotonic). The
// scheduler goroutine is the session's single writer, so logSeq is only ever
// touched here, in endTurn, and at startup load.
func (s *Session) commitEnv(env api.Envelope) (uint64, error) {
	s.mu.Lock()
	s.logSeq++
	next := s.logSeq
	env.Seq = next
	s.mu.Unlock()

	if err := s.persist.AppendMessage(s.dbID, db.Message{Seq: next, ChatMessage: *env.Message}); err != nil {
		return 0, fmt.Errorf("persist message: %w", err)
	}

	s.mu.Lock()
	s.bufferLocked(env)
	subs := s.subs
	s.mu.Unlock()
	s.sendTo(subs, env)
	return next, nil
}

// commitQuery persists one request's usage/error under the given turn, at the
// query's origin. It is the OnQuery hook the agent is given; a persist failure
// aborts the turn (fail-fast, like a message that cannot be persisted), since
// the database must never fall behind what the bus has told subscribers.
func (s *Session) commitQuery(turnSeq uint64, q agent.Query) error {
	row := db.Query{TurnSeq: turnSeq, Idx: q.Idx, CachedInput: q.CachedInput, UncachedInput: q.UncachedInput, Output: q.Output, Stopped: q.Stopped}
	if q.Err != nil {
		row.Error = q.Err.Error()
	}
	if err := s.persist.AppendQuery(s.dbID, row); err != nil {
		return fmt.Errorf("persist query: %w", err)
	}
	return nil
}

// rebuildFromPersisted reconstructs the in-memory replay cache for a session
// loaded from the persister at startup: the ring buffer gets one message
// envelope per committed message (in seq order, with assistant HTML re-rendered
// from the stored markdown) and logSeq resumes at the highest message seq so
// the next commit takes the next position. Turn-completion markers are not
// persisted, so the rebuilt buffer contains only message commits and the
// running flag starts false.
func (s *Session) rebuildFromPersisted(ps db.Session) {
	s.logSeq = ps.MaxSeq
	for _, m := range ps.Messages {
		env := api.Envelope{Kind: api.KindMessage, Message: &m.ChatMessage, Seq: m.Seq}
		if m.Role == "assistant" && m.Content != "" {
			env.MessageHTML = render.Markdown(m.Content)
		}
		s.bufferLocked(env)
	}
	// No turns are in flight at startup, so every persisted query belongs to a
	// completed turn: sum it into the running session total so the session
	// total survives restarts.
	for _, q := range ps.Queries {
		s.totalCached += q.CachedInput
		s.totalUncached += q.UncachedInput
		s.totalOutput += q.Output
	}
}

// endTurn logs and broadcasts a turn-completion marker so late subscribers see
// it even if the turn finished before they connected, and clears the running
// flag. The marker is live-only (it is not persisted); a restarted session
// starts with a clean bus containing only its committed messages.
func (s *Session) endTurn(env api.Envelope) {
	s.mu.Lock()
	s.logSeq++
	env.Seq = s.logSeq
	// A completed turn's usage joins the session running total, and the new
	// total rides on the completion marker: events are ordered, so a client
	// that always sets (never adds) its session-total display from the marker's
	// totals converges on the true session total even across replays, and the
	// baseline rendered at page load can never be double-counted.
	s.totalCached += env.CachedInput
	s.totalUncached += env.UncachedInput
	s.totalOutput += env.Output
	env.TotalCachedInput = s.totalCached
	env.TotalUncachedInput = s.totalUncached
	env.TotalOutput = s.totalOutput
	s.running = false
	s.bufferLocked(env)
	subs := s.subs
	s.mu.Unlock()
	s.sendTo(subs, env)
}

// publish broadcasts a live envelope (an LLM event or a system-side fact like a
// tool result) to every subscriber without logging it: it is real-time only and
// not replayed to late subscribers.
func (s *Session) publish(env api.Envelope) {
	s.mu.Lock()
	subs := s.subs
	s.mu.Unlock()
	s.sendTo(subs, env)
}

// emitLive records tool-run state for reconnect reconstruction, then broadcasts
// the envelope live. It is the emit sink RunTurn is given, so every tool
// envelope the agent produces updates the in-flight run registry before going
// out to subscribers.
func (s *Session) emitLive(env api.Envelope) {
	s.trackToolRun(env)
	s.publish(env)
}

// trackToolRun updates the in-flight run registry from the live tool envelopes:
// tool_started creates a run, tool_result_delta appends to its partial output,
// and the terminal tool_result removes it. Other envelope kinds are ignored.
func (s *Session) trackToolRun(env api.Envelope) {
	switch env.Kind {
	case api.KindToolStarted:
		if env.ToolCallID == "" {
			return
		}
		s.mu.Lock()
		if s.runs == nil {
			s.runs = make(map[string]*toolRun)
		}
		// onRunStarted registered the run's cancel func before the tool began;
		// keep it when the start marker (emitted after the tool starts) fills in
		// the identity, so a Cancel click between registration and start is not
		// lost.
		run := &toolRun{
			callID:    env.ToolCallID,
			name:      env.Name,
			args:      env.Arguments,
			startedAt: env.StartedAt,
		}
		if existing, ok := s.runs[env.ToolCallID]; ok && existing.cancel != nil {
			run.cancel = existing.cancel
		}
		s.runs[env.ToolCallID] = run
		s.mu.Unlock()
	case api.KindToolResultDelta:
		if env.ToolCallID == "" {
			return
		}
		s.mu.Lock()
		if r, ok := s.runs[env.ToolCallID]; ok {
			r.output.WriteString(env.Delta)
		}
		s.mu.Unlock()
	case api.KindToolResult, api.KindToolCancelled:
		// Both terminal kinds end the run: it leaves the in-flight set whether
		// it finished normally or was cancelled.
		s.mu.Lock()
		delete(s.runs, env.ToolCallID)
		s.mu.Unlock()
	}
}

// onRunStarted is the agent's per-run-start hook: it records the run's cancel
// function in the in-flight registry before the tool starts, so a user can
// cancel it (the UI Cancel button) from the moment it is registered. The cancel
// aborts the run's context, which for a local tool kills its process group and
// for a remote tool stops the wait on the stream and signals the client.
func (s *Session) onRunStarted(callID string, cancel func()) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.runs == nil {
		s.runs = make(map[string]*toolRun)
	}
	r, ok := s.runs[callID]
	if !ok {
		r = &toolRun{callID: callID}
		s.runs[callID] = r
	}
	r.cancel = cancel
}

// CancelRun cancels the in-flight tool run with the given call id. For a local
// run this kills the command's process group; for a remote run it cancels the
// agent's wait on the stream (closing the exec pipe) and tells the connected
// execution client to stop its running command. It returns an error for an
// unknown call id (the run already finished, or never existed).
func (s *Session) CancelRun(callID string) error {
	s.mu.Lock()
	run, ok := s.runs[callID]
	if !ok || run == nil {
		s.mu.Unlock()
		return fmt.Errorf("unknown run %q", callID)
	}
	cancel := run.cancel
	// If a remote execution client is connected, it runs the command and must
	// be told to stop it. The agent runs tools sequentially, so at most one
	// command is in flight and a bare Cancel=true reaches it.
	var execCh chan api.ExecRequest
	if s.execCh != nil {
		execCh = s.execCh
	}
	s.mu.Unlock()

	if cancel != nil {
		cancel()
	}
	if execCh != nil {
		select {
		case execCh <- api.ExecRequest{Cancel: true}:
		default:
			// Client's request buffer is full; the run ends anyway once the
			// agent's stream closes (remoteProvider closes its pipe on cancel),
			// so don't block a cancel on a slow client.
		}
	}
	return nil
}

// Stop aborts the currently running turn: it cancels the turn's context, which
// stops the model stream (committing any partial reply, marked interrupted)
// and any running tool (whose per-call context derives from the turn's), and
// ends the turn with a stopped marker. It is the backend for the UI's Stop
// button. It returns an error when no turn is running (idle, or already
// finished), so a double-click or a late click is rejected rather than
// cancelling the next turn.
func (s *Session) Stop() error {
	s.mu.Lock()
	cancel := s.turnCancel
	s.mu.Unlock()
	if cancel == nil {
		return errors.New("no turn running")
	}
	cancel()
	return nil
}

// Runs returns a snapshot of the session's in-flight tool runs, ordered by
// start time. It is what lets a client reconstruct running blocks after a
// reconnect mid-run.
func (s *Session) Runs() []api.RunInfo {
	s.mu.Lock()
	defer s.mu.Unlock()
	out := make([]api.RunInfo, 0, len(s.runs))
	for _, r := range s.runs {
		out = append(out, api.RunInfo{
			CallID:    r.callID,
			Name:      r.name,
			Arguments: r.args,
			StartedAt: r.startedAt,
			Output:    r.output.String(),
		})
	}
	sort.Slice(out, func(i, j int) bool { return out[i].StartedAt < out[j].StartedAt })
	return out
}

// bufferLocked appends env to the replayable log, evicting the oldest when it
// exceeds the bound. mu must be held.
func (s *Session) bufferLocked(env api.Envelope) {
	s.log = append(s.log, env)
	if len(s.log) > logEventsMax {
		s.log = s.log[len(s.log)-logEventsMax:]
	}
}

// sendTo delivers env to each subscriber channel without blocking or holding
// the session lock. A subscriber whose buffer is full is dropped rather than
// stalling every other subscriber and the scheduler; it recovers via resync on
// its next poll/connect.
func (s *Session) sendTo(subs []chan api.Envelope, env api.Envelope) {
	for _, sub := range subs {
		select {
		case sub <- env:
		default:
		}
	}
}

// From returns a channel of envelopes to a subscriber starting at bus position
// since. It synchronously replays buffered positions newer than since, then
// streams live events. Subscriber registration happens under the same lock as
// the replay, so nothing between the caller's snapshot and this subscription is
// lost or duplicated. If since is too far behind the buffer it emits a single
// resync envelope and stops — the caller must refetch history and resubscribe.
// Callers should cancel ctx to stop the subscription.
func (s *Session) From(ctx context.Context, since uint64) <-chan api.Envelope {
	out := make(chan api.Envelope, subBuffer)
	sub := make(chan api.Envelope, subBuffer)

	s.mu.Lock()
	if len(s.log) > 0 && since+1 < s.log[0].Seq {
		s.mu.Unlock()
		out <- api.Envelope{Kind: api.KindResync}
		close(out)
		return out
	}
	var replay []api.Envelope
	for _, env := range s.log {
		if env.Seq > since {
			replay = append(replay, env)
		}
	}
	s.subs = append(s.subs, sub)
	s.mu.Unlock()

	// Replay buffered positions first, then stream live. Both go through the
	// forwarder so a large backlog drains concurrently with the HTTP handler
	// instead of blocking here; registration above ensures no gap between the
	// caller's snapshot and this subscription.
	go func() {
		defer close(out)
		for _, env := range replay {
			out <- env
		}
		forward(ctx, out, sub, func() { s.detach(sub) })
	}()
	return out
}

// forward forwards envelopes from a subscriber channel to out until ctx is
// done, then drains anything already buffered so events published up to
// cancellation are never lost, and detaches the channel. It does not close out.
func forward(ctx context.Context, out chan<- api.Envelope, sub <-chan api.Envelope, detach func()) {
	for {
		select {
		case env := <-sub:
			out <- env
		case <-ctx.Done():
			for {
				select {
				case env := <-sub:
					out <- env
				default:
					detach()
					return
				}
			}
		}
	}
}

// detach removes a subscriber channel.
func (s *Session) detach(sub chan api.Envelope) {
	s.mu.Lock()
	defer s.mu.Unlock()
	for i, c := range s.subs {
		if c == sub {
			s.subs = append(s.subs[:i], s.subs[i+1:]...)
			return
		}
	}
}

func (s *Session) nextTurn() int64 {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.turn++
	s.running = true
	return s.turn
}
