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
	"porter/internal/exec"
	"porter/internal/humanize"
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
	// AppendVariant writes one completed humanize pass (a rewritten assistant
	// reply) for a committed message. Variants are derived data — view-only
	// attachments, never part of history or the model's context.
	AppendVariant(sessionID int64, v db.Variant) error
	// LoadSession returns a session's full persisted state: its creation time,
	// every committed message in seq order, every query record, and the highest
	// seq written.
	LoadSession(id int64) (db.Session, error)
	// ListSessions returns every session, newest first, with the raw content of
	// each session's first user message (the sidebar preview source).
	ListSessions() ([]db.Summary, error)
	// ArchiveSession marks a session archived at the given epoch-ms time,
	// folding it out of the active sidebar list.
	ArchiveSession(id int64, at int64) error
	// UnarchiveSession clears a session's archived flag, moving it back to the
	// active list.
	UnarchiveSession(id int64) error
	// RenameSession sets (or clears) a session's custom display name. Empty
	// clears it back to the preview fallback (the first user message). Names
	// are display-only: the id stays the session's identity.
	RenameSession(id int64, name string) error
}

// Store owns the live set of sessions. Sessions are created by persisting a
// row (the database is the id allocator and the source of truth), and their
// committed history and bus position come from the persister. Each session
// serializes its own history writes. Sessions' schedules run on the store's
// context, which should outlive any single request (the session server's
// lifetime).
type Store struct {
	ctx     context.Context
	cancel  context.CancelFunc
	mu      sync.Mutex
	persist Persister
	// hub serves the session's MCP tools (FindMCP, CallMCP). It is shared
	// across sessions: the registry is loaded once at server startup from
	// porter.mcp.json. Nil when no MCP hub was configured.
	hub      *mcp.Hub
	sessions map[string]*Session

	// Execution hosts: persistent agents (e.g. on a laptop) that can
	// provision execution contexts for new sessions. hosts maps every
	// registered host by id; pendingHostCtx holds a host's base context for
	// the window between its context POST and its RegisterHost, keyed by host
	// id. pending maps in-flight provisions by provider id, so both the host
	// channel (Provision) and the session's RegisterExec (ProvisionRegistered,
	// called by the server) can resolve them.
	hosts          map[string]*host
	pendingHostCtx map[string]*api.ExecContext
	pending        map[string]*pendingProvision
	hostSeq        int
	// sandboxes tracks, per session, the worktree sandbox a host provisioned
	// for it (recorded when the provider registers), so archiving the session
	// can release the sandbox.
	sandboxes map[string]*sandbox
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
	return &Store{
		ctx:       ctx,
		cancel:    cancel,
		persist:   persist,
		hub:       hub,
		sessions:  map[string]*Session{},
		hosts:     map[string]*host{},
		pending:   map[string]*pendingProvision{},
		sandboxes: map[string]*sandbox{},
	}
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
	s := newSession(id, client, nil, st.persist, dbID, now, 0, "", st.hub)
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
			ID:         fmt.Sprintf("session_%d", s.ID),
			CreatedAt:  s.CreatedAt,
			Name:       s.Name,
			Preview:    previewOf(s.FirstUser),
			ArchivedAt: s.ArchivedAt,
		})
	}
	return out
}

// Archive marks a session archived at the current time, folding it out of the
// active sidebar list and into the Archived folder (ordered most-recently
// archived first). The flag is persisted before memory is updated (fail-fast,
// matching Create), so the list and the session can never disagree. Archive is
// purely organizational: history, running turns, and the event bus are
// unaffected. It is idempotent for an already-archived session.
func (st *Store) Archive(id string) error {
	ses, ok := st.Get(id)
	if !ok {
		return db.ErrNotFound
	}
	now := time.Now().UnixMilli()
	if err := st.persist.ArchiveSession(ses.dbID, now); err != nil {
		return fmt.Errorf("archive session: %w", err)
	}
	ses.setArchived(now)
	return nil
}

// Unarchive clears a session's archived flag, moving it back to the active
// list. Persist first, then update memory, mirroring Archive. Idempotent.
func (st *Store) Unarchive(id string) error {
	ses, ok := st.Get(id)
	if !ok {
		return db.ErrNotFound
	}
	if err := st.persist.UnarchiveSession(ses.dbID); err != nil {
		return fmt.Errorf("unarchive session: %w", err)
	}
	ses.setArchived(0)
	return nil
}

// Rename sets (or clears) a session's custom display name, then broadcasts
// the change live so every open client updates in place. Empty clears the
// name back to the preview fallback. Names are display-only: the session id
// stays the identity, so renaming never affects links, exec, or the turn
// tree. Persist first, then memory, then the bus — a subscriber that handles
// session_renamed reads the new name. Renaming an archived session keeps it
// archived (only the name column is touched).
func (st *Store) Rename(id, name string) error {
	ses, ok := st.Get(id)
	if !ok {
		return db.ErrNotFound
	}
	if err := st.persist.RenameSession(ses.dbID, name); err != nil {
		return fmt.Errorf("rename session: %w", err)
	}
	ses.setName(name)
	ses.publish(api.Envelope{Kind: api.KindSessionRenamed, SessionName: name})
	return nil
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
		s := newSession(id, client, nil, st.persist, ps.ID, ps.CreatedAt, ps.ArchivedAt, ps.Name, st.hub)
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
	// archivedAt is the epoch-ms time the session was archived, or 0 when
	// active. It comes from the persister (survives restarts) and is what the
	// web sidebar uses to fold archived sessions into the Archived folder.
	archivedAt int64
	// name is the session's custom display name ("" = none; the sidebar and
	// chat header fall back to the first-message preview). It is display-only:
	// the id remains the identity, so renaming never affects links, exec, or
	// history.
	name string

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
	// liveSeq is the session's monotonic live-stream position, and liveTail
	// is the replayable tail of the in-flight assistant block: every KindLLM
	// envelope published since the last commit, in order, stamped with its
	// liveSeq. Committed state is not duplicated here — commitEnv clears the
	// tail — so the tail is bounded by the largest single block (the deltas of
	// one streaming reply, reasoning, or tool-call sequence), not by session
	// history or an arbitrary ring. It is what lets a subscriber that connects
	// mid-turn (or whose /view render wiped a live bubble) catch the current
	// stream: From replays it after the committed ring and before live events,
	// and Live exposes it for a fresh fetch. Live events other than KindLLM
	// (tool runs) are not tailed; they are already reconstructible from the
	// in-flight run registry (Runs).
	liveSeq  uint64
	liveTail []api.Envelope

	// Execution provider registry. execClients maps every provider that can
	// run this session's tools to its descriptor, keyed by id; activeExec is
	// the id of the active one ("local" for the server process, the default).
	// Remote clients are added on connect and removed on disconnect; a client
	// the user deselects stays connected (its ch is live) but receives no
	// tool calls until selected again. pendingCtx holds a client's reported
	// environment context for the window between its context POST (which
	// happens before the exec connection, see the REPL) and its RegisterExec,
	// keyed by client id ("" is the legacy slot for clients that don't
	// identify themselves).
	execClients map[string]*execClient
	activeExec  string
	pendingCtx  map[string]*api.ExecContext
	// local is the execution provider used when no remote client is active —
	// the server process running tools itself. It is built lazily from the
	// server's own discovered context (localProvider) so "local" is a
	// first-class provider in the selector; SetProvider replaces it (tests,
	// embedders).
	local tools.Provider
	// localCtx is the cached discovery of the server process's own
	// environment (system, cwd, files, skills), reported for the local
	// provider and the picker.
	localCtx  *api.ExecContext
	clientSeq int

	execCalls map[string]*execCall
	execSeq   int

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

	// notices carries execution-provider status changes awaiting commit
	// (connected/disconnected/selected). The HTTP handlers that observe the
	// exec connection — RegisterExec, UnregisterExec, SelectExec — only queue
	// here; the scheduler goroutine commits them at a turn boundary, keeping
	// the scheduler the session's single history writer. Committing from the
	// handler goroutine directly could persist a system message into the
	// middle of a turn's tool-call exchange — between an assistant tool_calls
	// message and its tool results — which DeepSeek rejects with
	// "insufficient tool messages following tool_calls message". Best-effort:
	// a full queue drops the notice (real-time status still streams via
	// KindExecStatus).
	notices chan string

	// humanize tracks in-flight plain-language passes. variantIdx is the next
	// pass index to allocate per message, seeded from the persisted count on
	// first use so a restarted server keeps numbering where it left off; the
	// counter is what makes concurrent passes (auto + manual) never collide on
	// an index. It is guarded by its own mutex so a pass running on a
	// background goroutine never contends with the scheduler's history lock.
	humanizeMu sync.Mutex
	variantIdx map[uint64]int
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

func newSession(id string, client *llm.Client, js tools.Provider, persist Persister, dbID int64, createdAt int64, archivedAt int64, name string, hub *mcp.Hub) *Session {
	// A nil js means no local provider was injected (the server's own
	// process); it is built lazily on first use from the discovered local
	// context, so "local" reports the same system/cwd/files/skills a remote
	// client would.
	return &Session{
		id:          id,
		dbID:        dbID,
		client:      client,
		local:       js,
		persist:     persist,
		createdAt:   createdAt,
		archivedAt:  archivedAt,
		name:        name,
		hub:         hub,
		execClients: map[string]*execClient{},
		activeExec:  "local",
		queue:       make(chan string, 16),
		notices:     make(chan string, 32),
		variantIdx:  map[uint64]int{},
	}
}

// SetProvider replaces the session's local execution provider — the one used
// when no remote client is active, and the default before any client
// connects. It is guarded by mu so a test or embedder can swap local
// execution mid-session without racing a running turn.
func (s *Session) SetProvider(js tools.Provider) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.local = js
}

// provider returns the current execution provider — routing to the active
// remote client when one is selected, else the local provider — wrapped in a
// composite that also exposes the MCP surface when any server is available:
// the server's own configured hub (served here, never crossing the exec
// channel) plus the MCP servers the active provider hosts (reported in its
// context; served on the provider's machine). When the active provider is
// local, only the server hub's servers are exposed.
func (s *Session) provider() tools.Provider {
	s.mu.Lock()
	defer s.mu.Unlock()
	js := s.activeProviderLocked()
	hub := s.hub
	var remote []api.MCPServer
	if c, ok := s.execClients[s.activeExec]; ok && c.connected && c.kind != "local" && c.ctx != nil {
		remote = c.ctx.MCPServers
	}
	if hub == nil && len(remote) == 0 {
		return js
	}
	if hub == nil {
		hub = mcp.New(nil)
	}
	return &mcp.Composite{Exec: js, Hub: hub, Remote: remote}
}

// activeProviderLocked returns the provider for the session's active
// execution client: a remoteProvider routing to that client when a remote is
// selected, else the local provider (the server process).
func (s *Session) activeProviderLocked() tools.Provider {
	if c, ok := s.execClients[s.activeExec]; ok && c.connected && c.kind != "local" {
		return &remoteProvider{sess: s}
	}
	if s.local == nil {
		ctx := s.localContextLocked()
		s.local = &localProvider{d: tools.NewDispatcherWithSkills(ctx.Skills), ctx: ctx}
	}
	return s.local
}

// localContextLocked returns the server process's own environment context,
// discovered once and cached. Discovery is best-effort: on failure an empty
// context is reported rather than failing the request.
func (s *Session) localContextLocked() api.ExecContext {
	if s.localCtx == nil {
		ctx, err := exec.Discover("")
		if err != nil {
			ctx = api.ExecContext{}
		}
		s.localCtx = &ctx
	}
	return *s.localCtx
}

// loop is the turn scheduler. It consumes queued user messages one at a time,
// so only one turn runs per session and history writes are strictly ordered.
// Provider notices are committed here too (see flushNotices), so every
// history write happens on this goroutine and a notice can never land inside
// a turn's tool-call exchange.
func (s *Session) loop(ctx context.Context) {
	for {
		select {
		case <-ctx.Done():
			return
		case content := <-s.queue:
			// Flush any notices queued before this turn so the provider
			// narrative lands at the boundary, not inside the turn.
			s.flushNotices()
			s.runTurn(ctx, content)
		case n := <-s.notices:
			// A provider change arrived (possibly mid-turn; it waits in the
			// channel until the turn ends, then commits at the boundary). The
			// receive already consumed n; flushNotices drains any that piled up
			// behind it and commits them together.
			s.flushNotices(n)
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

// Archived reports whether the session has been archived (folded out of the
// active sidebar list into the Archived folder). Chatting with an archived
// session unarchives it.
func (s *Session) Archived() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.archivedAt > 0
}

// setArchived updates the session's archived timestamp. at is epoch-ms; 0
// clears the flag (back to active).
func (s *Session) setArchived(at int64) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.archivedAt = at
}

// Name returns the session's custom display name ("" when none is set).
func (s *Session) Name() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.name
}

// setName updates the session's custom display name. It is called after the
// rename is persisted, so a subscriber that reads the name sees the new value.
func (s *Session) setName(name string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.name = name
}

// Preview returns the single-line preview of the session's first user message
// ("" when the session has no messages yet). It is the sidebar/header
// fallback when no custom name is set.
func (s *Session) Preview() string {
	ps, err := s.persist.LoadSession(s.dbID)
	if err != nil {
		log.Printf("load session %s: %v", s.id, err)
		return ""
	}
	for _, m := range ps.Messages {
		if m.Role == "user" {
			return previewOf(m.Content)
		}
	}
	return ""
}

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
	// The tool-output metadata rides on the committed envelope so the live UI
	// renders the same size/truncation/recall badge /view renders from the
	// persisted copy.
	env.ToolOutput = m.ToolOutput
	seq, err := s.commitEnv(env)
	if err != nil {
		return err
	}
	// Auto humanize: a long plain-prose assistant reply gets a plain-language
	// pass in the background, shown as a "Humanized" tab next to the original.
	// The pass runs off the turn queue (a fresh LLM request carrying just the
	// text), so it never blocks the turn or the next message; the scheduler
	// broadcasts variant_started here (same goroutine, so it always follows
	// the message commit), and the pass goroutine broadcasts variant_ready
	// with the rewrite when it finishes. Intermediate assistant messages that
	// carry tool calls are never humanized — only the final plain-text reply.
	if s.client != nil && m.Role == "assistant" && m.Content != "" && len(m.ToolCalls) == 0 && humanize.Should(m.Content) {
		s.startHumanize(seq, m.Content, -1)
	}
	return nil
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
	// A commit supersedes the in-flight block: the tail held only that block's
	// deltas (the agent streams one assistant message at a time, and any other
	// commit — user, tool result, system notice — lands at a boundary where
	// the tail is empty), so clear it here. The clear happens under the same
	// lock as bufferLocked, so From can never snapshot a commit alongside the
	// deltas it replaced.
	s.liveTail = nil
	subs := s.subs
	s.mu.Unlock()
	s.sendTo(subs, env)
	return next, nil
}

// humanizeContext returns the full prose transcript of the conversation
// leading up to the message at msgSeq — every user message and assistant
// reply with content, unbounded, so a humanize pass is grounded in what was
// asked without re-sending raw history (tool traffic, reasoning, and system
// notices are excluded; see humanize.Transcript). A read failure is
// best-effort: the pass runs without context rather than failing.
func (s *Session) humanizeContext(msgSeq uint64) string {
	ps, err := s.persist.LoadSession(s.dbID)
	if err != nil {
		log.Printf("humanize session %s: load context: %v", s.id, err)
		return ""
	}
	return humanize.Transcript(ps.Messages, msgSeq)
}

// startHumanize launches a background plain-language pass over content,
// attaching it to the committed assistant message at msgSeq. source is the
// variant index the pass chains from (-1 = the original message). The pass
// announces itself on the bus (variant_started) before it runs, so the UI can
// show a pending tab immediately; its terminal state is broadcast as
// variant_ready and persisted when done. A failure is not fatal: the variant
// is persisted with its error so the tab renders as failed and "+" can retry
// from the previous good version.
func (s *Session) startHumanize(msgSeq uint64, content string, source int) {
	// A session without an LLM client (tests, embedders) cannot run a pass;
	// the auto path already guards before calling, this covers the manual
	// endpoint defensively.
	if s.client == nil {
		log.Printf("humanize session %s message %d: no LLM client", s.id, msgSeq)
		return
	}
	// Index allocation: the automatic first pass (source -1) claims index 0 —
	// the message just committed, so no variant can exist for it yet. Manual
	// passes seed the counter from the persisted count on first use, so a
	// restarted server keeps numbering where it left off and concurrent passes
	// never collide.
	s.humanizeMu.Lock()
	idx := s.variantIdx[msgSeq]
	s.variantIdx[msgSeq] = idx + 1
	s.humanizeMu.Unlock()
	if source >= 0 && idx == 0 {
		// Manual pass on a message with no in-memory counter yet: seed from
		// the persisted count (a restarted server, or a message humanized
		// before this session loaded).
		idx = s.seedVariantIdx(msgSeq)
	}
	s.commitVariant(api.Envelope{
		Kind: api.KindVariantStarted,
		Variant: &api.Variant{
			MessageSeq: msgSeq,
			Index:      idx,
			Source:     source,
		},
	}, nil)
	go s.runHumanize(msgSeq, idx, source, content)
}

// seedVariantIdx returns the next free pass index for a message by counting
// its persisted variants. It is only used when the in-memory counter is empty
// for a message that already has variants (e.g. after a restart), so it is a
// rare full-session read, not per-pass overhead.
func (s *Session) seedVariantIdx(msgSeq uint64) int {
	ps, err := s.persist.LoadSession(s.dbID)
	n := 0
	if err == nil {
		for _, v := range ps.Variants {
			if v.MessageSeq == msgSeq && v.Index >= n {
				n = v.Index + 1
			}
		}
	}
	s.humanizeMu.Lock()
	if cur, ok := s.variantIdx[msgSeq]; ok && cur > n {
		n = cur
	}
	s.variantIdx[msgSeq] = n + 1
	s.humanizeMu.Unlock()
	return n
}

// runHumanize is the background pass itself: one plain-language rewrite over
// the given content, then a variant_ready commit with the result (or the
// error). It runs on its own goroutine with its own context, so a slow
// provider never blocks the turn queue or the bus.
func (s *Session) runHumanize(msgSeq uint64, idx, source int, content string) {
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()
	// Ground the rewrite in what was asked: a compact transcript of the
	// conversation up to (not including) the message being humanized. The
	// pass runs off the turn queue, so it reads the persisted history itself.
	out, err := humanize.Rewrite(ctx, s.client, s.humanizeContext(msgSeq), content)
	if err != nil {
		log.Printf("humanize session %s message %d: %v", s.id, msgSeq, err)
		s.commitVariant(api.Envelope{
			Kind: api.KindVariant,
			Variant: &api.Variant{
				MessageSeq:    msgSeq,
				Index:         idx,
				Source:        source,
				Error:         err.Error(),
				PromptVersion: humanize.PromptVersion,
			},
		}, &db.Variant{
			MessageSeq:    msgSeq,
			Index:         idx,
			Source:        source,
			Error:         err.Error(),
			PromptVersion: humanize.PromptVersion,
			CreatedAt:     time.Now().UnixMilli(),
		})
		return
	}
	s.commitVariant(api.Envelope{
		Kind: api.KindVariant,
		Variant: &api.Variant{
			MessageSeq:    msgSeq,
			Index:         idx,
			Source:        source,
			Content:       out,
			HTML:          render.Markdown(out),
			PromptVersion: humanize.PromptVersion,
		},
	}, &db.Variant{
		MessageSeq:    msgSeq,
		Index:         idx,
		Source:        source,
		Content:       out,
		PromptVersion: humanize.PromptVersion,
		CreatedAt:     time.Now().UnixMilli(),
	})
}

// commitVariant stamps a variant envelope with the next bus position, buffers
// it for replay, and broadcasts it. Terminal variants (v != nil) are persisted
// first, fail-fast like a message commit; variant_started (v == nil) is
// live-and-replay-only — in-flight state is not persisted, so a pass that
// never finishes (e.g. a restart) simply leaves no row and /view renders no
// tab. Variants ride the same bus counter as messages so their order relative
// to commits is well-defined and a reconnect replays a pass's start marker.
// Unlike commitEnv, the live tail is NOT cleared: a variant commit is
// unrelated to the in-flight assistant block and must not wipe it.
func (s *Session) commitVariant(env api.Envelope, v *db.Variant) (uint64, error) {
	s.mu.Lock()
	s.logSeq++
	next := s.logSeq
	env.Seq = next
	s.mu.Unlock()

	if v != nil {
		if err := s.persist.AppendVariant(s.dbID, *v); err != nil {
			return 0, fmt.Errorf("persist variant: %w", err)
		}
	}

	s.mu.Lock()
	s.bufferLocked(env)
	subs := s.subs
	s.mu.Unlock()
	s.sendTo(subs, env)
	return next, nil
}

// HumanizeMessage starts a manual plain-language pass on a committed assistant
// message, chaining from its latest variant (or the original message when none
// exist). It is the "+" button's backend: the pass runs in the background like
// the automatic one. It returns an error when the message is not a humanizable
// assistant reply (so the handler can 404).
func (s *Session) HumanizeMessage(msgSeq uint64) error {
	ps, err := s.persist.LoadSession(s.dbID)
	if err != nil {
		return err
	}
	var content string
	var found bool
	for _, m := range ps.Messages {
		if m.Seq == msgSeq && m.Role == "assistant" && m.Content != "" && len(m.ToolCalls) == 0 {
			content = m.Content
			found = true
			break
		}
	}
	if !found {
		return fmt.Errorf("no humanizable assistant message at seq %d", msgSeq)
	}
	// Chain from the latest variant that produced content; a failed pass has
	// none, so it falls back to the previous good version (or the original).
	source := -1
	for _, v := range ps.Variants {
		if v.MessageSeq == msgSeq && v.Content != "" && v.Index > source {
			source = v.Index
			content = v.Content
		}
	}
	s.startHumanize(msgSeq, content, source)
	return nil
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
		// The tool-output metadata rides on the replayed commit too, so a
		// reconnecting client renders the badge exactly as /view does.
		env.ToolOutput = m.ToolOutput
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
	// Defensive clear: a turn that ends without its final message committing
	// (e.g. a tool cancelled mid-run) must not leave a stale tail that the
	// next block's replay would prepend to.
	s.liveTail = nil
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
	// Live LLM envelopes join the tail so a late subscriber can catch the
	// current stream (see liveTail). Stamping under the same lock as the
	// append keeps liveSeq and the tail consistent with what From/Live
	// snapshot, so a subscriber sees every envelope exactly once: either in
	// the replayed tail or as a live event, never both and never neither.
	if env.Kind == api.KindLLM {
		s.liveSeq++
		env.LiveSeq = s.liveSeq
		s.liveTail = append(s.liveTail, env)
	}
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
	// If a remote execution client is active, it runs the command and must be
	// told to stop it. The agent runs tools sequentially, so at most one
	// command is in flight and a bare Cancel=true reaches it.
	var execCh chan api.ExecRequest
	if c, ok := s.execClients[s.activeExec]; ok && c.ch != nil {
		execCh = c.ch
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

// Live returns the session's in-flight LLM stream tail: every live LLM
// envelope since the last commit, in order, plus the live position of the
// newest one. It is the stream counterpart of Runs — what lets a client whose
// live view was wiped (a /view swap, a missed reconnect window) re-seed the
// streaming assistant bubble from the server's authoritative partial stream
// instead of waiting for the turn's next commit.
func (s *Session) Live() (uint64, []api.Envelope) {
	s.mu.Lock()
	defer s.mu.Unlock()
	out := make([]api.Envelope, len(s.liveTail))
	copy(out, s.liveTail)
	return s.liveSeq, out
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
	// The live tail rides on the same subscription, between the committed
	// replay and live events: a subscriber joining mid-turn gets the
	// in-flight block from its first delta. Snapshotting it under the same
	// lock as registration guarantees the ordering — an envelope published
	// before this point is in the tail, one published after arrives live, so
	// nothing is lost or duplicated. (On resync the tail is skipped: the
	// caller reloads, and the fresh page reconnects with a current seq.)
	tail := append([]api.Envelope(nil), s.liveTail...)
	s.subs = append(s.subs, sub)
	s.mu.Unlock()

	// Replay buffered positions first (committed, then the live tail), then
	// stream live. Both go through the forwarder so a large backlog drains
	// concurrently with the HTTP handler instead of blocking here;
	// registration above ensures no gap between the caller's snapshot and
	// this subscription.
	go func() {
		defer close(out)
		for _, env := range replay {
			out <- env
		}
		for _, env := range tail {
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
