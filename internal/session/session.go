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
	"fmt"
	"sort"
	"strings"
	"sync"

	"porter/internal/agent"
	"porter/internal/api"
	"porter/internal/llm"
	"porter/internal/render"
	"porter/internal/tools"
)

// logEventsMax bounds the ring buffer the bus replays to a resubscribing
// client. A subscriber older than this is told to resync from history instead.
const logEventsMax = 256

// subBuffer is the per-subscriber event buffer. It must comfortably fit a
// turn's envelopes as an HTTP handler drains it.
const subBuffer = 64

// TurnMeta is the committed metadata for one completed turn: its token usage
// and any error. It is stored alongside history so the view endpoint can render
// usage without reading the bus log.
type TurnMeta struct {
	TurnID int64
	Input  int
	Output int
	Error  string
}

// Store owns the live set of sessions. It serializes id allocation and lookup,
// but each session serializes its own history writes. Sessions' schedules run on
// the store's context, which should outlive any single request (the session
// server's lifetime).
type Store struct {
	ctx      context.Context
	mu       sync.Mutex
	next     int
	sessions map[string]*Session
}

// NewStore returns an empty store. Pass a context to bind the session schedules'
// lifetime to it; otherwise they live for the process lifetime.
func NewStore(ctxs ...context.Context) *Store {
	ctx := context.Background()
	if len(ctxs) > 0 {
		ctx = ctxs[0]
	}
	return &Store{ctx: ctx, sessions: map[string]*Session{}}
}

// Create makes a new session and starts its turn scheduler.
func (st *Store) Create(client *llm.Client) *Session {
	st.mu.Lock()
	st.next++
	id := fmt.Sprintf("sess-%d", st.next)
	s := newSession(id, client, nil)
	st.sessions[id] = s
	st.mu.Unlock()
	go s.loop(st.ctx)
	return s
}

// Get returns the session with the given id and whether it exists.
func (st *Store) Get(id string) (*Session, bool) {
	st.mu.Lock()
	defer st.mu.Unlock()
	ses, ok := st.sessions[id]
	return ses, ok
}

// Session is one conversation. All state is guarded by mu; the scheduler
// goroutine owns turn execution.
type Session struct {
	id     string
	client *llm.Client

	mu      sync.Mutex
	js      tools.Provider
	logSeq  uint64
	turn    int64
	history []llm.ChatMessage
	turns   []TurnMeta
	log     []api.Envelope
	subs    []chan api.Envelope

	// Execution provider connection state.
	execCh    chan api.ExecRequest
	execCalls map[string]*execCall
	execSeq   int

	// In-flight tool runs keyed by call_id. The agent's live envelopes are
	// recorded here (started/delta/result) so a client that connects or
	// reconnects mid-run can reconstruct running blocks from the authoritative
	// server state instead of missing everything that streamed while it was
	// away. Runs are removed when their terminal result arrives.
	runs map[string]*toolRun

	queue chan string
}

// toolRun is the accumulated state of one in-flight tool execution: what was
// called, when it started (server clock), and the partial output produced so
// far.
type toolRun struct {
	callID    string
	name      string
	args      string
	startedAt int64
	output    strings.Builder
}

func newSession(id string, client *llm.Client, js tools.Provider) *Session {
	if js == nil {
		js = tools.NewDispatcher()
	}
	return &Session{
		id:     id,
		client: client,
		js:     js,
		queue:  make(chan string, 16),
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
// execution when none has been registered.
func (s *Session) provider() tools.Provider {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.js == nil {
		s.js = tools.NewDispatcher()
	}
	return s.js
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
	s.commit(llm.UserMessage(content))

	done := api.Envelope{Kind: api.KindTurnDone, TurnID: turnID}
	res, err := agent.RunTurn(ctx, s.client, s.snapshot(), s.provider(), s.emitLive, func(m llm.ChatMessage) {
		s.commit(m)
	})
	if err != nil {
		done.Error = err.Error()
	} else {
		done.Input = res.Usage.Input
		done.Output = res.Usage.Output
	}
	s.endTurn(done)
}

// Enqueue adds a user message for the scheduler to process. It never blocks a
// single message; the scheduler paces turns serially.
func (s *Session) Enqueue(content string) {
	s.queue <- content
}

// ID returns the session's id.
func (s *Session) ID() string { return s.id }

// Snapshot returns the authoritative committed history and the bus position to
// resume from.
func (s *Session) Snapshot() api.SessionHistory {
	s.mu.Lock()
	defer s.mu.Unlock()
	return api.SessionHistory{
		History: append([]llm.ChatMessage{}, s.history...),
		Seq:     s.logSeq,
	}
}

// Turns returns metadata for all completed turns (usage and errors). The slice
// is a defensive copy.
func (s *Session) Turns() []TurnMeta {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]TurnMeta{}, s.turns...)
}

// commit appends m to the authoritative history, stamps it on the bus log with
// the next position, and publishes it to every subscriber. For assistant
// messages it also pre-renders the content to HTML (markdown) and carries it on
// the envelope, so the SSE client renders the committed copy identically to the
// /view endpoint rather than approximating it client-side.
func (s *Session) commit(m llm.ChatMessage) uint64 {
	env := api.Envelope{Kind: api.KindMessage, Message: &m}
	if m.Role == "assistant" && m.Content != "" {
		env.MessageHTML = render.Markdown(m.Content)
	}
	s.mu.Lock()
	s.logSeq++
	env.Seq = s.logSeq
	s.history = append(s.history, m)
	s.bufferLocked(env)
	subs := s.subs
	s.mu.Unlock()
	s.sendTo(subs, env)
	return env.Seq
}

// endTurn logs and broadcasts a turn-completion marker so late subscribers see
// it even if the turn finished before they connected. It also records the
// turn's metadata (usage, error) so the view endpoint can render it without
// reading the bus log.
func (s *Session) endTurn(env api.Envelope) {
	s.mu.Lock()
	s.logSeq++
	env.Seq = s.logSeq
	s.turns = append(s.turns, TurnMeta{
		TurnID: env.TurnID,
		Input:  env.Input,
		Output: env.Output,
		Error:  env.Error,
	})
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
		s.runs[env.ToolCallID] = &toolRun{
			callID:    env.ToolCallID,
			name:      env.Name,
			args:      env.Arguments,
			startedAt: env.StartedAt,
		}
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
	case api.KindToolResult:
		s.mu.Lock()
		delete(s.runs, env.ToolCallID)
		s.mu.Unlock()
	}
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
	return s.turn
}

// snapshot returns a defensive copy of the committed history.
func (s *Session) snapshot() []llm.ChatMessage {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]llm.ChatMessage{}, s.history...)
}
