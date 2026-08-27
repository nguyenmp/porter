package session

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"porter/internal/agent"
	"porter/internal/api"
	"porter/internal/config"
	"porter/internal/db"
	"porter/internal/llm"
	"porter/internal/tools"
)

// commitN appends n distinct user messages and returns their seqs.
func commitN(t *testing.T, s *Session, prefix string, n int) {
	t.Helper()
	for i := 1; i <= n; i++ {
		if err := s.commit(llm.UserMessage(fmt.Sprintf("%s%d", prefix, i))); err != nil {
			t.Fatalf("commit %s%d: %v", prefix, i, err)
		}
	}
}

func TestCommitAppendsHistoryAndSeq(t *testing.T) {
	s := newTestSession(t, "s")
	commitN(t, s, "m", 3)
	snap := s.Snapshot()
	if snap.Seq != 3 {
		t.Errorf("seq = %d, want 3", snap.Seq)
	}
	if len(snap.History) != 3 {
		t.Errorf("history len = %d, want 3", len(snap.History))
	}
	if snap.History[0].Role != "user" || snap.History[0].Content != "m1" {
		t.Errorf("history[0] = %+v, want user m1", snap.History[0])
	}
}

func TestReplayOnlyAfterSince(t *testing.T) {
	s := newTestSession(t, "s")
	commitN(t, s, "m", 5)

	ctx, cancel := context.WithCancel(context.Background())
	stream := s.From(ctx, 3)
	// Replay is pushed into the stream synchronously; cancelling then draining
	// yields exactly the entries newer than since (seq 4 and 5).
	cancel()
	var seqs []int
	for env := range stream {
		if env.Kind == api.KindMessage {
			seqs = append(seqs, int(env.Seq))
		}
	}
	if len(seqs) != 2 || seqs[0] != 4 || seqs[1] != 5 {
		t.Errorf("replay after since=3 = %v, want [4 5]", seqs)
	}
}

// TestResubscribeContinuity is the reconnect race: subscribe from a snapshot's
// seq, then commit new messages. Everything committed after the snapshot must
// arrive with no gap and no overlap.
func TestResubscribeContinuity(t *testing.T) {
	s := newTestSession(t, "s")
	commitN(t, s, "base", 10)
	snap := s.Snapshot()
	if snap.Seq != 10 {
		t.Fatalf("snapshot seq = %d, want 10", snap.Seq)
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	stream := s.From(ctx, snap.Seq)

	var seqs []int
	done := make(chan struct{})
	go func() {
		defer close(done)
		for env := range stream {
			if env.Kind == api.KindMessage {
				seqs = append(seqs, int(env.Seq))
			}
		}
	}()

	commitN(t, s, "new", 5)

	cancel()
	<-done

	if len(seqs) != 5 {
		t.Fatalf("got %d committed seqs %v, want 5 (%d..%d)", len(seqs), seqs, 11, 15)
	}
	for i, got := range seqs {
		if want := 11 + i; got != want {
			t.Errorf("seq[%d] = %d, want %d", i, got, want)
		}
	}
}

func TestResyncWhenBehindBuffer(t *testing.T) {
	s := newTestSession(t, "s")
	commitN(t, s, "m", logEventsMax+5) // evicts the oldest entries

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	stream := s.From(ctx, 1) // behind the ring's oldest
	var got []api.Envelope
	for env := range stream {
		got = append(got, env)
	}
	if len(got) != 1 || got[0].Kind != api.KindResync {
		t.Fatalf("expected one resync envelope, got %+v", got)
	}
}

// TestLateSubscriberSeesTurnDone ensures a subscriber that connects after a turn
// already finished still sees the terminal marker via replay, so it is not left
// waiting for a live event it will never get.
func TestLateSubscriberSeesTurnDone(t *testing.T) {
	s := newTestSession(t, "s")
	s.commit(llm.UserMessage("hi"))
	s.commit(llm.AssistantMessage("done", "", nil))
	s.endTurn(api.Envelope{Kind: api.KindTurnDone, TurnID: 1})

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	stream := s.From(ctx, 0)
	var sawTurnDone bool
	for env := range stream {
		if env.Kind == api.KindTurnDone {
			sawTurnDone = true
			break
		}
	}
	if !sawTurnDone {
		t.Errorf("late subscriber never saw turn_completed")
	}
}

// TestLargeReplayDrains ensures a backlog larger than the channel buffer is
// replayed to a draining subscriber rather than deadlocking.
func TestLargeReplayDrains(t *testing.T) {
	s := newTestSession(t, "s")
	commitN(t, s, "m", 200) // > subBuffer

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	var count int
	for env := range s.From(ctx, 0) {
		if env.Kind == api.KindMessage {
			count++
			if count == 200 {
				cancel() // stop the live stream once the backlog is drained
			}
		}
	}
	if count != 200 {
		t.Errorf("replayed %d messages, want 200", count)
	}
}

// TestEnqueueRunsTurn drives the scheduler end to end against a real LLM
// client and a fake upstream that replies plainly.
func TestEnqueueRunsTurn(t *testing.T) {
	llmSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w,
			`data: {"choices":[{"delta":{"content":"done"},"finish_reason":"stop"}],"usage":{"prompt_tokens":2,"completion_tokens":3}}`+"\n\n"+
				`data: [DONE]`+"\n")
	}))
	defer llmSrv.Close()

	client := llm.NewClient(config.Config{BaseURL: llmSrv.URL + "/v1", Model: "m", APIKey: "k"}, nil)
	nctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	st := NewStore(memPersister(t), nctx)
	s, err := st.Create(client)
	if err != nil {
		t.Fatalf("Create: %v", err)
	}
	s.Enqueue("hello")

	deadline := time.After(5 * time.Second)
	var snap api.SessionHistory
	for {
		snap = s.Snapshot()
		if len(snap.History) >= 2 && snap.History[1].Role == "assistant" {
			break
		}
		select {
		case <-deadline:
			t.Fatalf("timed out waiting for turn; history = %+v", snap.History)
		case <-time.After(10 * time.Millisecond):
		}
	}
	if snap.History[0].Content != "hello" || snap.History[1].Content != "done" {
		t.Errorf("history = %+v, want user 'hello' then assistant 'done'", snap.History)
	}
}

// TestRunningAndQueueDepth covers the status-indicator primitives directly:
// a fresh session is idle with an empty queue, and enqueued messages are
// visible via QueueDepth without waiting for a turn to start.
func TestRunningAndQueueDepth(t *testing.T) {
	s := newTestSession(t, "s")
	if s.Running() {
		t.Errorf("fresh session should not report a running turn")
	}
	if d := s.QueueDepth(); d != 0 {
		t.Errorf("fresh session queue depth = %d, want 0", d)
	}
	for i := 0; i < 3; i++ {
		s.Enqueue(fmt.Sprintf("m%d", i))
	}
	if d := s.QueueDepth(); d != 3 {
		t.Errorf("queue depth after 3 enqueues = %d, want 3", d)
	}
}

func TestSetProviderDefaultsToLocal(t *testing.T) {
	st := NewStore(memPersister(t))
	s, err := st.Create(nil)
	if err != nil {
		t.Fatalf("Create: %v", err)
	}

	p := s.provider()
	if _, ok := p.(*tools.Dispatcher); !ok {
		t.Errorf("default provider = %T, want *tools.Dispatcher", p)
	}

	registered := &tools.Dispatcher{}
	s.SetProvider(registered)
	if got := s.provider(); got != registered {
		t.Errorf("provider after SetProvider = %T, want registered", got)
	}
}

// newTestSession returns a Session backed by a real in-memory SQLite database,
// so commit/Snapshot/From exercise the same persisted path the server runs
// (DB-first writes, DB-backed reads) without touching disk.
func newTestSession(t *testing.T, id string) *Session {
	t.Helper()
	d := memPersister(t)
	rowID, err := d.CreateSession(time.Now().UnixMilli())
	if err != nil {
		t.Fatalf("create session row: %v", err)
	}
	return newSession(id, nil, nil, d, rowID, time.Now().UnixMilli())
}

// memPersister opens a fresh in-memory SQLite database as a Persister.
func memPersister(t *testing.T) Persister {
	t.Helper()
	d, err := db.Open(":memory:")
	if err != nil {
		t.Fatalf("open in-memory db: %v", err)
	}
	t.Cleanup(func() { d.Close() })
	return d
}
func TestTrackToolRunsForReconnect(t *testing.T) {
	s := newTestSession(t, "s")

	// No runs yet.
	if got := s.Runs(); len(got) != 0 {
		t.Fatalf("Runs before any tool = %+v, want empty", got)
	}

	// A run starts (server clock), emits partial output, then another run starts.
	s.trackToolRun(api.Envelope{Kind: api.KindToolStarted, ToolCallID: "call-1", Name: "shell", Arguments: `{"command":"echo hi"}`, StartedAt: 1000})
	s.trackToolRun(api.Envelope{Kind: api.KindToolResultDelta, ToolCallID: "call-1", Delta: "partial line\n"})
	s.trackToolRun(api.Envelope{Kind: api.KindToolResultDelta, ToolCallID: "call-1", Delta: "more line\n"})
	s.trackToolRun(api.Envelope{Kind: api.KindToolStarted, ToolCallID: "call-2", Name: "shell", Arguments: `{"command":"sleep 5"}`, StartedAt: 2000})

	runs := s.Runs()
	if len(runs) != 2 {
		t.Fatalf("Runs = %+v, want 2 in-flight", runs)
	}
	// Ordered by start time: call-1 first, call-2 second.
	if runs[0].CallID != "call-1" || runs[1].CallID != "call-2" {
		t.Errorf("run order = %s, %s; want call-1, call-2", runs[0].CallID, runs[1].CallID)
	}
	// Accumulated partial output is authoritative.
	if runs[0].Output != "partial line\nmore line\n" {
		t.Errorf("run call-1 output = %q, want both deltas", runs[0].Output)
	}
	if runs[0].StartedAt != 1000 || runs[1].StartedAt != 2000 {
		t.Errorf("started clocks = %d, %d; want 1000, 2000", runs[0].StartedAt, runs[1].StartedAt)
	}
	if runs[0].Name != "shell" || runs[0].Arguments == "" {
		t.Errorf("run call-1 identity = %+v, want shell + args", runs[0])
	}

	// The terminal result removes the run (it is now committed, not in-flight).
	s.trackToolRun(api.Envelope{Kind: api.KindToolResult, ToolCallID: "call-1", Result: "partial line\nmore line\nexit code: 0\n"})
	runs = s.Runs()
	if len(runs) != 1 || runs[0].CallID != "call-2" {
		t.Fatalf("Runs after call-1 finished = %+v, want only call-2", runs)
	}

	// Non-tool envelopes and unknown call ids are ignored safely.
	s.trackToolRun(api.Envelope{Kind: api.KindLLM})
	s.trackToolRun(api.Envelope{Kind: api.KindToolResultDelta, ToolCallID: "nope", Delta: "x"})
	if len(s.Runs()) != 1 {
		t.Errorf("Runs after no-ops = %+v, want still 1", s.Runs())
	}
}

// TestCancelRunCancelsInFlightTool verifies the session's cancel endpoint: it
// calls the run's cancel func (the agent's per-run context), and a subsequent
// tool_cancelled envelope (which the agent emits) removes the run from the
// in-flight set. Unknown call ids are rejected.
func TestCancelRunCancelsInFlightTool(t *testing.T) {
	s := newTestSession(t, "s")

	// Register the run's cancel func (what the agent's OnRunStarted hook does)
	// and the start marker (what trackToolRun does when the tool begins).
	var cancelled bool
	s.onRunStarted("call-1", func() { cancelled = true })
	s.trackToolRun(api.Envelope{Kind: api.KindToolStarted, ToolCallID: "call-1", Name: "shell", Arguments: `{"command":"sleep 5"}`, StartedAt: 1000})
	if len(s.Runs()) != 1 {
		t.Fatalf("runs before cancel = %+v, want 1 in-flight", s.Runs())
	}

	if err := s.CancelRun("call-1"); err != nil {
		t.Fatalf("CancelRun: %v", err)
	}
	if !cancelled {
		t.Errorf("run's cancel func was not called")
	}

	// The agent responds to cancellation by emitting tool_cancelled, which
	// removes the run from the in-flight set (like tool_result does).
	s.trackToolRun(api.Envelope{Kind: api.KindToolCancelled, ToolCallID: "call-1"})
	if got := s.Runs(); len(got) != 0 {
		t.Errorf("runs after cancellation = %+v, want empty", got)
	}

	// Cancelling an unknown/finished run is an error (the UI hides the button
	// once the run ends, so this is defensive).
	if err := s.CancelRun("nope"); err == nil {
		t.Errorf("CancelRun(unknown) error = nil, want error")
	}
}

// TestToolStartKeepsRegisteredCancel covers the ordering race between the
// agent's OnRunStarted hook (registers the cancel func) and the tool_started
// envelope (fills in identity): the start marker must not wipe the cancel.
func TestToolStartKeepsRegisteredCancel(t *testing.T) {
	s := newTestSession(t, "s")

	var cancelled bool
	s.onRunStarted("call-1", func() { cancelled = true })
	// The start marker arrives after onRunStarted; the cancel must survive it.
	s.trackToolRun(api.Envelope{Kind: api.KindToolStarted, ToolCallID: "call-1", Name: "shell", Arguments: `{"command":"echo hi"}`, StartedAt: 1})

	if err := s.CancelRun("call-1"); err != nil {
		t.Fatalf("CancelRun: %v", err)
	}
	if !cancelled {
		t.Errorf("cancel func lost between onRunStarted and tool_started")
	}
}

// TestDeriveTurnsAggregatesQueries verifies the derived-turn model: usage is
// stored per query (the origin) and Turns sums each turn's queries, while a
// failed query's error marks its turn. Two turns, the first with two queries
// (one of which failed), the second with one successful query.
func TestDeriveTurnsAggregatesQueries(t *testing.T) {
	s := newTestSession(t, "s")

	// Turn 1: user message, then two queries — one succeeds (usage), one fails.
	commitN(t, s, "turn", 1)
	ps, err := s.Persisted()
	if err != nil {
		t.Fatalf("Persisted: %v", err)
	}
	turn1 := ps.Messages[0].Seq

	if err := s.commitQuery(turn1, agentQuery(0, 2, 3, nil)); err != nil {
		t.Fatalf("commitQuery turn1 q0: %v", err)
	}
	if err := s.commitQuery(turn1, agentQuery(1, 4, 5, fmt.Errorf("rate limit"))); err != nil {
		t.Fatalf("commitQuery turn1 q1: %v", err)
	}

	// Turn 2: user message, one successful query.
	commitN(t, s, "turn", 1)
	ps, err = s.Persisted()
	if err != nil {
		t.Fatalf("Persisted: %v", err)
	}
	turn2 := ps.Messages[len(ps.Messages)-1].Seq
	if err := s.commitQuery(turn2, agentQuery(0, 8, 13, nil)); err != nil {
		t.Fatalf("commitQuery turn2 q0: %v", err)
	}

	ps, err = s.Persisted()
	if err != nil {
		t.Fatalf("Persisted: %v", err)
	}
	turns := DeriveTurns(ps)
	if len(turns) != 2 {
		t.Fatalf("turns = %+v, want 2", turns)
	}
	// Turn 1: usage sums across its two queries; the failed query marks it.
	if turns[0].UserSeq != turn1 || turns[0].Input != 6 || turns[0].Output != 8 {
		t.Errorf("turn 1 = %+v, want user %d, usage 6/8", turns[0], turn1)
	}
	if !strings.Contains(turns[0].Error, "rate limit") {
		t.Errorf("turn 1 error = %q, want the failed query's error", turns[0].Error)
	}
	// Turn 2: just its one query's usage, no error.
	if turns[1].UserSeq != turn2 || turns[1].Input != 8 || turns[1].Output != 13 || turns[1].Error != "" {
		t.Errorf("turn 2 = %+v, want user %d, usage 8/13, no error", turns[1], turn2)
	}
}

// agentQuery builds an agent.Query for commitQuery.
func agentQuery(idx, input, output int, err error) agent.Query {
	return agent.Query{Idx: idx, Input: input, Output: output, Err: err}
}
