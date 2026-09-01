package session

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"porter/internal/agent"
	"porter/internal/api"
	"porter/internal/codec"
	"porter/internal/config"
	"porter/internal/db"
	"porter/internal/humanize"
	"porter/internal/llm"
	"porter/internal/mcp"
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

// TestLiveTailReplayedToLateSubscriber is the mid-turn reload case: a
// subscriber that connects after the stream started must receive the in-flight
// block's deltas (the live tail) in order, with monotonic live positions, so it
// catches up immediately instead of waiting for the turn's commit.
func TestLiveTailReplayedToLateSubscriber(t *testing.T) {
	s := newTestSession(t, "s")
	if err := s.commit(llm.UserMessage("hi")); err != nil {
		t.Fatalf("commit user: %v", err)
	}

	// The turn streams: two content deltas, then a reasoning delta.
	s.publish(api.Envelope{Kind: api.KindLLM, Event: &codec.Event{Type: codec.TypeMessageDelta, Delta: "Hel"}})
	s.publish(api.Envelope{Kind: api.KindLLM, Event: &codec.Event{Type: codec.TypeMessageDelta, Delta: "lo"}})
	s.publish(api.Envelope{Kind: api.KindLLM, Event: &codec.Event{Type: codec.TypeReasoningDelta, Reasoning: "think"}})

	ctx, cancel := context.WithCancel(context.Background())
	stream := s.From(ctx, 1) // the user commit is the last committed seq
	cancel()
	var got []api.Envelope
	for env := range stream {
		got = append(got, env)
	}
	if len(got) != 3 {
		t.Fatalf("late subscriber got %d envelopes %+v, want the 3 tail deltas", len(got), got)
	}
	if got[0].LiveSeq != 1 || got[1].LiveSeq != 2 || got[2].LiveSeq != 3 {
		t.Errorf("live_seq = %d,%d,%d; want 1,2,3", got[0].LiveSeq, got[1].LiveSeq, got[2].LiveSeq)
	}
	if got[0].Event.Delta != "Hel" || got[1].Event.Delta != "lo" || got[2].Event.Reasoning != "think" {
		t.Errorf("tail order/content wrong: %+v", got)
	}
}

// TestLiveTailThenLiveContinuity is the no-gap/no-dup property: a subscriber
// that connects mid-block gets the tail, and deltas published afterwards arrive
// live — each envelope exactly once, in order.
func TestLiveTailThenLiveContinuity(t *testing.T) {
	s := newTestSession(t, "s")
	if err := s.commit(llm.UserMessage("hi")); err != nil {
		t.Fatalf("commit user: %v", err)
	}
	s.publish(api.Envelope{Kind: api.KindLLM, Event: &codec.Event{Type: codec.TypeMessageDelta, Delta: "Hel"}})

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	stream := s.From(ctx, 1)

	var mu sync.Mutex
	var got []api.Envelope
	done := make(chan struct{})
	go func() {
		defer close(done)
		for env := range stream {
			mu.Lock()
			got = append(got, env)
			mu.Unlock()
		}
	}()
	waitFor := func(n int) bool {
		t.Helper()
		deadline := time.Now().Add(2 * time.Second)
		for {
			mu.Lock()
			c := len(got)
			mu.Unlock()
			if c >= n {
				return true
			}
			if time.Now().After(deadline) {
				return false
			}
			time.Sleep(5 * time.Millisecond)
		}
	}

	if !waitFor(1) {
		t.Fatal("tail delta never arrived")
	}
	// A delta published after subscription must arrive live, after the tail.
	s.publish(api.Envelope{Kind: api.KindLLM, Event: &codec.Event{Type: codec.TypeMessageDelta, Delta: "lo"}})
	if !waitFor(2) {
		t.Fatal("live delta never arrived")
	}
	cancel()
	<-done

	if len(got) != 2 || got[0].LiveSeq != 1 || got[1].LiveSeq != 2 {
		t.Fatalf("got %d envelopes %+v, want exactly tail(1) then live(2)", len(got), got)
	}
}

// TestLiveTailClearedOnCommit ensures a commit supersedes the in-flight block:
// the tail is dropped the moment its message commits, so a subscriber that
// connects after the commit replays only committed history, never stale deltas.
func TestLiveTailClearedOnCommit(t *testing.T) {
	s := newTestSession(t, "s")
	if err := s.commit(llm.UserMessage("hi")); err != nil {
		t.Fatalf("commit user: %v", err)
	}
	s.publish(api.Envelope{Kind: api.KindLLM, Event: &codec.Event{Type: codec.TypeMessageDelta, Delta: "partial"}})

	if seq, events := s.Live(); seq != 1 || len(events) != 1 {
		t.Fatalf("Live before commit = seq %d, %d events; want 1, 1", seq, len(events))
	}
	if err := s.commit(llm.AssistantMessage("partial", "", nil)); err != nil {
		t.Fatalf("commit assistant: %v", err)
	}
	if seq, events := s.Live(); len(events) != 0 {
		t.Errorf("Live after commit = seq %d, %d events; want empty tail (the commit supersedes the deltas)", seq, len(events))
	}

	// A late subscriber now replays only committed messages, no LLM deltas.
	ctx, cancel := context.WithCancel(context.Background())
	stream := s.From(ctx, 1)
	cancel()
	for env := range stream {
		if env.Kind == api.KindLLM {
			t.Errorf("replayed LLM envelope after commit: %+v", env)
		}
	}
}

// TestLiveTailSkipsToolEnvelopes pins the tail's scope to the LLM stream: tool
// envelopes are reconstructed via the in-flight run registry (Runs), so they
// must not accumulate in the tail.
func TestLiveTailSkipsToolEnvelopes(t *testing.T) {
	s := newTestSession(t, "s")
	if err := s.commit(llm.UserMessage("hi")); err != nil {
		t.Fatalf("commit user: %v", err)
	}
	s.publish(api.Envelope{Kind: api.KindToolStarted, ToolCallID: "c1", Name: "shell"})
	if _, events := s.Live(); len(events) != 0 {
		t.Fatalf("Live after tool_started = %+v; want empty (tools come from /runs)", events)
	}
	s.publish(api.Envelope{Kind: api.KindLLM, Event: &codec.Event{Type: codec.TypeMessageDelta, Delta: "x"}})
	seq, events := s.Live()
	if len(events) != 1 || events[0].LiveSeq != 1 || events[0].Event.Delta != "x" {
		t.Errorf("Live after LLM = seq %d, %+v; want just the delta at live_seq 1", seq, events)
	}
}

// TestLiveSeqMonotonicAcrossCommits ensures live positions never reset: the
// next block's deltas continue the counter, so a client's lastLiveSeq keeps
// working as a dedup key across the whole session.
func TestLiveSeqMonotonicAcrossCommits(t *testing.T) {
	s := newTestSession(t, "s")
	if err := s.commit(llm.UserMessage("hi")); err != nil {
		t.Fatalf("commit user: %v", err)
	}
	s.publish(api.Envelope{Kind: api.KindLLM, Event: &codec.Event{Type: codec.TypeMessageDelta, Delta: "a"}})
	s.publish(api.Envelope{Kind: api.KindLLM, Event: &codec.Event{Type: codec.TypeMessageDelta, Delta: "b"}})
	if err := s.commit(llm.AssistantMessage("ab", "", nil)); err != nil {
		t.Fatalf("commit assistant: %v", err)
	}
	s.publish(api.Envelope{Kind: api.KindLLM, Event: &codec.Event{Type: codec.TypeMessageDelta, Delta: "c"}})

	seq, events := s.Live()
	if seq != 3 {
		t.Errorf("live seq after second block = %d, want 3 (monotonic, not reset)", seq)
	}
	if len(events) != 1 || events[0].LiveSeq != 3 {
		t.Errorf("tail after second block = %+v; want only delta c at live_seq 3", events)
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

	st := NewStore(memPersister(t), nil, nctx)
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
	st := NewStore(memPersister(t), nil)
	s, err := st.Create(nil)
	if err != nil {
		t.Fatalf("Create: %v", err)
	}

	// With no remote client connected, the provider is the local one: it runs
	// tools in the server process and reports the server's own environment, so
	// "local" is a first-class provider rather than an anonymous fallback.
	p := s.provider()
	if _, ok := p.(*localProvider); !ok {
		t.Errorf("default provider = %T, want *localProvider", p)
	}
	if env := p.Environment(); !strings.Contains(env, "Working directory") {
		t.Errorf("local provider environment = %q, want the server's context", env)
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
	return newSession(id, nil, nil, d, rowID, time.Now().UnixMilli(), 0, "", nil)
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

// startLoop runs the session's scheduler, matching production (Store.Create and
// Store.Load start it). Provider notices are committed by this goroutine, so
// tests that assert on notices must run it and wait (see waitForHistory).
func startLoop(t *testing.T, s *Session) {
	t.Helper()
	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)
	go s.loop(ctx)
}

// waitForHistory polls the session's committed history until want is satisfied.
// Notices commit asynchronously on the scheduler goroutine, so assertions on
// them must wait rather than read immediately after the triggering call.
func waitForHistory(t *testing.T, s *Session, want func([]llm.ChatMessage) bool) []llm.ChatMessage {
	t.Helper()
	deadline := time.After(5 * time.Second)
	for {
		msgs, _ := s.loadMessages()
		if want(msgs) {
			return msgs
		}
		select {
		case <-deadline:
			t.Fatalf("timed out waiting for history condition; history = %+v", msgs)
		case <-time.After(10 * time.Millisecond):
		}
	}
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

	if err := s.commitQuery(turn1, agentQuery(0, 0, 2, 3, nil)); err != nil {
		t.Fatalf("commitQuery turn1 q0: %v", err)
	}
	if err := s.commitQuery(turn1, agentQuery(1, 0, 4, 5, fmt.Errorf("rate limit"))); err != nil {
		t.Fatalf("commitQuery turn1 q1: %v", err)
	}

	// Turn 2: user message, one successful query.
	commitN(t, s, "turn", 1)
	ps, err = s.Persisted()
	if err != nil {
		t.Fatalf("Persisted: %v", err)
	}
	turn2 := ps.Messages[len(ps.Messages)-1].Seq
	// Turn 2's query reports a cache split: 3 cached + 5 miss = 8 in, 13 out.
	if err := s.commitQuery(turn2, agentQuery(0, 3, 5, 13, nil)); err != nil {
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
	if turns[0].UserSeq != turn1 || turns[0].UncachedInput != 6 || turns[0].CachedInput != 0 || turns[0].Output != 8 {
		t.Errorf("turn 1 = %+v, want user %d, usage 6/8 (all uncached)", turns[0], turn1)
	}
	if !strings.Contains(turns[0].Error, "rate limit") {
		t.Errorf("turn 1 error = %q, want the failed query's error", turns[0].Error)
	}
	// Turn 2: just its one query's usage, no error.
	if turns[1].UserSeq != turn2 || turns[1].CachedInput != 3 || turns[1].UncachedInput != 5 || turns[1].Output != 13 || turns[1].Error != "" {
		t.Errorf("turn 2 = %+v, want user %d, usage 3 cached + 5 miss / 13 out, no error", turns[1], turn2)
	}
}

// agentQuery builds an agent.Query for commitQuery.
func agentQuery(idx, cached, uncached, output int, err error) agent.Query {
	return agent.Query{Idx: idx, CachedInput: cached, UncachedInput: uncached, Output: output, Err: err}
}

// TestExecStatusReflectsProvider verifies the provider status transitions with
// registration: local when no client is connected, remote (with the reported
// context) once one registers, and local again after it unregisters.
func TestExecStatusReflectsProvider(t *testing.T) {
	s := newTestSession(t, "s")
	if st := s.ExecStatus(); st.Connected || st.Kind != "local" {
		t.Errorf("fresh status = %+v, want local/disconnected", st)
	}

	s.SetExecContext(api.ExecContext{System: "linux/amd64", CWD: "/work", Files: []string{"README.md"}})
	ch := make(chan api.ExecRequest, 1)
	id := s.RegisterExec(ch, "", "", "")

	st := s.ExecStatus()
	if !st.Connected || st.Kind != "remote" {
		t.Errorf("registered status = %+v, want remote/connected", st)
	}
	if st.Context == nil || st.Context.CWD != "/work" {
		t.Errorf("registered status context = %+v, want the reported cwd", st.Context)
	}

	s.UnregisterExec(id)
	if st := s.ExecStatus(); st.Connected || st.Kind != "local" {
		t.Errorf("after unregister status = %+v, want local/disconnected", st)
	}
}

// TestProviderChangeCommitsSystemNotices verifies that registering and
// unregistering an execution provider commits short role-"system" messages, so
// the model sees the environment change on its next turn.
func TestProviderChangeCommitsSystemNotices(t *testing.T) {
	s := newTestSession(t, "s")
	startLoop(t, s)
	s.SetExecContext(api.ExecContext{System: "linux/amd64", CWD: "/work"})
	id := s.RegisterExec(make(chan api.ExecRequest, 1), "", "", "")
	s.UnregisterExec(id)

	// The notices are queued and committed by the scheduler, possibly coalesced
	// into one message; wait until the disconnect notice has landed.
	msgs := waitForHistory(t, s, func(h []llm.ChatMessage) bool {
		for _, m := range h {
			if m.Role == "system" && strings.Contains(m.Content, "disconnected") {
				return true
			}
		}
		return false
	})
	var sys []llm.ChatMessage
	for _, m := range msgs {
		if m.Role != "system" {
			t.Errorf("non-system message in notice history = %+v", m)
		}
		sys = append(sys, m)
	}
	if len(sys) == 0 || len(sys) > 2 {
		t.Fatalf("history = %+v, want 1-2 system notices", sys)
	}
	if !strings.Contains(sys[0].Content, "execution provider connected") || !strings.Contains(sys[0].Content, "/work") {
		t.Errorf("connect notice = %+v", sys[0])
	}
	if !strings.Contains(sys[len(sys)-1].Content, "disconnected") {
		t.Errorf("disconnect notice = %+v", sys[len(sys)-1])
	}
}

// TestProviderChangePublishesStatus verifies the exec_status envelope is
// published on register and unregister, so clients get real-time status.
func TestProviderChangePublishesStatus(t *testing.T) {
	s := newTestSession(t, "s")
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	stream := s.From(ctx, 0)

	done := make(chan struct{})
	var kinds []string
	var statuses []api.ExecStatus
	go func() {
		defer close(done)
		for env := range stream {
			if env.Kind == api.KindExecStatus && env.ExecStatus != nil {
				kinds = append(kinds, env.Kind)
				statuses = append(statuses, *env.ExecStatus)
			}
		}
	}()

	id := s.RegisterExec(make(chan api.ExecRequest, 1), "", "", "")
	s.UnregisterExec(id)

	// Give the publish goroutine a moment, then stop the stream.
	time.Sleep(50 * time.Millisecond)
	cancel()
	<-done

	if len(statuses) < 2 {
		t.Fatalf("got %d exec_status envelopes %v, want >= 2", len(statuses), kinds)
	}
	if !statuses[0].Connected || statuses[0].Kind != "remote" {
		t.Errorf("first status = %+v, want remote", statuses[0])
	}
	if statuses[len(statuses)-1].Connected || statuses[len(statuses)-1].Kind != "local" {
		t.Errorf("last status = %+v, want local", statuses[len(statuses)-1])
	}
}

// TestRemoteProviderDefsAndEnvironmentFromContext verifies the session's remote
// provider exposes the client's reported context: load_skill appears when
// skills are registered, and Environment returns the context system message.
func TestRemoteProviderDefsAndEnvironmentFromContext(t *testing.T) {
	s := newTestSession(t, "s")

	// No context yet: remote provider (once registered) exposes only shell and
	// no environment.
	id := s.RegisterExec(make(chan api.ExecRequest, 1), "", "", "")
	p := s.provider()
	if len(p.Defs()) != 1 {
		t.Errorf("Defs with no context = %d tools, want 1 (shell)", len(p.Defs()))
	}
	if env := p.Environment(); env != "" {
		t.Errorf("Environment with no context = %q, want empty", env)
	}
	s.UnregisterExec(id)

	// With a skill reported, the same provider exposes load_skill and a context
	// system message mentioning the skill and cwd.
	s.SetExecContext(api.ExecContext{
		System: "linux/amd64",
		CWD:    "/work",
		Files:  []string{"README.md"},
		Skills: []api.Skill{{Name: "my-skill", Description: "does things", Path: "/work/skills/my-skill/SKILL.md"}},
	})
	id = s.RegisterExec(make(chan api.ExecRequest, 1), "", "", "")
	p = s.provider()
	defs := p.Defs()
	if len(defs) != 2 || defs[1].Function.Name != "load_skill" {
		t.Errorf("Defs with skills = %+v, want shell + load_skill", defs)
	}
	env := p.Environment()
	for _, want := range []string{"linux/amd64", "/work", "README.md", "my-skill: does things"} {
		if !strings.Contains(env, want) {
			t.Errorf("Environment missing %q:\n%s", want, env)
		}
	}
}

// TestStopAbortsRunningTurn verifies the session's Stop: it cancels the running
// turn's context (the func runTurn registers), a second Stop while the same
// turn runs is a no-op error, and once the turn's deferred cleanup clears the
// cancel func, Stop reports no turn running.
func TestStopAbortsRunningTurn(t *testing.T) {
	s := newTestSession(t, "s")

	// Idle: no turn running.
	if err := s.Stop(); err == nil {
		t.Fatalf("Stop on idle session error = nil, want error")
	}

	// A turn starts: runTurn registers the turn's cancel func.
	var cancelled bool
	s.mu.Lock()
	s.turnCancel = func() { cancelled = true }
	s.mu.Unlock()

	if err := s.Stop(); err != nil {
		t.Fatalf("Stop: %v", err)
	}
	if !cancelled {
		t.Errorf("turn's cancel func was not called")
	}
	// A second Stop on the same turn is a harmless no-op (context cancel is
	// idempotent); the UI guards against double-clicks, and a racing click can
	// never cancel the *next* turn because the func is cleared when this one
	// ends.
	if err := s.Stop(); err != nil {
		t.Errorf("second Stop error = %v, want nil (idempotent)", err)
	}

	// The turn ends: the deferred cleanup clears the cancel func (as runTurn's
	// defer does), so Stop reports idle again.
	s.mu.Lock()
	s.turnCancel = nil
	s.mu.Unlock()
	if err := s.Stop(); err == nil {
		t.Errorf("Stop after turn cleanup error = nil, want error")
	}
}

// TestDeriveTurnsAggregatesStopped verifies a turn the user stopped (any of
// its queries marked stopped) derives with Stopped set, distinct from an
// error, and that usage still aggregates from the stopped query's partial
// tokens.
func TestDeriveTurnsAggregatesStopped(t *testing.T) {
	s := newTestSession(t, "s")

	// Turn 1: user message + a stopped query carrying the partial usage.
	commitN(t, s, "turn", 1)
	ps, err := s.Persisted()
	if err != nil {
		t.Fatalf("Persisted: %v", err)
	}
	turn1 := ps.Messages[0].Seq
	q := agentQuery(0, 1, 2, 3, nil)
	q.Stopped = true
	if err := s.commitQuery(turn1, q); err != nil {
		t.Fatalf("commitQuery: %v", err)
	}

	ps, err = s.Persisted()
	if err != nil {
		t.Fatalf("Persisted: %v", err)
	}
	turns := DeriveTurns(ps)
	if len(turns) != 1 {
		t.Fatalf("turns = %+v, want 1", turns)
	}
	if !turns[0].Stopped {
		t.Errorf("turn Stopped = false, want true")
	}
	if turns[0].Error != "" {
		t.Errorf("turn Error = %q, want empty (a stop is not an error)", turns[0].Error)
	}
	if turns[0].CachedInput != 1 || turns[0].UncachedInput != 2 || turns[0].Output != 3 {
		t.Errorf("turn usage = %d/%d/%d, want the stopped query's partial usage 1/2/3",
			turns[0].CachedInput, turns[0].UncachedInput, turns[0].Output)
	}
}

// TestCommittedToolEnvelopeCarriesToolOutput verifies the tool-output metadata
// rides on the committed KindMessage envelope — for the live bus and, after a
// rebuild from the persister, for a reconnecting client's replay — so the UI
// renders the badge identically whether live or from history.
func TestCommittedToolEnvelopeCarriesToolOutput(t *testing.T) {
	s := newTestSession(t, "s")
	meta := &llm.ToolOutputMeta{Truncated: true, TotalBytes: 2550, ShownBytes: 1536}
	m := llm.ToolResult("call-1", "big output")
	m.ToolOutput = meta
	if err := s.commit(m); err != nil {
		t.Fatalf("commit: %v", err)
	}

	// The live commit path (session.commit) stamps the metadata onto the
	// committed KindMessage envelope.
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	var live api.Envelope
	for env := range s.From(ctx, 0) {
		live = env
		break
	}
	if live.Kind != api.KindMessage || live.Message == nil || live.Message.Role != "tool" {
		t.Fatalf("live first envelope = %+v, want the tool message", live)
	}
	if live.ToolOutput == nil || *live.ToolOutput != *meta {
		t.Errorf("live committed envelope ToolOutput = %+v, want %+v", live.ToolOutput, meta)
	}

	// Rebuild from the persister (simulating a server restart): the replay
	// envelope must carry the metadata too, matching /view.
	ps, err := s.Persisted()
	if err != nil {
		t.Fatalf("Persisted: %v", err)
	}
	s2 := newSession(s.ID(), nil, nil, s.persist, ps.ID, ps.CreatedAt, ps.ArchivedAt, ps.Name, nil)
	s2.rebuildFromPersisted(ps)

	ctx2, cancel2 := context.WithCancel(context.Background())
	defer cancel2()
	var replayed api.Envelope
	for env := range s2.From(ctx2, 0) {
		replayed = env
		break
	}
	if replayed.Kind != api.KindMessage || replayed.Message == nil || replayed.Message.Role != "tool" {
		t.Fatalf("replay first envelope = %+v, want the tool message", replayed)
	}
	if replayed.ToolOutput == nil || *replayed.ToolOutput != *meta {
		t.Errorf("replayed envelope ToolOutput = %+v, want %+v", replayed.ToolOutput, meta)
	}
}

func TestStoreArchiveUnarchive(t *testing.T) {
	st := NewStore(memPersister(t), nil)
	s, err := st.Create(nil)
	if err != nil {
		t.Fatalf("Create: %v", err)
	}
	if s.Archived() {
		t.Error("fresh session Archived() = true, want false")
	}

	// Archive: the session reports archived and the list carries a stamp.
	if err := st.Archive(s.ID()); err != nil {
		t.Fatalf("Archive: %v", err)
	}
	if !s.Archived() {
		t.Error("Archived() after Archive = false, want true")
	}
	list := st.List()
	if len(list) != 1 || list[0].ArchivedAt == 0 {
		t.Fatalf("list after archive = %+v, want one archived row", list)
	}
	// The stamp is persisted: a fresh session rebuilt from the same persister
	// sees it (a restarted server keeps the folder state).
	ps, err := s.Persisted()
	if err != nil {
		t.Fatalf("Persisted: %v", err)
	}
	if ps.ArchivedAt == 0 {
		t.Errorf("persisted archived_at = 0, want the archive stamp")
	}

	// Unarchive clears the flag everywhere.
	if err := st.Unarchive(s.ID()); err != nil {
		t.Fatalf("Unarchive: %v", err)
	}
	if s.Archived() {
		t.Error("Archived() after Unarchive = true, want false")
	}
	list = st.List()
	if len(list) != 1 || list[0].ArchivedAt != 0 {
		t.Fatalf("list after unarchive = %+v, want one active row", list)
	}

	// Unknown session: ErrNotFound on both operations.
	if err := st.Archive("session_9999"); err != db.ErrNotFound {
		t.Errorf("Archive unknown = %v, want db.ErrNotFound", err)
	}
	if err := st.Unarchive("session_9999"); err != db.ErrNotFound {
		t.Errorf("Unarchive unknown = %v, want db.ErrNotFound", err)
	}
}

func TestStoreRenameBroadcasts(t *testing.T) {
	st := NewStore(memPersister(t), nil)
	s, err := st.Create(nil)
	if err != nil {
		t.Fatalf("Create: %v", err)
	}

	// A fresh session has no name.
	if got := s.Name(); got != "" {
		t.Fatalf("fresh Name() = %q, want empty", got)
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	stream := s.From(ctx, 0)

	// Rename: memory updates, and a live session_renamed envelope carries the
	// new name to subscribers (persist-first ordering is by construction;
	// the db test covers persistence).
	if err := st.Rename(s.ID(), "work chat"); err != nil {
		t.Fatalf("Rename: %v", err)
	}
	if got := s.Name(); got != "work chat" {
		t.Errorf("Name() = %q, want %q", got, "work chat")
	}
	select {
	case env := <-stream:
		if env.Kind != api.KindSessionRenamed || env.SessionName != "work chat" {
			t.Fatalf("envelope = %+v, want session_renamed with name", env)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("no session_renamed envelope within 5s")
	}

	// The name is not part of history: a subscriber that only replays the
	// committed log sees no session_renamed (it is live-only).
	ps, err := s.Persisted()
	if err != nil {
		t.Fatalf("Persisted: %v", err)
	}
	if ps.Name != "work chat" {
		t.Errorf("persisted name = %q, want %q", ps.Name, "work chat")
	}

	// Clearing publishes the empty name (falls back to the preview).
	if err := st.Rename(s.ID(), ""); err != nil {
		t.Fatalf("Rename clear: %v", err)
	}
	if got := s.Name(); got != "" {
		t.Errorf("Name() after clear = %q, want empty", got)
	}
	select {
	case env := <-stream:
		if env.Kind != api.KindSessionRenamed || env.SessionName != "" {
			t.Fatalf("envelope = %+v, want session_renamed with empty name", env)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("no session_renamed envelope within 5s")
	}

	// The list carries the name for the sidebar.
	if err := st.Rename(s.ID(), "sidebar name"); err != nil {
		t.Fatalf("Rename: %v", err)
	}
	list := st.List()
	if len(list) != 1 || list[0].Name != "sidebar name" {
		t.Fatalf("list after rename = %+v, want one row with name", list)
	}

	// Unknown session: ErrNotFound, and no broadcast.
	if err := st.Rename("session_9999", "x"); err != db.ErrNotFound {
		t.Fatalf("Rename unknown = %v, want db.ErrNotFound", err)
	}
}

// TestExecStatusListsAllClients verifies the status carries the full registry —
// local plus every connected client, with the active id — so the web picker
// can render every provider without polling.
func TestExecStatusListsAllClients(t *testing.T) {
	s := newTestSession(t, "s")
	s.RegisterExec(make(chan api.ExecRequest, 1), "laptop", "laptop", "remote")
	s.RegisterExec(make(chan api.ExecRequest, 1), "server", "server", "remote")

	st := s.ExecStatus()
	if st.ActiveID != "server" {
		t.Errorf("active after connects = %q, want server (last connect takes over)", st.ActiveID)
	}
	if len(st.Clients) != 3 {
		t.Fatalf("clients = %d, want 3 (local + laptop + server)", len(st.Clients))
	}
	// Local is always present and listed first.
	if st.Clients[0].ID != "local" || !st.Clients[0].Connected {
		t.Errorf("clients[0] = %+v, want the local provider first", st.Clients[0])
	}
	seen := map[string]bool{}
	for _, c := range st.Clients {
		seen[c.ID] = true
	}
	for _, want := range []string{"local", "laptop", "server"} {
		if !seen[want] {
			t.Errorf("clients missing %q: %+v", want, st.Clients)
		}
	}
}

// TestSelectExecSwitchesActiveProvider is the picker's core: selecting a
// connected client makes it the active provider (the model's Environment
// switches to it), the deselected client stays registered (no reconnect
// needed to switch back), and selecting local runs tools in the server process
// while remotes stay connected.
func TestSelectExecSwitchesActiveProvider(t *testing.T) {
	s := newTestSession(t, "s")
	s.SetExecContext(api.ExecContext{ID: "laptop", System: "darwin/arm64", CWD: "/Users/mark"})
	s.RegisterExec(make(chan api.ExecRequest, 1), "laptop", "laptop", "remote")
	s.SetExecContext(api.ExecContext{ID: "server", System: "linux/amd64", CWD: "/app"})
	s.RegisterExec(make(chan api.ExecRequest, 1), "server", "server", "remote")

	// The second client to connect takes over, same as before the registry.
	if st := s.ExecStatus(); st.ActiveID != "server" {
		t.Errorf("active after connects = %q, want server", st.ActiveID)
	}
	if env := s.provider().Environment(); !strings.Contains(env, "/app") {
		t.Errorf("environment after connects = %q, want the server context", env)
	}

	// Select the laptop: the provider's environment switches to it, and both
	// remotes remain in the registry (the laptop never disconnected).
	if err := s.SelectExec("laptop"); err != nil {
		t.Fatalf("SelectExec(laptop): %v", err)
	}
	st := s.ExecStatus()
	if st.ActiveID != "laptop" {
		t.Errorf("active after select = %q, want laptop", st.ActiveID)
	}
	if env := s.provider().Environment(); !strings.Contains(env, "/Users/mark") {
		t.Errorf("environment after select = %q, want the laptop context", env)
	}
	if len(st.Clients) != 3 {
		t.Errorf("clients after select = %d, want 3 (local + laptop + server)", len(st.Clients))
	}

	// Select local: tools run in the server process; the remotes stay listed.
	if err := s.SelectExec("local"); err != nil {
		t.Fatalf("SelectExec(local): %v", err)
	}
	st = s.ExecStatus()
	if st.ActiveID != "local" || st.Connected {
		t.Errorf("after select local = %+v, want local/disconnected", st)
	}
	if _, ok := s.provider().(*localProvider); !ok {
		t.Errorf("provider after select local = %T, want *localProvider", s.provider())
	}
	if len(st.Clients) != 3 {
		t.Errorf("clients after select local = %d, want 3 (remotes still connected)", len(st.Clients))
	}

	// And back to the laptop: it was standing by the whole time.
	if err := s.SelectExec("laptop"); err != nil {
		t.Fatalf("SelectExec(laptop) again: %v", err)
	}
	if st := s.ExecStatus(); st.ActiveID != "laptop" {
		t.Errorf("active after reselect = %q, want laptop", st.ActiveID)
	}
}

// TestSelectExecErrorsOnUnknown verifies selecting a provider that isn't
// connected is rejected, and selecting the already-active provider is a no-op
// success (no error, no duplicate notice).
func TestSelectExecErrorsOnUnknown(t *testing.T) {
	s := newTestSession(t, "s")
	if err := s.SelectExec("ghost"); err == nil {
		t.Error("SelectExec(ghost) = nil, want error for an unknown client")
	}
	if err := s.SelectExec("local"); err != nil {
		t.Errorf("SelectExec(local) when already active = %v, want nil", err)
	}
}

// TestSelectExecCommitsSystemNotices verifies selecting a provider commits a
// short role-"system" notice naming it, so the model sees the environment
// change on its next turn.
func TestSelectExecCommitsSystemNotices(t *testing.T) {
	s := newTestSession(t, "s")
	startLoop(t, s)
	s.SetExecContext(api.ExecContext{ID: "laptop", System: "darwin/arm64", CWD: "/Users/mark"})
	s.RegisterExec(make(chan api.ExecRequest, 1), "laptop", "laptop", "remote")

	if err := s.SelectExec("local"); err != nil {
		t.Fatalf("SelectExec(local): %v", err)
	}
	if err := s.SelectExec("laptop"); err != nil {
		t.Fatalf("SelectExec(laptop): %v", err)
	}

	// Notices commit asynchronously on the scheduler, possibly coalesced; wait
	// until the final laptop-selection notice has landed.
	msgs := waitForHistory(t, s, func(h []llm.ChatMessage) bool {
		if len(h) == 0 {
			return false
		}
		// The selection notice is "execution provider: laptop (...)", distinct
		// from the connect notice "execution provider connected: laptop (...)".
		last := h[len(h)-1]
		return last.Role == "system" && strings.Contains(last.Content, "execution provider: laptop") && strings.Contains(last.Content, "darwin/arm64")
	})
	last := msgs[len(msgs)-1]
	if !strings.Contains(last.Content, "execution provider") {
		t.Errorf("select-laptop notice = %+v, want a system notice naming the laptop", last)
	}
	// Somewhere in the (possibly coalesced) notice history, the intermediate
	// selection of local must be recorded.
	var sawLocal bool
	for _, m := range msgs {
		if m.Role == "system" && strings.Contains(m.Content, "local") {
			sawLocal = true
		}
	}
	if !sawLocal {
		t.Errorf("no notice recording the local selection; history = %+v", msgs)
	}
}

// TestUnregisterNonActiveKeepsActiveProvider verifies a standby client's
// disconnect doesn't change the active provider — it just drops out of the
// registry — while the active client's disconnect reverts to local.
func TestUnregisterNonActiveKeepsActiveProvider(t *testing.T) {
	s := newTestSession(t, "s")
	laptop := s.RegisterExec(make(chan api.ExecRequest, 1), "laptop", "laptop", "remote")
	server := s.RegisterExec(make(chan api.ExecRequest, 1), "server", "server", "remote")

	s.UnregisterExec(laptop)
	st := s.ExecStatus()
	if st.ActiveID != "server" {
		t.Errorf("active after standby disconnect = %q, want server", st.ActiveID)
	}
	if len(st.Clients) != 2 {
		t.Errorf("clients after standby disconnect = %d, want 2 (local + server)", len(st.Clients))
	}

	s.UnregisterExec(server)
	st = s.ExecStatus()
	if st.ActiveID != "local" || st.Connected {
		t.Errorf("after active disconnect = %+v, want local/disconnected", st)
	}
	if len(st.Clients) != 1 {
		t.Errorf("clients after active disconnect = %d, want 1 (local only)", len(st.Clients))
	}
}

// TestRemoteProviderRunsOnActiveClient verifies tool calls route to the active
// client's channel only: a deselected client's channel receives nothing, and
// switching selection routes the next call to the newly active client.
func TestRemoteProviderRunsOnActiveClient(t *testing.T) {
	s := newTestSession(t, "s")
	laptopCh := make(chan api.ExecRequest, 4)
	serverCh := make(chan api.ExecRequest, 4)
	s.RegisterExec(laptopCh, "laptop", "laptop", "remote")
	s.RegisterExec(serverCh, "server", "server", "remote")

	rc, err := s.provider().Run(context.Background(), "shell", []byte(`{"command":"pwd"}`))
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	defer rc.Close()

	select {
	case req := <-serverCh:
		if req.Name != "shell" {
			t.Errorf("server got call %q, want shell", req.Name)
		}
	default:
		t.Error("active server client never received the call")
	}
	select {
	case <-laptopCh:
		t.Error("standby laptop client received a call; only the active provider should")
	default:
	}

	// Switching to the laptop routes the next call there.
	if err := s.SelectExec("laptop"); err != nil {
		t.Fatalf("SelectExec(laptop): %v", err)
	}
	rc2, err := s.provider().Run(context.Background(), "shell", []byte(`{"command":"pwd"}`))
	if err != nil {
		t.Fatalf("Run after select: %v", err)
	}
	defer rc2.Close()
	select {
	case <-laptopCh:
	default:
		t.Error("laptop client never received the call after selection")
	}
}

// TestProviderNoticeDeferredToTurnBoundary is the regression test for the
// production 400 "An assistant message with 'tool_calls' must be followed by
// tool messages responding to each 'tool_call_id'": an execution-provider
// disconnect that lands while a tool is mid-flight must NOT persist a system
// notice between the assistant tool_calls message and its tool result. The
// notice is queued and committed by the scheduler only after the turn ends.
func TestProviderNoticeDeferredToTurnBoundary(t *testing.T) {
	var mu sync.Mutex
	reqCount := 0
	llmSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		reqCount++
		n := reqCount
		mu.Unlock()
		w.Header().Set("Content-Type", "text/event-stream")
		if n == 1 {
			// First request: the model asks for a shell tool call.
			fmt.Fprint(w,
				`data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"shell","arguments":"{\"command\":\"echo hi\"}"}}]},"finish_reason":"tool_calls"}]}`+"\n\n"+
					`data: [DONE]`+"\n")
			return
		}
		// Second request: plain reply, no more tools.
		fmt.Fprint(w,
			`data: {"choices":[{"delta":{"content":"done"},"finish_reason":"stop"}],"usage":{"prompt_tokens":2,"completion_tokens":3}}`+"\n\n"+
				`data: [DONE]`+"\n")
	}))
	defer llmSrv.Close()

	client := llm.NewClient(config.Config{BaseURL: llmSrv.URL + "/v1", Model: "m", APIKey: "k"}, nil)
	nctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	st := NewStore(memPersister(t), nil, nctx)
	s, err := st.Create(client)
	if err != nil {
		t.Fatalf("Create: %v", err)
	}

	ch := make(chan api.ExecRequest, 8)
	id := s.RegisterExec(ch, "laptop", "laptop", "remote")
	s.Enqueue("run a tool")

	// Act as the execution client: wait for the tool call, then disconnect
	// mid-flight. UnregisterExec closes the agent's pipe with
	// "execution client disconnected", so the tool result lands as an error.
	var req api.ExecRequest
	select {
	case req = <-ch:
	case <-time.After(5 * time.Second):
		t.Fatal("timed out waiting for the tool call on the exec channel")
	}
	if req.Name != "shell" {
		t.Fatalf("tool call name = %q, want shell", req.Name)
	}
	s.UnregisterExec(id)

	// The turn finishes (tool result + final reply), then the queued disconnect
	// notice commits at the boundary. Wait for both.
	msgs := waitForHistory(t, s, func(h []llm.ChatMessage) bool {
		var done, notice bool
		for _, m := range h {
			if m.Role == "assistant" && strings.Contains(m.Content, "done") {
				done = true
			}
			if m.Role == "system" && strings.Contains(m.Content, "disconnected") {
				notice = true
			}
		}
		return done && notice
	})

	// The invariant: every assistant message with tool_calls is immediately
	// followed by a role-"tool" message responding to its first call id — no
	// system notice (or anything else) wedged in between.
	var sawToolCall bool
	for i, m := range msgs {
		if m.Role != "assistant" || len(m.ToolCalls) == 0 {
			continue
		}
		sawToolCall = true
		if i+1 >= len(msgs) {
			t.Fatalf("assistant tool_calls message is last in history = %+v", msgs)
		}
		next := msgs[i+1]
		if next.Role != "tool" {
			t.Fatalf("message after assistant tool_calls = role %q, want tool (wedge!); history = %+v", next.Role, msgs)
		}
		if next.ToolCallID != m.ToolCalls[0].ID {
			t.Errorf("tool result call id = %q, want %q", next.ToolCallID, m.ToolCalls[0].ID)
		}
	}
	if !sawToolCall {
		t.Fatalf("no assistant tool_calls message in history: %+v", msgs)
	}

	// The disconnect notice must exist and must be committed after the tool
	// result, not between the tool call and its result.
	var toolIdx, noticeIdx = -1, -1
	for i, m := range msgs {
		if m.Role == "tool" {
			toolIdx = i
		}
		if m.Role == "system" && strings.Contains(m.Content, "disconnected") {
			noticeIdx = i
		}
	}
	if noticeIdx == -1 {
		t.Fatalf("disconnect notice missing from history: %+v", msgs)
	}
	if toolIdx == -1 || noticeIdx < toolIdx {
		t.Errorf("disconnect notice (idx %d) not after tool result (idx %d); history = %+v", noticeIdx, toolIdx, msgs)
	}
}

// TestProviderExposesRemoteMCPServers proves the session's provider wraps the
// active remote client's reported MCP servers into the Composite, so FindMCP
// lists them (hosted on the client) and CallMCP for one routes down the exec
// channel. It is the integration seam between the exec context and the MCP
// surface.
func TestProviderExposesRemoteMCPServers(t *testing.T) {
	s := newTestSession(t, "session_1")
	remote := []api.MCPServer{{
		Name: "retool", Description: "Retool", Host: "laptop",
		Tools: []api.MCPTool{{Name: "whoami", Description: "Who am I"}},
	}}
	ch := make(chan api.ExecRequest, 4)
	s.SetExecContext(api.ExecContext{ID: "laptop", System: "darwin/arm64", CWD: "/home/me", MCPServers: remote})
	id := s.RegisterExec(ch, "laptop", "laptop", "remote")
	defer s.UnregisterExec(id)

	p := s.provider()
	// The active client is remote, so its MCP servers are exposed through the
	// Composite even though the session has no server-side hub.
	comp, ok := p.(*mcp.Composite)
	if !ok {
		t.Fatalf("provider = %T, want *mcp.Composite", p)
	}
	if len(comp.Remote) != 1 || comp.Remote[0].Name != "retool" || comp.Remote[0].Host != "laptop" {
		t.Fatalf("Composite.Remote = %+v", comp.Remote)
	}

	// Defs include the MCP tools (shell + FindMCP + CallMCP).
	names := map[string]bool{}
	for _, d := range p.Defs() {
		names[d.Function.Name] = true
	}
	for _, want := range []string{"shell", mcp.FindTool, mcp.CallTool} {
		if !names[want] {
			t.Errorf("Defs missing %q: %v", want, names)
		}
	}

	// CallMCP for the remote server routes down the exec channel: a request
	// lands on the client's channel with the CallMCP name.
	go func() {
		p.Run(context.Background(), mcp.CallTool, []byte(`{"server_name":"retool","tool_name":"whoami"}`))
	}()
	select {
	case req := <-ch:
		if req.Name != mcp.CallTool {
			t.Errorf("exec request name = %q, want CallMCP", req.Name)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("no exec request routed to the client")
	}
}

// longReply is an assistant reply that clears the auto-humanize thresholds
// (>= MinWords words and >= MinSentences sentences, prose not code).
func longReply() string {
	return strings.Repeat("This is a perfectly ordinary sentence with more than enough words to qualify. ", 6)
}

// llmStub returns a streaming Chat Completions server that replies with long
// on every request. The turn and the humanize pass share it, so the rewritten
// variant equals the original reply.
func llmStub(t *testing.T, long string) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprintf(w,
			`data: {"choices":[{"delta":{"content":%q},"finish_reason":"stop"}],"usage":{"prompt_tokens":2,"completion_tokens":3}}`+"\n\n"+
				`data: [DONE]`+"\n", long)
	}))
	t.Cleanup(srv.Close)
	return srv
}

// waitVariant drains a bus subscription until a variant_ready for the given
// index arrives (or the deadline hits), returning every variant envelope seen.
func waitVariant(t *testing.T, stream <-chan api.Envelope, wantIdx int) (started, ready *api.Variant) {
	t.Helper()
	deadline := time.After(5 * time.Second)
	for {
		select {
		case env := <-stream:
			switch env.Kind {
			case api.KindVariantStarted:
				if env.Variant != nil {
					started = env.Variant
				}
			case api.KindVariant:
				if env.Variant != nil && env.Variant.Index == wantIdx {
					return started, env.Variant
				}
			}
		case <-deadline:
			t.Fatalf("timed out waiting for variant %d; started=%+v ready=%+v", wantIdx, started, ready)
		}
	}
}

// TestCommitAutoHumanizesLongReply drives the automatic plain-language pass
// end to end: a long assistant reply qualifies at commit, the background pass
// runs against the LLM stub, and the bus carries variant_started then
// variant_ready with the rewrite persisted.
func TestCommitAutoHumanizesLongReply(t *testing.T) {
	long := longReply()
	srv := llmStub(t, long)
	client := llm.NewClient(config.Config{BaseURL: srv.URL + "/v1", Model: "m", APIKey: "k"}, nil)
	nctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	st := NewStore(memPersister(t), nil, nctx)
	s, err := st.Create(client)
	if err != nil {
		t.Fatalf("Create: %v", err)
	}

	ctx, stop := context.WithCancel(context.Background())
	defer stop()
	stream := s.From(ctx, 0)
	s.Enqueue("hello")

	started, ready := waitVariant(t, stream, 0)
	if started == nil || started.Index != 0 || started.Source != -1 {
		t.Errorf("variant_started = %+v, want first pass (idx 0, source -1)", started)
	}
	// Rewrite trims its output, so compare against the trimmed source.
	want := strings.TrimSpace(long)
	if ready.Content != want {
		t.Errorf("variant_ready content = %q, want the rewrite %q", ready.Content, want)
	}
	if ready.Error != "" {
		t.Errorf("variant_ready error = %q, want none", ready.Error)
	}
	if ready.HTML == "" {
		t.Errorf("variant_ready html = empty, want server-rendered markdown")
	}

	// The terminal pass is persisted: a reload renders the same tab.
	ps, err := s.Persisted()
	if err != nil {
		t.Fatalf("Persisted: %v", err)
	}
	if len(ps.Variants) != 1 || ps.Variants[0].Content != want || ps.Variants[0].PromptVersion != humanize.PromptVersion {
		t.Errorf("persisted variants = %+v, want the completed pass", ps.Variants)
	}
}

// TestManualHumanizeChainsFromLatest verifies the "+" path: a manual pass on
// a message that already has a variant chains from that variant (source = its
// index) and allocates the next index, so passes never collide.
func TestManualHumanizeChainsFromLatest(t *testing.T) {
	long := longReply()
	srv := llmStub(t, long)
	client := llm.NewClient(config.Config{BaseURL: srv.URL + "/v1", Model: "m", APIKey: "k"}, nil)
	nctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	st := NewStore(memPersister(t), nil, nctx)
	s, err := st.Create(client)
	if err != nil {
		t.Fatalf("Create: %v", err)
	}

	ctx, stop := context.WithCancel(context.Background())
	defer stop()
	stream := s.From(ctx, 0)
	// Commit the turn's messages directly; the long reply's commit triggers
	// the automatic first pass.
	if err := s.commit(llm.UserMessage("hello")); err != nil {
		t.Fatalf("commit user: %v", err)
	}
	if err := s.commit(llm.AssistantMessage(long, "", nil)); err != nil {
		t.Fatalf("commit assistant: %v", err)
	}
	waitVariant(t, stream, 0) // let the auto pass finish

	if err := s.HumanizeMessage(2); err != nil {
		t.Fatalf("HumanizeMessage: %v", err)
	}
	started, ready := waitVariant(t, stream, 1)
	if started == nil || started.Index != 1 || started.Source != 0 {
		t.Errorf("manual variant_started = %+v, want idx 1 chained from idx 0", started)
	}
	if ready.Index != 1 || ready.Source != 0 {
		t.Errorf("manual variant_ready = %+v, want idx 1 chained from idx 0", ready)
	}

	ps, err := s.Persisted()
	if err != nil {
		t.Fatalf("Persisted: %v", err)
	}
	if len(ps.Variants) != 2 || ps.Variants[1].Index != 1 || ps.Variants[1].Source != 0 {
		t.Errorf("persisted variants = %+v, want two chained passes", ps.Variants)
	}
}

// TestHumanizeSkipsShortReplies ensures the auto pass never fires for replies
// under the thresholds: no variant envelopes, no rows, no extra LLM calls.
func TestHumanizeSkipsShortReplies(t *testing.T) {
	srv := llmStub(t, longReply())
	client := llm.NewClient(config.Config{BaseURL: srv.URL + "/v1", Model: "m", APIKey: "k"}, nil)
	nctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	st := NewStore(memPersister(t), nil, nctx)
	s, err := st.Create(client)
	if err != nil {
		t.Fatalf("Create: %v", err)
	}

	ctx, stop := context.WithCancel(context.Background())
	defer stop()
	stream := s.From(ctx, 0)
	if err := s.commit(llm.UserMessage("hello")); err != nil {
		t.Fatalf("commit user: %v", err)
	}
	if err := s.commit(llm.AssistantMessage("done", "", nil)); err != nil {
		t.Fatalf("commit assistant: %v", err)
	}

	// Drain for a moment and assert no variant activity appeared.
	deadline := time.After(300 * time.Millisecond)
	for {
		select {
		case env := <-stream:
			if env.Kind == api.KindVariantStarted || env.Kind == api.KindVariant {
				t.Fatalf("short reply produced a variant envelope: %+v", env)
			}
		case <-deadline:
			ps, err := s.Persisted()
			if err != nil {
				t.Fatalf("Persisted: %v", err)
			}
			if len(ps.Variants) != 0 {
				t.Errorf("persisted variants = %+v, want none for a short reply", ps.Variants)
			}
			return
		}
	}
}
