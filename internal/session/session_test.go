package session

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"porter/internal/api"
	"porter/internal/config"
	"porter/internal/llm"
	"porter/internal/tools"
)

// commitN appends n distinct user messages and returns their seqs.
func commitN(t *testing.T, s *Session, prefix string, n int) {
	t.Helper()
	for i := 1; i <= n; i++ {
		s.commit(llm.UserMessage(fmt.Sprintf("%s%d", prefix, i)))
	}
}

func TestCommitAppendsHistoryAndSeq(t *testing.T) {
	s := newSession("s", nil, nil)
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
	s := newSession("s", nil, nil)
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
	s := newTestSession("s")
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
	s := newSession("s", nil, nil)
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
	s := newSession("s", nil, nil)
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
	s := newSession("s", nil, nil)
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

	st := NewStore(nctx)
	s := st.Create(client)
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

// TestSetProviderDefaultsToLocal verifies a fresh session resolves to a local
// provider and can swap to a registered one without racing.
// TestRunningAndQueueDepth covers the status-indicator primitives directly:
// a fresh session is idle with an empty queue, and enqueued messages are
// visible via QueueDepth without waiting for a turn to start.
func TestRunningAndQueueDepth(t *testing.T) {
	s := newTestSession("s")
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
	st := NewStore()
	s := st.Create(nil)

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

func newTestSession(id string) *Session {
	return newSession(id, nil, nil)
}
func TestTrackToolRunsForReconnect(t *testing.T) {
	s := newTestSession("s")

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
