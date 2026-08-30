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
	return newSession(id, nil, nil, d, rowID, time.Now().UnixMilli(), 0, nil)
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
	s.SetExecContext(api.ExecContext{System: "linux/amd64", CWD: "/work"})
	id := s.RegisterExec(make(chan api.ExecRequest, 1), "", "", "")
	s.UnregisterExec(id)

	snap := s.Snapshot()
	if len(snap.History) != 2 {
		t.Fatalf("history = %+v, want 2 system notices", snap.History)
	}
	if snap.History[0].Role != "system" || !strings.Contains(snap.History[0].Content, "execution provider connected") || !strings.Contains(snap.History[0].Content, "/work") {
		t.Errorf("connect notice = %+v", snap.History[0])
	}
	if snap.History[1].Role != "system" || !strings.Contains(snap.History[1].Content, "disconnected") {
		t.Errorf("disconnect notice = %+v", snap.History[1])
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
	s2 := newSession(s.ID(), nil, nil, s.persist, ps.ID, ps.CreatedAt, ps.ArchivedAt, nil)
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
	s.SetExecContext(api.ExecContext{ID: "laptop", System: "darwin/arm64", CWD: "/Users/mark"})
	s.RegisterExec(make(chan api.ExecRequest, 1), "laptop", "laptop", "remote")

	if err := s.SelectExec("local"); err != nil {
		t.Fatalf("SelectExec(local): %v", err)
	}
	snap := s.Snapshot()
	last := snap.History[len(snap.History)-1]
	if last.Role != "system" || !strings.Contains(last.Content, "execution provider") || !strings.Contains(last.Content, "local") {
		t.Errorf("select-local notice = %+v, want a system notice naming local", last)
	}

	if err := s.SelectExec("laptop"); err != nil {
		t.Fatalf("SelectExec(laptop): %v", err)
	}
	snap = s.Snapshot()
	last = snap.History[len(snap.History)-1]
	if last.Role != "system" || !strings.Contains(last.Content, "laptop") || !strings.Contains(last.Content, "darwin/arm64") {
		t.Errorf("select-laptop notice = %+v, want a system notice naming the laptop", last)
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
