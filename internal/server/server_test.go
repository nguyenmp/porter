package server

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"porter/internal/api"
	"porter/internal/client"
	"porter/internal/codec"
	"porter/internal/config"
	"porter/internal/llm"
	"porter/internal/recall"
	"porter/internal/tools"
)

// plainLLM serves an SSE reply "hi" with usage 1/2 for every request.
func plainLLM() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w,
			`data: {"choices":[{"delta":{"content":"hi"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":2}}`+"\n\n"+
				`data: [DONE]`+"\n")
	}
}

// newTestServer stands up a real porter server backed by the given fake LLM
// and its own temporary SQLite database (so tests never share or pollute the
// working-directory porter.db).
func newTestServer(t *testing.T, llmHandler http.HandlerFunc) *httptest.Server {
	t.Helper()
	_, ts := startServerDB(t, filepath.Join(t.TempDir(), "porter.db"), llmHandler)
	return ts
}

// startServerDB stands up a porter server on a specific database path and
// returns both the backing *Server (so a test can Close it to simulate a
// restart) and its HTTP endpoint. Both are cleaned up when the test ends.
func startServerDB(t *testing.T, dbPath string, llmHandler http.HandlerFunc) (*Server, *httptest.Server) {
	t.Helper()
	llmSrv := httptest.NewServer(llmHandler)
	s, err := newServer(config.Config{BaseURL: llmSrv.URL + "/v1", Model: "m", APIKey: "k"}, dbPath, "")
	if err != nil {
		t.Fatalf("newServer: %v", err)
	}
	ts := httptest.NewServer(s.Handler())
	t.Cleanup(func() {
		s.Close()
		ts.Close()
		llmSrv.Close()
	})
	return s, ts
}

// runOneTurn creates a session, appends a user message, subscribes until the
// turn completes, and returns the observed envelopes plus the history.
func runOneTurn(t *testing.T, base, prompt string) ([]api.Envelope, api.SessionHistory) {
	t.Helper()
	c := client.New(base)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	info, err := c.Create(ctx)
	if err != nil {
		t.Fatalf("Create: %v", err)
	}
	if err := c.Append(ctx, info.ID, prompt); err != nil {
		t.Fatalf("Append: %v", err)
	}

	var got []api.Envelope
	done := false
	err = c.Subscribe(ctx, info.ID, info.Seq, func(env api.Envelope) { got = append(got, env) },
		func(env api.Envelope) bool {
			if env.Kind == api.KindTurnDone {
				done = true
				return true
			}
			return false
		})
	if err != nil {
		t.Fatalf("Subscribe: %v", err)
	}
	if !done {
		t.Fatalf("no turn_completed observed; got %d envelopes", len(got))
	}
	h, err := c.History(ctx, info.ID)
	if err != nil {
		t.Fatalf("History: %v", err)
	}
	return got, h
}

func TestSessionCommitsHistory(t *testing.T) {
	srv := newTestServer(t, plainLLM())
	got, h := runOneTurn(t, srv.URL, "hello")

	if len(h.History) < 2 {
		t.Fatalf("history len = %d, want >= 2", len(h.History))
	}
	if h.History[0].Role != "user" || h.History[0].Content != "hello" {
		t.Errorf("history[0] = %+v, want user 'hello'", h.History[0])
	}
	if h.History[1].Role != "assistant" || !strings.Contains(h.History[1].Content, "hi") {
		t.Errorf("history[1] = %+v, want assistant reply 'hi'", h.History[1])
	}

	// The user message and the final assistant reply were both committed.
	var userCommit, turnDone bool
	for _, env := range got {
		if env.Kind == api.KindMessage {
			if env.Message != nil && env.Message.Role == "user" {
				userCommit = true
			}
		}
		if env.Kind == api.KindTurnDone {
			turnDone = true
			if env.CachedInput != 0 || env.UncachedInput != 1 || env.Output != 2 {
				t.Errorf("turn done usage = %d cached/%d miss/%d out, want 0/1/2", env.CachedInput, env.UncachedInput, env.Output)
			}
		}
	}
	if !userCommit {
		t.Errorf("bus missing committed user message; got %+v", got)
	}
	if !turnDone {
		t.Errorf("bus missing turn_completed")
	}
}

func TestSessionsAreIndependent(t *testing.T) {
	srv := newTestServer(t, plainLLM())
	_, h1 := runOneTurn(t, srv.URL, "first")
	_, h2 := runOneTurn(t, srv.URL, "second")

	if len(h1.History) < 2 {
		t.Errorf("session 1 history len = %d, want >= 2", len(h1.History))
	}
	if len(h2.History) < 2 {
		t.Errorf("session 2 history len = %d, want >= 2", len(h2.History))
	}
}

// toolThenReplyLLM asks for a shell tool call on the first request, then replies
// plainly on the second.
func toolThenReplyLLM() http.HandlerFunc {
	var mu sync.Mutex
	n := 0
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		mu.Lock()
		n++
		call := n
		mu.Unlock()
		if call == 1 {
			fmt.Fprint(w,
				`data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1","type":"function","function":{"name":"shell","arguments":"{\"command\":\"echo hi\"}"}}]},"finish_reason":"tool_calls"}]}`+"\n\n"+
					`data: [DONE]`+"\n")
			return
		}
		fmt.Fprint(w,
			`data: {"choices":[{"delta":{"content":"done"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1}}`+"\n\n"+
				`data: [DONE]`+"\n")
	}
}

// TestToolResultEnvelope ensures a tool result — a system-side fact, not LLM
// output — is delivered as its own envelope kind on the bus.
func TestToolResultEnvelope(t *testing.T) {
	srv := newTestServer(t, toolThenReplyLLM())
	got, _ := runOneTurn(t, srv.URL, "run it")

	var sawToolResult bool
	for _, env := range got {
		if env.Kind == api.KindToolResult {
			sawToolResult = true
			if env.Name != "shell" {
				t.Errorf("tool result name = %q, want shell", env.Name)
			}
			if !strings.Contains(env.Result, "exit code: 0") || !strings.Contains(env.Result, "hi") {
				t.Errorf("tool result = %q, want exit code 0 with echo output", env.Result)
			}
		}
	}
	if !sawToolResult {
		t.Errorf("bus missing a tool_result envelope; got %+v", got)
	}
}

// TestRunsReportsInFlightTool tests the reconnect story end to end: a tool that
// is still running appears on /runs with the server's clock and the partial
// output accumulated so far, and disappears once it completes.
func TestRunsReportsInFlightTool(t *testing.T) {
	srv := newTestServer(t, toolThenReplyLLM())
	c := client.New(srv.URL)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	info, err := c.Create(ctx)
	if err != nil {
		t.Fatalf("Create: %v", err)
	}

	// Register as the execution provider; the dispatch hands back a pipe we
	// control, so the tool "runs" until we write output and close it.
	pr, pw := io.Pipe()
	dispatch := func(ctx context.Context, name string, args []byte) (io.ReadCloser, error) {
		return pr, nil
	}
	go func() { _ = c.ServeExec(ctx, info.ID, dispatch) }()
	time.Sleep(100 * time.Millisecond)

	if err := c.Append(ctx, info.ID, "run it"); err != nil {
		t.Fatalf("Append: %v", err)
	}

	// The tool is running on the server: /runs must report it.
	var run api.RunInfo
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		rr, err := c.Runs(ctx, info.ID)
		if err != nil {
			t.Fatalf("Runs: %v", err)
		}
		if len(rr.Runs) > 0 {
			if rr.Now <= 0 {
				t.Errorf("Runs now = %d, want server clock", rr.Now)
			}
			run = rr.Runs[0]
			break
		}
		time.Sleep(20 * time.Millisecond)
	}
	if run.CallID == "" {
		t.Fatal("tool never appeared on /runs")
	}
	if run.Name != "shell" || !strings.Contains(run.Arguments, "echo hi") {
		t.Errorf("in-flight run = %+v, want shell + echo hi args", run)
	}
	if run.StartedAt <= 0 {
		t.Errorf("in-flight run started_at = %d, want server clock", run.StartedAt)
	}

	// Stream partial output; it must accumulate on the server.
	_, _ = pw.Write([]byte("partial line 1\n"))
	deadline = time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		rr, _ := c.Runs(ctx, info.ID)
		if len(rr.Runs) > 0 && strings.Contains(rr.Runs[0].Output, "partial line 1") {
			break
		}
		time.Sleep(20 * time.Millisecond)
	}
	rr, err := c.Runs(ctx, info.ID)
	if err != nil {
		t.Fatalf("Runs: %v", err)
	}
	if len(rr.Runs) != 1 || !strings.Contains(rr.Runs[0].Output, "partial line 1") {
		t.Fatalf("runs after partial output = %+v, want accumulated output", rr.Runs)
	}

	// Complete the tool: the run leaves /runs and history gains the result.
	_, _ = pw.Write([]byte("rest\n"))
	_ = pw.Close()
	deadline = time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		rr, _ = c.Runs(ctx, info.ID)
		if len(rr.Runs) == 0 {
			break
		}
		time.Sleep(20 * time.Millisecond)
	}
	if len(rr.Runs) != 0 {
		t.Errorf("runs after completion = %+v, want empty", rr.Runs)
	}
	h, err := c.History(ctx, info.ID)
	if err != nil {
		t.Fatalf("History: %v", err)
	}
	var found bool
	for _, m := range h.History {
		if m.Role == "tool" && strings.Contains(m.Content, "partial line 1") {
			found = true
		}
	}
	if !found {
		t.Errorf("committed tool result missing partial output; history = %+v", h.History)
	}
}

// slowStreamLLM streams two content deltas with delays so a test can observe
// the in-flight live tail mid-stream (the turn stays uncommitted while the
// deltas accumulate server-side).
func slowStreamLLM() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		f, _ := w.(http.Flusher)
		write := func(s string) {
			fmt.Fprint(w, s)
			if f != nil {
				f.Flush()
			}
		}
		write(`data: {"choices":[{"delta":{"content":"Hel"},"finish_reason":null}]}` + "\n\n")
		time.Sleep(300 * time.Millisecond)
		write(`data: {"choices":[{"delta":{"content":"lo"},"finish_reason":null}]}` + "\n\n")
		time.Sleep(300 * time.Millisecond)
		write(`data: {"choices":[{"delta":{"content":""},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":2}}` + "\n\n")
		write(`data: [DONE]` + "\n")
	}
}

// TestLiveReturnsInFlightStream drives a real streaming turn and checks
// GET /live serves the in-flight LLM tail (with monotonic live positions)
// while the model is mid-stream, then empties once the turn commits — the
// re-seed a mid-turn reload uses to catch the active stream.
func TestLiveReturnsInFlightStream(t *testing.T) {
	s, ts := startServerDB(t, filepath.Join(t.TempDir(), "porter.db"), slowStreamLLM())

	ses, err := s.store.Create(s.client)
	if err != nil {
		t.Fatalf("Create: %v", err)
	}
	ses.Enqueue("stream it")

	var lr api.LiveResponse
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		resp, err := http.Get(ts.URL + "/api/sessions/" + ses.ID() + "/live")
		if err != nil {
			t.Fatalf("GET /live: %v", err)
		}
		err = json.NewDecoder(resp.Body).Decode(&lr)
		resp.Body.Close()
		if err != nil {
			t.Fatalf("decode /live: %v", err)
		}
		if len(lr.Events) >= 2 {
			break
		}
		time.Sleep(20 * time.Millisecond)
	}
	if len(lr.Events) != 2 {
		t.Fatalf("/live had %d events %+v, want the 2 streamed deltas", len(lr.Events), lr.Events)
	}
	if lr.Seq != 2 {
		t.Errorf("/live seq = %d, want 2", lr.Seq)
	}
	if lr.Events[0].LiveSeq != 1 || lr.Events[0].Event.Delta != "Hel" {
		t.Errorf("event 0 = %+v, want live_seq 1 delta Hel", lr.Events[0])
	}
	if lr.Events[1].LiveSeq != 2 || lr.Events[1].Event.Delta != "lo" {
		t.Errorf("event 1 = %+v, want live_seq 2 delta lo", lr.Events[1])
	}

	// Once the turn commits, the tail is superseded by the committed message.
	deadline = time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		resp, err := http.Get(ts.URL + "/api/sessions/" + ses.ID() + "/live")
		if err != nil {
			t.Fatalf("GET /live: %v", err)
		}
		err = json.NewDecoder(resp.Body).Decode(&lr)
		resp.Body.Close()
		if err != nil {
			t.Fatalf("decode /live: %v", err)
		}
		if len(lr.Events) == 0 {
			return
		}
		time.Sleep(20 * time.Millisecond)
	}
	t.Fatal("live tail never cleared after the turn committed")
}

// TestLiveUnknownSession checks GET /live on a missing session returns 404.
func TestLiveUnknownSession(t *testing.T) {
	srv := newTestServer(t, plainLLM())
	resp, err := http.Get(srv.URL + "/api/sessions/session_999/live")
	if err != nil {
		t.Fatalf("GET /live: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusNotFound {
		t.Errorf("status = %d, want 404", resp.StatusCode)
	}
}

// TestServerRunsToolViaExecProvider verifies that a client registered as the
// session's execution provider (via ServeExec) runs the agent's tool calls and
// streams results back into History, instead of the server running them.
func TestServerRunsToolViaExecProvider(t *testing.T) {
	srv := newTestServer(t, toolThenReplyLLM())
	c := client.New(srv.URL)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel() // close the exec connection before server cleanup

	info, err := c.Create(ctx)
	if err != nil {
		t.Fatalf("Create: %v", err)
	}

	// Register as the session's execution provider, running tools in-process
	// and recording that we did.
	var mu sync.Mutex
	ranOnClient := false
	dispatch := func(ctx context.Context, name string, args []byte) (io.ReadCloser, error) {
		mu.Lock()
		ranOnClient = true
		mu.Unlock()
		return tools.NewDispatcher().Run(ctx, name, args)
	}
	go func() { _ = c.ServeExec(ctx, info.ID, dispatch) }()
	// Give the registration round-trip a moment to land before the turn runs.
	time.Sleep(100 * time.Millisecond)

	if err := c.Append(ctx, info.ID, "run it"); err != nil {
		t.Fatalf("Append: %v", err)
	}
	var h api.SessionHistory
	done := false
	err = c.Subscribe(ctx, info.ID, info.Seq, nil, func(env api.Envelope) bool {
		if env.Kind == api.KindTurnDone {
			done = true
			return true
		}
		return false
	})
	if err != nil {
		t.Fatalf("Subscribe: %v", err)
	}
	if !done {
		t.Fatal("no turn_completed observed")
	}
	h, err = c.History(ctx, info.ID)
	if err != nil {
		t.Fatalf("History: %v", err)
	}

	mu.Lock()
	ran := ranOnClient
	mu.Unlock()
	if !ran {
		t.Errorf("tool call did not run through the registered client")
	}
	var found bool
	for _, m := range h.History {
		if m.Role == "tool" && strings.Contains(m.Content, "hi") {
			found = true
		}
	}
	if !found {
		t.Errorf("expected a committed tool result; history = %+v", h.History)
	}
}

func TestUnknownSession(t *testing.T) {
	srv := newTestServer(t, plainLLM())
	c := client.New(srv.URL)
	ctx := context.Background()

	if _, err := c.History(ctx, "nope"); err == nil {
		t.Errorf("History for unknown session should error")
	}
	if err := c.Append(ctx, "nope", "hi"); err == nil {
		t.Errorf("Append for unknown session should error")
	}
	// Subscribe for an unknown session: fetch directly to check status.
	resp, err := http.Get(srv.URL + "/api/sessions/nope/events")
	if err != nil || resp.StatusCode != http.StatusNotFound {
		t.Errorf("events for unknown session: status = %v, err = %v", resp.StatusCode, err)
	}
	if resp != nil {
		resp.Body.Close()
	}
}

func TestEmptyAppendRejected(t *testing.T) {
	srv := newTestServer(t, plainLLM())
	c := client.New(srv.URL)
	ctx := context.Background()
	info, err := c.Create(ctx)
	if err != nil {
		t.Fatalf("Create: %v", err)
	}
	// Two empty appends (one blank, one whitespace) must both be rejected.
	for _, v := range []string{"", "   "} {
		body, _ := json.Marshal(api.AppendRequest{Content: v})
		req, _ := http.NewRequestWithContext(ctx, http.MethodPost, srv.URL+"/api/sessions/"+info.ID+"/messages", strings.NewReader(string(body)))
		req.Header.Set("Content-Type", "application/json")
		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			t.Fatalf("append: %v", err)
		}
		defer resp.Body.Close()
		if resp.StatusCode != http.StatusBadRequest {
			t.Errorf("empty append status = %d, want 400", resp.StatusCode)
		}
	}
}
func TestListSessions(t *testing.T) {
	srv := newTestServer(t, plainLLM())
	c := client.New(srv.URL)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	// Two sessions, created with a gap so their order is deterministic; the
	// older one gets a user message so its preview is populated.
	oldInfo, err := c.Create(ctx)
	if err != nil {
		t.Fatalf("Create: %v", err)
	}
	if err := c.Append(ctx, oldInfo.ID, "first message here"); err != nil {
		t.Fatalf("Append: %v", err)
	}
	time.Sleep(10 * time.Millisecond)
	newInfo, err := c.Create(ctx)
	if err != nil {
		t.Fatalf("Create: %v", err)
	}

	resp, err := http.Get(srv.URL + api.SessionsPath)
	if err != nil {
		t.Fatalf("GET %s: %v", api.SessionsPath, err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200", resp.StatusCode)
	}
	var list api.SessionsResponse
	if err := json.NewDecoder(resp.Body).Decode(&list); err != nil {
		t.Fatalf("decode list: %v", err)
	}
	if len(list.Sessions) != 2 {
		t.Fatalf("got %d sessions, want 2: %+v", len(list.Sessions), list.Sessions)
	}
	// Newest first.
	if list.Sessions[0].ID != newInfo.ID {
		t.Errorf("first = %q, want newest %q", list.Sessions[0].ID, newInfo.ID)
	}
	if list.Sessions[1].ID != oldInfo.ID {
		t.Errorf("second = %q, want oldest %q", list.Sessions[1].ID, oldInfo.ID)
	}
	// Preview comes from the first user message; the empty session has none.
	if list.Sessions[1].Preview != "first message here" {
		t.Errorf("preview = %q, want %q", list.Sessions[1].Preview, "first message here")
	}
	if list.Sessions[0].Preview != "" {
		t.Errorf("new session should have no preview, got %q", list.Sessions[0].Preview)
	}
	// CreatedAt timestamps should be populated and in descending order.
	if list.Sessions[0].CreatedAt < list.Sessions[1].CreatedAt {
		t.Errorf("CreatedAt not newest-first: %d < %d", list.Sessions[0].CreatedAt, list.Sessions[1].CreatedAt)
	}
}

// TestQueueDepthOnUserCommit verifies each user message_committed reports how
// many turns are still queued behind it. Turn 1's tool is held open so the
// remaining messages pile up deterministically; releasing it lets the queue
// drain one turn at a time, and the subscriber observes queue depths 2, 1, 0.
func TestQueueDepthOnUserCommit(t *testing.T) {
	srv := newTestServer(t, toolThenReplyLLM())
	c := client.New(srv.URL)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	info, err := c.Create(ctx)
	if err != nil {
		t.Fatalf("Create: %v", err)
	}

	// Hold a tool open (the exec provider's pipe) so turn 1 stays running.
	pr, pw := io.Pipe()
	dispatch := func(ctx context.Context, name string, args []byte) (io.ReadCloser, error) {
		return pr, nil
	}
	go func() { _ = c.ServeExec(ctx, info.ID, dispatch) }()
	time.Sleep(100 * time.Millisecond)

	// Subscribe before any message is sent so every commit is delivered live
	// (or replayed from since=0 for anything committed before registration).
	var mu sync.Mutex
	var got []api.Envelope
	turnDone := 0
	subDone := make(chan error, 1)
	go func() {
		subDone <- c.Subscribe(ctx, info.ID, info.Seq, func(env api.Envelope) {
			mu.Lock()
			got = append(got, env)
			mu.Unlock()
		}, func(env api.Envelope) bool {
			mu.Lock()
			defer mu.Unlock()
			if env.Kind == api.KindTurnDone {
				turnDone++
			}
			return turnDone >= 4 // turn 1 plus the three queued turns
		})
	}()
	time.Sleep(100 * time.Millisecond) // let the subscription register

	// Turn 1: committed (queue empty), tool held open.
	if err := c.Append(ctx, info.ID, "a"); err != nil {
		t.Fatalf("Append(a): %v", err)
	}
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		rr, err := c.Runs(ctx, info.ID)
		if err != nil {
			t.Fatalf("Runs: %v", err)
		}
		if len(rr.Runs) > 0 {
			break
		}
		time.Sleep(20 * time.Millisecond)
	}
	// Queue three more while turn 1 runs.
	for _, content := range []string{"b", "c", "d"} {
		if err := c.Append(ctx, info.ID, content); err != nil {
			t.Fatalf("Append(%q): %v", content, err)
		}
	}

	// Release turn 1; the remaining turns drain serially.
	_, _ = pw.Write([]byte("rest\n"))
	_ = pw.Close()

	select {
	case err := <-subDone:
		if err != nil {
			t.Fatalf("Subscribe: %v", err)
		}
	case <-time.After(10 * time.Second):
		t.Fatal("timed out waiting for turns to drain")
	}

	mu.Lock()
	defer mu.Unlock()
	var queues []int
	for _, env := range got {
		if env.Kind == api.KindMessage && env.Message != nil && env.Message.Role == "user" {
			queues = append(queues, env.Queue)
		}
	}
	// Turn 1 starts with an empty queue; b/c/d then report 2/1/0 left behind.
	want := []int{0, 2, 1, 0}
	if len(queues) != len(want) {
		t.Fatalf("user commit queues = %v, want %v", queues, want)
	}
	for i, q := range queues {
		if q != want[i] {
			t.Errorf("user commit %d queue = %d, want %d", i, q, want[i])
		}
	}
}

// TestIndexReportsRunningAndQueue verifies the page seeds the status indicator
// with the live turn state at render time: running=true while a turn's tool is
// in flight, and queue=1 when a second message is waiting behind it.
func TestIndexReportsRunningAndQueue(t *testing.T) {
	srv := newTestServer(t, toolThenReplyLLM())
	c := client.New(srv.URL)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	info, err := c.Create(ctx)
	if err != nil {
		t.Fatalf("Create: %v", err)
	}

	// Hold a tool open (the exec provider's pipe) so turn 1 stays running
	// while we inspect the rendered page.
	pr, pw := io.Pipe()
	dispatch := func(ctx context.Context, name string, args []byte) (io.ReadCloser, error) {
		return pr, nil
	}
	go func() { _ = c.ServeExec(ctx, info.ID, dispatch) }()
	time.Sleep(100 * time.Millisecond)

	if err := c.Append(ctx, info.ID, "run it"); err != nil {
		t.Fatalf("Append: %v", err)
	}

	// Wait until turn 1 is visibly running (its tool is in-flight).
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		rr, err := c.Runs(ctx, info.ID)
		if err != nil {
			t.Fatalf("Runs: %v", err)
		}
		if len(rr.Runs) > 0 {
			break
		}
		time.Sleep(20 * time.Millisecond)
	}
	if err := c.Append(ctx, info.ID, "second"); err != nil {
		t.Fatalf("Append: %v", err)
	}

	resp, err := http.Get(srv.URL + "/?session=" + info.ID)
	if err != nil {
		t.Fatalf("GET /: %v", err)
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	s := string(body)
	if !strings.Contains(s, `id="status"`) {
		t.Errorf("index does not contain the status indicator")
	}
	if !strings.Contains(s, `data-running="true"`) {
		t.Errorf("index does not report the running turn (data-running=true)")
	}
	if !strings.Contains(s, `data-queue="1"`) {
		t.Errorf("index does not report queue depth 1 (data-queue=1)")
	}

	// Let turn 1 finish so the session drains before the test ends.
	_, _ = pw.Write([]byte("rest\n"))
	_ = pw.Close()
}

func TestCreateReturnsAllFields(t *testing.T) {
	srv := newTestServer(t, plainLLM())
	c := client.New(srv.URL)
	ctx := context.Background()

	info, err := c.Create(ctx)
	if err != nil {
		t.Fatalf("Create: %v", err)
	}
	if info.ID == "" {
		t.Error("ID is empty")
	}
	if info.History == nil {
		t.Error("History is nil; want non-nil (empty slice for new session)")
	}
	if len(info.History) != 0 {
		t.Errorf("History len = %d, want 0 for new session", len(info.History))
	}
	if info.Seq != 0 {
		t.Errorf("Seq = %d, want 0 for new session", info.Seq)
	}
}

func TestJSONContentType(t *testing.T) {
	srv := newTestServer(t, plainLLM())
	c := client.New(srv.URL)
	ctx := context.Background()

	// POST /api/sessions should return application/json.
	info, err := c.Create(ctx)
	if err != nil {
		t.Fatalf("Create: %v", err)
	}

	// Verify Create response Content-Type by making a raw request.
	createResp, err := http.Post(srv.URL+api.SessionsPath, "application/json", nil)
	if err != nil {
		t.Fatalf("POST sessions: %v", err)
	}
	defer createResp.Body.Close()
	if ct := createResp.Header.Get("Content-Type"); !strings.HasPrefix(ct, "application/json") {
		t.Errorf("Create Content-Type = %q, want application/json", ct)
	}

	// GET /api/sessions/{id} should return application/json.
	histResp, err := http.Get(srv.URL + "/api/sessions/" + info.ID)
	if err != nil {
		t.Fatalf("GET history: %v", err)
	}
	defer histResp.Body.Close()
	if ct := histResp.Header.Get("Content-Type"); !strings.HasPrefix(ct, "application/json") {
		t.Errorf("History Content-Type = %q, want application/json", ct)
	}
}

func TestIndexNoSessionRendersEmptyState(t *testing.T) {
	srv := newTestServer(t, plainLLM())

	// GET / with no session param should render the empty state rather than
	// auto-creating a throwaway session: no redirect, no chat div, but a
	// New chat button to create one explicitly.
	resp, err := http.Get(srv.URL + "/")
	if err != nil {
		t.Fatalf("GET /: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200", resp.StatusCode)
	}
	ct := resp.Header.Get("Content-Type")
	if !strings.HasPrefix(ct, "text/html") {
		t.Errorf("Content-Type = %q, want text/html", ct)
	}
	body, _ := io.ReadAll(resp.Body)
	s := string(body)
	if !strings.Contains(s, "<title>porter</title>") {
		t.Errorf("response does not contain <title>porter</title>")
	}
	if strings.Contains(s, `id="chat"`) {
		t.Errorf("empty state should not contain #chat div")
	}
	if !strings.Contains(s, "New chat") {
		t.Errorf("empty state should offer a New chat button")
	}
	if !strings.Contains(s, `hx-post="/api/sessions"`) {
		t.Errorf("empty state should include the New chat form posting to /api/sessions")
	}
}

func TestIndexPassesSessionParam(t *testing.T) {
	srv := newTestServer(t, plainLLM())
	c := client.New(srv.URL)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	info, err := c.Create(ctx)
	if err != nil {
		t.Fatalf("Create: %v", err)
	}

	resp, err := http.Get(srv.URL + "/?session=" + info.ID)
	if err != nil {
		t.Fatalf("GET /: %v", err)
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	s := string(body)
	if !strings.Contains(s, info.ID) {
		t.Errorf("response does not contain session id %q", info.ID)
	}
}

func TestIndexContainsSidebar(t *testing.T) {
	srv := newTestServer(t, plainLLM())
	c := client.New(srv.URL)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	info, err := c.Create(ctx)
	if err != nil {
		t.Fatalf("Create: %v", err)
	}

	resp, err := http.Get(srv.URL + "/?session=" + info.ID)
	if err != nil {
		t.Fatalf("GET /: %v", err)
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	s := string(body)
	if !strings.Contains(s, `id="app" data-session="`+info.ID+`"`) {
		t.Errorf("index does not carry the active session on #app")
	}
	if !strings.Contains(s, `class="sidebar"`) {
		t.Errorf("index does not contain a sidebar nav")
	}
	if !strings.Contains(s, `id="session-list"`) {
		t.Errorf("index does not contain the sidebar session list")
	}
	if strings.Contains(s, "localStorage") {
		t.Errorf("index script should not keep a client-side session registry")
	}
	if !strings.Contains(s, "fetch('/api/sessions')") {
		t.Errorf("index script does not fetch the session list from the server")
	}
	if !strings.Contains(s, "+ New chat") {
		t.Errorf("sidebar does not contain the New chat button")
	}
	if !strings.Contains(s, `id="status"`) {
		t.Errorf("index does not contain the connection status indicator")
	}
	if !strings.Contains(s, "htmx:sse-error") {
		t.Errorf("index does not wire an SSE error handler for connection status")
	}
	if !strings.Contains(s, `id="menu-btn"`) {
		t.Errorf("index does not contain the mobile sidebar hamburger")
	}
	if !strings.Contains(s, `id="sidebar-backdrop"`) {
		t.Errorf("index does not contain the mobile sidebar backdrop")
	}
	if !strings.Contains(s, "sidebar-open") {
		t.Errorf("index does not toggle the sidebar drawer via the sidebar-open class")
	}
	if !strings.Contains(s, "@media (max-width: 767px)") {
		t.Errorf("index does not ship the mobile layout media query")
	}
	if !strings.Contains(s, "safe-area-inset-bottom") {
		t.Errorf("index does not account for mobile safe-area insets")
	}
	// viewport-fit=cover is what makes the env(safe-area-inset-*) values above
	// actually take effect on iOS Safari (they are 0 without it), so the mobile
	// layout's notch/home-indicator handling is inert unless it is present.
	if !strings.Contains(s, `content="width=device-width, initial-scale=1, viewport-fit=cover"`) {
		t.Errorf("index viewport meta does not enable viewport-fit=cover for iOS safe-area insets")
	}
}
func TestIndexEmptyStateHasNoActiveSession(t *testing.T) {
	srv := newTestServer(t, plainLLM())

	resp, err := http.Get(srv.URL + "/")
	if err != nil {
		t.Fatalf("GET /: %v", err)
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	s := string(body)
	if !strings.Contains(s, `id="app" data-session=""`) {
		t.Errorf("empty state should carry an empty data-session on #app")
	}
	if !strings.Contains(s, `id="session-list"`) {
		t.Errorf("empty state should still render the sidebar session list")
	}
}

// runOneTurnID is like runOneTurn but also returns the session id.
func runOneTurnID(t *testing.T, base, prompt string) (string, []api.Envelope, api.SessionHistory) {
	t.Helper()
	c := client.New(base)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	info, err := c.Create(ctx)
	if err != nil {
		t.Fatalf("Create: %v", err)
	}
	if err := c.Append(ctx, info.ID, prompt); err != nil {
		t.Fatalf("Append: %v", err)
	}

	var got []api.Envelope
	done := false
	err = c.Subscribe(ctx, info.ID, info.Seq, func(env api.Envelope) { got = append(got, env) },
		func(env api.Envelope) bool {
			if env.Kind == api.KindTurnDone {
				done = true
				return true
			}
			return false
		})
	if err != nil {
		t.Fatalf("Subscribe: %v", err)
	}
	if !done {
		t.Fatalf("no turn_completed observed; got %d envelopes", len(got))
	}
	h, err := c.History(ctx, info.ID)
	if err != nil {
		t.Fatalf("History: %v", err)
	}
	return info.ID, got, h
}

func TestViewRendersHistory(t *testing.T) {
	srv := newTestServer(t, plainLLM())
	id, _, h := runOneTurnID(t, srv.URL, "hello")
	_ = h

	// Fetch the view fragment for the session.
	resp, err := http.Get(srv.URL + "/api/sessions/" + id + "/view")
	if err != nil {
		t.Fatalf("GET view: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200", resp.StatusCode)
	}
	body, _ := io.ReadAll(resp.Body)
	s := string(body)
	if !strings.Contains(s, `class="msg msg-user"`) {
		t.Errorf("view does not contain user message div")
	}
	if !strings.Contains(s, `class="msg msg-assistant"`) {
		t.Errorf("view does not contain assistant message div")
	}
	if !strings.Contains(s, "hello") {
		t.Errorf("view does not contain 'hello'")
	}
	if !strings.Contains(s, "hi") {
		t.Errorf("view does not contain 'hi'")
	}
}

func TestViewRendersToolResultTiming(t *testing.T) {
	srv := newTestServer(t, toolThenReplyLLM())
	id, got, _ := runOneTurnID(t, srv.URL, "run it")

	// Sanity: the live bus carried the start and terminal envelopes with clocks.
	var sawStart, sawTerminal bool
	for _, env := range got {
		switch env.Kind {
		case api.KindToolStarted:
			sawStart = env.StartedAt > 0
		case api.KindToolResult:
			sawTerminal = env.StartedAt > 0 && env.FinishedAt >= env.StartedAt
		}
	}
	if !sawStart {
		t.Errorf("bus missing tool_started with start clock")
	}
	if !sawTerminal {
		t.Errorf("bus missing terminal tool_result with start/finish clocks")
	}

	// The committed /view must render the result with its call context (name +
	// args snippet) and a server-derived duration.
	resp, err := http.Get(srv.URL + "/api/sessions/" + id + "/view")
	if err != nil {
		t.Fatalf("GET view: %v", err)
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	s := string(body)
	// The args snippet is HTML-escaped in the summary (deliberately, like all
	// other tool text), so assert on the escaped form.
	for _, want := range []string{
		`tool call: shell`,   // tool name in the call header
		`tool result: shell`, // tool name in the result header
		`echo hi`,            // args snippet from the call
		`&#34;command&#34;`,  // escaped JSON in the args snippet
		`exit_code: 0 · `,    // exit status + duration marker in the result header
		`title="`,            // wall-clock tooltip
	} {
		if !strings.Contains(s, want) {
			t.Errorf("view missing %q; got:\n%s", want, s)
		}
	}
}

func TestViewNotFound(t *testing.T) {
	srv := newTestServer(t, plainLLM())
	resp, err := http.Get(srv.URL + "/api/sessions/nope/view")
	if err != nil {
		t.Fatalf("GET view: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusNotFound {
		t.Errorf("status = %d, want 404", resp.StatusCode)
	}
}

func TestIndexConnectsSSEWhenSessionSet(t *testing.T) {
	srv := newTestServer(t, plainLLM())
	c := client.New(srv.URL)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	info, err := c.Create(ctx)
	if err != nil {
		t.Fatalf("Create: %v", err)
	}

	resp, err := http.Get(srv.URL + "/?session=" + info.ID)
	if err != nil {
		t.Fatalf("GET /: %v", err)
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	s := string(body)
	wantConnect := fmt.Sprintf(`sse-connect="/api/sessions/%s/stream?since=`, info.ID)
	if !strings.Contains(s, wantConnect) {
		t.Errorf("index does not contain sse-connect for session stream; got: %s", s)
	}
	if !strings.Contains(s, `hx-get="/api/sessions/`+info.ID+`/view"`) {
		t.Errorf("index does not contain hx-get for initial session view")
	}
	// The SSE handler must be wired via the htmx 2.0.6 attribute form
	// (kebab-case event name). The old combined `hx-on="evt: fn"` syntax and
	// the camelCase `hx-on:htmx:sseMessage` form are both no-ops in htmx 2.x.
	if !strings.Contains(s, `hx-on:htmx:sse-message="sseMessage(event)"`) {
		t.Errorf("index does not contain hx-on:htmx:sse-message handler")
	}
}

func TestIndexContainsMessageForm(t *testing.T) {
	srv := newTestServer(t, plainLLM())
	c := client.New(srv.URL)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	info, err := c.Create(ctx)
	if err != nil {
		t.Fatalf("Create: %v", err)
	}

	resp, err := http.Get(srv.URL + "/?session=" + info.ID)
	if err != nil {
		t.Fatalf("GET /: %v", err)
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	s := string(body)
	if !strings.Contains(s, `hx-post="/api/sessions/`+info.ID+`/messages"`) {
		t.Errorf("index does not contain hx-post for message form")
	}
	if !strings.Contains(s, `name="content"`) {
		t.Errorf("index does not contain content input field")
	}
	if !strings.Contains(s, `class="msg-form"`) {
		t.Errorf("index does not contain msg-form class")
	}
}

func TestFormEncodedAppend(t *testing.T) {
	srv := newTestServer(t, plainLLM())
	c := client.New(srv.URL)
	ctx := context.Background()
	info, err := c.Create(ctx)
	if err != nil {
		t.Fatalf("Create: %v", err)
	}
	// Submit as form-encoded, the way HTMX would.
	form := url.Values{"content": {"hello from form"}}
	req, _ := http.NewRequestWithContext(ctx, http.MethodPost, srv.URL+"/api/sessions/"+info.ID+"/messages", strings.NewReader(form.Encode()))
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("POST form: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusAccepted {
		t.Errorf("form append status = %d, want 202", resp.StatusCode)
	}
	// Verify the message was queued by checking history after the turn completes.
	done := false
	err = c.Subscribe(ctx, info.ID, info.Seq, nil, func(env api.Envelope) bool {
		if env.Kind == api.KindTurnDone {
			done = true
			return true
		}
		return false
	})
	if err != nil {
		t.Fatalf("Subscribe: %v", err)
	}
	if !done {
		t.Fatal("no turn_completed observed")
	}
	h, err := c.History(ctx, info.ID)
	if err != nil {
		t.Fatalf("History: %v", err)
	}
	if len(h.History) < 1 || h.History[0].Role != "user" || h.History[0].Content != "hello from form" {
		t.Errorf("history[0] = %+v, want user 'hello from form'", h.History[0])
	}
}

func TestFormEncodedAppendEmptyRejected(t *testing.T) {
	srv := newTestServer(t, plainLLM())
	c := client.New(srv.URL)
	ctx := context.Background()
	info, err := c.Create(ctx)
	if err != nil {
		t.Fatalf("Create: %v", err)
	}
	form := url.Values{"content": {"  "}}
	req, _ := http.NewRequestWithContext(ctx, http.MethodPost, srv.URL+"/api/sessions/"+info.ID+"/messages", strings.NewReader(form.Encode()))
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("POST form: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusBadRequest {
		t.Errorf("empty form append status = %d, want 400", resp.StatusCode)
	}
}

func TestJSONAppendNoHXTrigger(t *testing.T) {
	// JSON append is the existing client path; it should accept the message
	// without setting the stale HX-Trigger header from the old polling UI
	// (live updates now arrive over SSE).
	srv := newTestServer(t, plainLLM())
	c := client.New(srv.URL)
	ctx := context.Background()
	info, err := c.Create(ctx)
	if err != nil {
		t.Fatalf("Create: %v", err)
	}
	body, _ := json.Marshal(api.AppendRequest{Content: "hi"})
	req, _ := http.NewRequestWithContext(ctx, http.MethodPost, srv.URL+"/api/sessions/"+info.ID+"/messages", strings.NewReader(string(body)))
	req.Header.Set("Content-Type", "application/json")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("POST json: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusAccepted {
		t.Errorf("JSON append status = %d, want 202", resp.StatusCode)
	}
	if resp.Header.Get("HX-Trigger") != "" {
		t.Errorf("JSON append set stale HX-Trigger = %q, want empty", resp.Header.Get("HX-Trigger"))
	}
}

// readSSE reads Server-Sent Events from r, stopping once a turn_completed
// envelope is assembled. It returns the parsed envelopes in order.
func readSSE(t *testing.T, r io.Reader) []api.Envelope {
	t.Helper()
	var out []api.Envelope
	sc := bufio.NewScanner(r)
	var pending *api.Envelope
	for sc.Scan() {
		line := sc.Text()
		if line == "" {
			if pending != nil {
				out = append(out, *pending)
				done := pending.Kind == api.KindTurnDone
				pending = nil
				if done {
					return out
				}
			}
			continue
		}
		if strings.HasPrefix(line, "event: ") {
			if pending == nil {
				pending = &api.Envelope{}
			}
			pending.Kind = strings.TrimPrefix(line, "event: ")
			continue
		}
		if strings.HasPrefix(line, "data: ") {
			payload := strings.TrimPrefix(line, "data: ")
			if pending == nil {
				pending = &api.Envelope{}
			}
			if err := json.Unmarshal([]byte(payload), pending); err != nil {
				t.Fatalf("unmarshal sse data: %v", err)
			}
			continue
		}
	}
	if err := sc.Err(); err != nil {
		t.Fatalf("scan sse: %v", err)
	}
	if pending != nil {
		out = append(out, *pending)
	}
	return out
}

func TestStreamSSEEvents(t *testing.T) {
	srv := newTestServer(t, plainLLM())
	c := client.New(srv.URL)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	info, err := c.Create(ctx)
	if err != nil {
		t.Fatalf("Create: %v", err)
	}
	if err := c.Append(ctx, info.ID, "hello"); err != nil {
		t.Fatalf("Append: %v", err)
	}

	resp, err := http.Get(srv.URL + "/api/sessions/" + info.ID + "/stream?since=0")
	if err != nil {
		t.Fatalf("GET stream: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200", resp.StatusCode)
	}
	if ct := resp.Header.Get("Content-Type"); ct != "text/event-stream" {
		t.Errorf("Content-Type = %q, want text/event-stream", ct)
	}

	got := readSSE(t, resp.Body)
	var userCommit, turnDone bool
	for _, env := range got {
		if env.Kind == "" {
			t.Errorf("envelope missing kind: %+v", env)
		}
		if env.Kind == api.KindMessage && env.Message != nil && env.Message.Role == "user" {
			userCommit = true
		}
		if env.Kind == api.KindTurnDone {
			turnDone = true
		}
	}
	if !userCommit {
		t.Errorf("stream missing committed user message; got %+v", got)
	}
	if !turnDone {
		t.Errorf("stream missing turn_completed")
	}
}

func TestStreamUnknownSession(t *testing.T) {
	srv := newTestServer(t, plainLLM())
	resp, err := http.Get(srv.URL + "/api/sessions/nope/stream")
	if err != nil {
		t.Fatalf("GET stream: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusNotFound {
		t.Errorf("status = %d, want 404", resp.StatusCode)
	}
}

func TestStreamReplaysFromSince(t *testing.T) {
	srv := newTestServer(t, plainLLM())
	c := client.New(srv.URL)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	info, err := c.Create(ctx)
	if err != nil {
		t.Fatalf("Create: %v", err)
	}
	if err := c.Append(ctx, info.ID, "hello"); err != nil {
		t.Fatalf("Append: %v", err)
	}

	// Wait for the turn to complete so we know the seq.
	h, err := c.History(ctx, info.ID)
	if err != nil {
		t.Fatalf("History: %v", err)
	}

	// Stream from the final seq; we should still see the turn_completed marker.
	resp, err := http.Get(srv.URL + "/api/sessions/" + info.ID + "/stream?since=" + strconv.FormatUint(h.Seq, 10))
	if err != nil {
		t.Fatalf("GET stream: %v", err)
	}
	defer resp.Body.Close()

	got := readSSE(t, resp.Body)
	if len(got) == 0 {
		t.Fatalf("expected at least one event; got none")
	}
	if got[len(got)-1].Kind != api.KindTurnDone {
		t.Errorf("last event = %q, want %q", got[len(got)-1].Kind, api.KindTurnDone)
	}
}

// markdownLLM streams a markdown reply (with a code fence) split across two
// deltas, so a test can compare how the streaming envelope carries the
// committed message vs how the Go /view endpoint renders it.
func markdownLLM() http.HandlerFunc {
	fence := "```go\ncode\n```"
	reply := "**bold**\n\n# Heading\n\n- item\n\n" + fence
	chunk1 := fmt.Sprintf(`data: {"choices":[{"delta":{"content":%q}}]}`+"\n\n", reply[:18])
	chunk2 := fmt.Sprintf(`data: {"choices":[{"delta":{"content":%q,"finish_reason":"stop"}}],"usage":{"prompt_tokens":1,"completion_tokens":9}}`+"\n\n", reply[18:])
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, chunk1)
		fmt.Fprint(w, chunk2)
		fmt.Fprint(w, "data: [DONE]\n")
	}
}

// TestStreamedAssistantHTMLMatchesView locks in the parity guarantee between
// the two render paths: the committed assistant message that arrives on the
// SSE stream must carry server-rendered markdown HTML (message_html) identical
// to what /view renders on reload, so the UI cannot drift.
func TestStreamedAssistantHTMLMatchesView(t *testing.T) {
	srv := newTestServer(t, markdownLLM())
	c := client.New(srv.URL)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	info, err := c.Create(ctx)
	if err != nil {
		t.Fatalf("Create: %v", err)
	}
	if err := c.Append(ctx, info.ID, "make md"); err != nil {
		t.Fatalf("Append: %v", err)
	}

	var got []api.Envelope
	err = c.Subscribe(ctx, info.ID, info.Seq, func(env api.Envelope) { got = append(got, env) },
		func(env api.Envelope) bool { return env.Kind == api.KindTurnDone })
	if err != nil {
		t.Fatalf("Subscribe: %v", err)
	}

	// The committed assistant envelope must carry pre-rendered markdown HTML.
	var streamHTML string
	for _, env := range got {
		if env.Kind == api.KindMessage && env.Message != nil && env.Message.Role == "assistant" {
			streamHTML = env.MessageHTML
			break
		}
	}
	if streamHTML == "" {
		t.Fatalf("assistant message_committed envelope has no message_html; got %+v", got)
	}
	for _, want := range []string{"<strong>bold</strong>", "<h1>Heading</h1>"} {
		if !strings.Contains(streamHTML, want) {
			t.Errorf("streamed assistant HTML missing %q: %s", want, streamHTML)
		}
	}
	if strings.Contains(streamHTML, "**bold**") {
		t.Errorf("streamed assistant HTML left literal markdown: %s", streamHTML)
	}

	// The /view render for the same message must produce the same HTML.
	viewResp, err := http.Get(srv.URL + "/api/sessions/" + info.ID + "/view")
	if err != nil {
		t.Fatalf("GET view: %v", err)
	}
	defer viewResp.Body.Close()
	body, _ := io.ReadAll(viewResp.Body)
	view := string(body)
	for _, frag := range []string{"<strong>bold</strong>", "<h1>Heading</h1>"} {
		if !strings.Contains(view, frag) {
			t.Errorf("/view HTML missing %q:\n%s", frag, view)
		}
	}
	if strings.Contains(view, "**bold**") {
		t.Errorf("/view HTML left literal markdown: %s", view)
	}
}

// TestStreamSSECarriesMessageHTML verifies the raw SSE stream (the transport
// the web UI consumes) serializes the pre-rendered message_html field on the
// committed assistant message, not just the NDJSON client path.
func TestStreamSSECarriesMessageHTML(t *testing.T) {
	srv := newTestServer(t, markdownLLM())
	c := client.New(srv.URL)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	info, err := c.Create(ctx)
	if err != nil {
		t.Fatalf("Create: %v", err)
	}
	if err := c.Append(ctx, info.ID, "make md"); err != nil {
		t.Fatalf("Append: %v", err)
	}

	resp, err := http.Get(srv.URL + "/api/sessions/" + info.ID + "/stream?since=0")
	if err != nil {
		t.Fatalf("GET stream: %v", err)
	}
	defer resp.Body.Close()

	got := readSSE(t, resp.Body)
	var saw bool
	for _, env := range got {
		if env.Kind == api.KindMessage && env.Message != nil && env.Message.Role == "assistant" {
			if !strings.Contains(env.MessageHTML, "<strong>bold</strong>") {
				t.Errorf("SSE assistant envelope message_html = %q, want rendered markdown", env.MessageHTML)
			}
			saw = true
		}
	}
	if !saw {
		t.Fatalf("SSE stream missing committed assistant message; got %d envelopes", len(got))
	}
}

// reasoningLLM streams a reply split between reasoning_content and content, the
// shape a reasoning-capable provider emits, so a test can verify reasoning is
// persisted and rendered on reload exactly as it streamed live.
func reasoningLLM() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w,
			`data: {"choices":[{"delta":{"reasoning_content":"let me think\nstep one"},"finish_reason":null}]}`+"\n\n"+
				`data: {"choices":[{"delta":{"reasoning_content":"\nstep two"},"finish_reason":null}]}`+"\n\n"+
				`data: {"choices":[{"delta":{"content":"The answer"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":2}}`+"\n\n"+
				`data: [DONE]`+"\n")
	}
}

// TestReasoningPersistsAcrossReload locks in the parity guarantee for reasoning
// blocks: they stream live as reasoning_delta events AND are committed onto the
// assistant message, so a hard refresh that re-renders from history (/view)
// shows the same reasoning block instead of losing it.
func TestReasoningPersistsAcrossReload(t *testing.T) {
	srv := newTestServer(t, reasoningLLM())
	id, got, h := runOneTurnID(t, srv.URL, "think it through")

	// The committed history must carry the streamed reasoning, separate from content.
	if len(h.History) < 2 || h.History[1].Role != "assistant" {
		t.Fatalf("history = %+v, want an assistant reply", h.History)
	}
	asst := h.History[1]
	if !strings.Contains(asst.Content, "The answer") {
		t.Errorf("assistant content = %q, want 'The answer'", asst.Content)
	}
	if !strings.Contains(asst.Reasoning, "step one") || !strings.Contains(asst.Reasoning, "step two") {
		t.Errorf("assistant reasoning = %q, want streamed 'step one'/'step two'", asst.Reasoning)
	}
	if strings.Contains(asst.Content, "step one") {
		t.Errorf("reasoning leaked into content: %q", asst.Content)
	}

	// The live bus delivered reasoning_delta events (the streaming path).
	var sawDelta bool
	for _, env := range got {
		if env.Kind == api.KindLLM && env.Event != nil && env.Event.Type == codec.TypeReasoningDelta {
			sawDelta = true
		}
	}
	if !sawDelta {
		t.Errorf("bus missing reasoning_delta events; got %+v", got)
	}

	// The committed message envelope carries reasoning, so an SSE replay that
	// has no live deltas can still render it.
	var commitReasoning string
	for _, env := range got {
		if env.Kind == api.KindMessage && env.Message != nil && env.Message.Role == "assistant" {
			commitReasoning = env.Message.Reasoning
		}
	}
	if !strings.Contains(commitReasoning, "step one") {
		t.Errorf("committed assistant envelope reasoning = %q, want step one", commitReasoning)
	}

	// The /view render (what a hard refresh fetches) shows the reasoning block.
	resp, err := http.Get(srv.URL + "/api/sessions/" + id + "/view")
	if err != nil {
		t.Fatalf("GET view: %v", err)
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	s := string(body)
	if !strings.Contains(s, `class="reasoning"`) {
		t.Errorf("/view does not render a reasoning block:\n%s", s)
	}
	// The full text rides on data-reasoning so JS can collapse it client-side
	// without ever flashing the complete block on screen.
	if !strings.Contains(s, `data-reasoning="`) {
		t.Errorf("/view reasoning missing data-reasoning attribute:\n%s", s)
	}
	if !strings.Contains(s, "step one") || !strings.Contains(s, "step two") {
		t.Errorf("/view reasoning missing streamed text:\n%s", s)
	}
	if !strings.Contains(s, "The answer") {
		t.Errorf("/view missing the answer content:\n%s", s)
	}
}

// TestRestartPersistsSessionState is the persistence proof: run a tool turn on
// one server, shut it down (stopping its schedulers and closing its database),
// boot a fresh server on the same database, and verify the session, its full
// committed history (tool call + result + timing + reasoning), the /view
// render, the sidebar list, and the SSE resume position all survive.
func TestRestartPersistsSessionState(t *testing.T) {
	dbPath := filepath.Join(t.TempDir(), "porter.db")

	// --- Server 1: run a tool turn that ends with a plain reply. ---
	srv1, ts1 := startServerDB(t, dbPath, toolThenReplyLLM())
	c1 := client.New(ts1.URL)
	ctx1, cancel1 := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel1()

	info, err := c1.Create(ctx1)
	if err != nil {
		t.Fatalf("Create: %v", err)
	}
	if !strings.HasPrefix(info.ID, "session_") {
		t.Errorf("session id = %q, want session_<n>", info.ID)
	}
	if err := c1.Append(ctx1, info.ID, "run it"); err != nil {
		t.Fatalf("Append: %v", err)
	}
	if err := c1.Subscribe(ctx1, info.ID, info.Seq, nil, func(env api.Envelope) bool {
		return env.Kind == api.KindTurnDone
	}); err != nil {
		t.Fatalf("Subscribe: %v", err)
	}
	srv1.Close() // stop schedulers + close the database: the process "restarts"

	// --- Server 2: same database, everything must be back. ---
	_, ts2 := startServerDB(t, dbPath, plainLLM())
	c2 := client.New(ts2.URL)
	ctx2, cancel2 := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel2()

	// Sidebar list: the session is present with the persisted preview.
	resp, err := http.Get(ts2.URL + api.SessionsPath)
	if err != nil {
		t.Fatalf("GET list: %v", err)
	}
	defer resp.Body.Close()
	var list api.SessionsResponse
	if err := json.NewDecoder(resp.Body).Decode(&list); err != nil {
		t.Fatalf("decode list: %v", err)
	}
	if len(list.Sessions) != 1 {
		t.Fatalf("list after restart = %+v, want 1 session", list.Sessions)
	}
	if list.Sessions[0].ID != info.ID {
		t.Errorf("session id after restart = %q, want %q", list.Sessions[0].ID, info.ID)
	}
	if list.Sessions[0].Preview != "run it" {
		t.Errorf("preview after restart = %q, want %q", list.Sessions[0].Preview, "run it")
	}

	// History: the full conversation, with tool calls and reasoning intact.
	h, err := c2.History(ctx2, info.ID)
	if err != nil {
		t.Fatalf("History after restart: %v", err)
	}
	if len(h.History) != 4 {
		t.Fatalf("history after restart len = %d, want 4: %+v", len(h.History), h.History)
	}
	if h.History[0].Role != "user" || h.History[0].Content != "run it" {
		t.Errorf("history[0] = %+v, want user 'run it'", h.History[0])
	}
	if h.History[1].Role != "assistant" || len(h.History[1].ToolCalls) != 1 {
		t.Fatalf("history[1] = %+v, want assistant with a tool call", h.History[1])
	}
	if h.History[1].ToolCalls[0].Function.Name != "shell" || !strings.Contains(h.History[1].ToolCalls[0].Function.Arguments, "echo hi") {
		t.Errorf("history[1] tool call = %+v, want shell + echo hi args", h.History[1].ToolCalls[0])
	}
	if h.History[2].Role != "tool" || h.History[2].ToolCallID != "c1" ||
		!strings.Contains(h.History[2].Content, "hi") || !strings.Contains(h.History[2].Content, "exit code: 0") {
		t.Errorf("history[2] = %+v, want the tool result", h.History[2])
	}
	if h.History[3].Role != "assistant" || !strings.Contains(h.History[3].Content, "done") {
		t.Errorf("history[3] = %+v, want the final reply", h.History[3])
	}

	// /view renders the tool call context, exit status, and a server-derived
	// duration — the tool timing (started/finished) survived as real columns.
	viewResp, err := http.Get(ts2.URL + "/api/sessions/" + info.ID + "/view")
	if err != nil {
		t.Fatalf("GET view: %v", err)
	}
	defer viewResp.Body.Close()
	vbody, _ := io.ReadAll(viewResp.Body)
	view := string(vbody)
	for _, want := range []string{
		`tool call: shell`,
		`tool result: shell`,
		`echo hi`,
		`exit_code: 0 · `,
		`title="`,
	} {
		if !strings.Contains(view, want) {
			t.Errorf("view after restart missing %q:\n%s", want, view)
		}
	}

	// SSE resume: subscribing from the persisted seq streams the NEXT turn with
	// no gap — no resync, no replay of the old commits — proving the bus
	// position survived. (Timing/started_at are json:"-" so they don't appear in
	// the history API; /view above proves they survived.)
	if err := c2.Append(ctx2, info.ID, "again"); err != nil {
		t.Fatalf("Append after restart: %v", err)
	}
	var envs []api.Envelope
	if err := c2.Subscribe(ctx2, info.ID, h.Seq, func(env api.Envelope) { envs = append(envs, env) },
		func(env api.Envelope) bool { return env.Kind == api.KindTurnDone }); err != nil {
		t.Fatalf("Subscribe after restart: %v", err)
	}
	if len(envs) == 0 {
		t.Fatal("resumed stream produced no envelopes")
	}
	for _, env := range envs {
		if env.Kind == api.KindResync {
			t.Errorf("resumed stream resynced (bus position lost); got %+v", envs)
		}
	}
	var sawNewUser bool
	for _, env := range envs {
		if env.Kind == api.KindMessage && env.Message != nil && env.Message.Role == "user" && env.Message.Content == "again" {
			sawNewUser = true
		}
	}
	if !sawNewUser {
		t.Errorf("resumed stream missing the new user message; got %+v", envs)
	}
}

// TestRestartPersistsReasoning locks in the restart guarantee for reasoning
// blocks: a reasoning-capable turn committed to one server is re-rendered by
// /view identically after a restart on the same database.
func TestRestartPersistsReasoning(t *testing.T) {
	dbPath := filepath.Join(t.TempDir(), "porter.db")

	srv1, ts1 := startServerDB(t, dbPath, reasoningLLM())
	c1 := client.New(ts1.URL)
	ctx1, cancel1 := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel1()
	info, err := c1.Create(ctx1)
	if err != nil {
		t.Fatalf("Create: %v", err)
	}
	if err := c1.Append(ctx1, info.ID, "think"); err != nil {
		t.Fatalf("Append: %v", err)
	}
	if err := c1.Subscribe(ctx1, info.ID, info.Seq, nil, func(env api.Envelope) bool {
		return env.Kind == api.KindTurnDone
	}); err != nil {
		t.Fatalf("Subscribe: %v", err)
	}
	srv1.Close()

	_, ts2 := startServerDB(t, dbPath, reasoningLLM())

	resp, err := http.Get(ts2.URL + "/api/sessions/" + info.ID + "/view")
	if err != nil {
		t.Fatalf("GET view: %v", err)
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	s := string(body)
	for _, want := range []string{
		`class="reasoning"`,
		`data-reasoning="let me think`,
		"step two",
		"The answer",
	} {
		if !strings.Contains(s, want) {
			t.Errorf("view after restart missing %q:\n%s", want, s)
		}
	}
}

// TestCancelStopsRunningTool is the end-to-end cancellation story: a
// long-running local command appears on /runs, cancelling it via the HTTP
// endpoint kills the process, emits tool_cancelled, removes the run, commits a
// cancelled tool message, and ends the turn without the model being called
// again.
func TestCancelStopsRunningTool(t *testing.T) {
	var mu sync.Mutex
	n := 0
	srv := newTestServer(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		mu.Lock()
		n++
		call := n
		mu.Unlock()
		if call == 1 {
			fmt.Fprint(w,
				`data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1","type":"function","function":{"name":"shell","arguments":"{\"command\":\"sleep 60\"}"}}]},"finish_reason":"tool_calls"}]}`+"\n\n"+
					`data: [DONE]`+"\n")
			return
		}
		// The turn must end on cancellation; a second model round-trip means the
		// partial result was fed back instead of stopping.
		t.Errorf("model was called again after cancellation")
		fmt.Fprint(w,
			`data: {"choices":[{"delta":{"content":"unexpected"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1}}`+"\n\n"+
				`data: [DONE]`+"\n")
	}))
	defer exec.Command("pkill", "-f", "^sleep 60$").Run() // tidy any stray survivor

	c := client.New(srv.URL)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	info, err := c.Create(ctx)
	if err != nil {
		t.Fatalf("Create: %v", err)
	}

	// Subscribe to the bus so we can observe the terminal envelope kinds.
	var busMu sync.Mutex
	var envelopes []api.Envelope
	subCtx, subCancel := context.WithCancel(context.Background())
	defer subCancel()
	subDone := make(chan struct{})
	go func() {
		defer close(subDone)
		_ = c.Subscribe(subCtx, info.ID, info.Seq, func(env api.Envelope) {
			busMu.Lock()
			envelopes = append(envelopes, env)
			busMu.Unlock()
		}, func(env api.Envelope) bool { return env.Kind == api.KindTurnDone })
	}()

	if err := c.Append(ctx, info.ID, "run it"); err != nil {
		t.Fatalf("Append: %v", err)
	}

	// Wait until the run is in-flight (it appears on /runs with the server clock).
	var callID string
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		rr, err := c.Runs(ctx, info.ID)
		if err != nil {
			t.Fatalf("Runs: %v", err)
		}
		if len(rr.Runs) > 0 {
			callID = rr.Runs[0].CallID
			break
		}
		time.Sleep(20 * time.Millisecond)
	}
	if callID == "" {
		t.Fatal("tool never appeared on /runs")
	}

	// Cancel it and wait for the turn to end.
	if err := c.Cancel(ctx, info.ID, callID); err != nil {
		t.Fatalf("Cancel: %v", err)
	}
	select {
	case <-subDone:
	case <-time.After(10 * time.Second):
		t.Fatal("turn never completed after cancel")
	}
	subCancel()

	// The bus carries tool_cancelled (not tool_result) for the cancelled run.
	var sawCancelled bool
	for _, env := range envelopes {
		if env.Kind == api.KindToolCancelled && env.ToolCallID == callID {
			sawCancelled = true
		}
		if env.Kind == api.KindToolResult && env.ToolCallID == callID {
			t.Errorf("normal tool_result emitted for a cancelled run")
		}
	}
	if !sawCancelled {
		t.Errorf("bus missing tool_cancelled; got %+v", envelopes)
	}

	// The run left the in-flight set.
	rr, err := c.Runs(ctx, info.ID)
	if err != nil {
		t.Fatalf("Runs: %v", err)
	}
	if len(rr.Runs) != 0 {
		t.Errorf("runs after cancel = %+v, want empty", rr.Runs)
	}

	// History gained a committed tool message marked cancelled.
	h, err := c.History(ctx, info.ID)
	if err != nil {
		t.Fatalf("History: %v", err)
	}
	var toolMsg *llm.ChatMessage
	for i := range h.History {
		if h.History[i].Role == "tool" {
			toolMsg = &h.History[i]
		}
	}
	if toolMsg == nil {
		t.Fatalf("history missing committed tool message; history = %+v", h.History)
	}
	if !toolMsg.Cancelled {
		t.Errorf("committed tool message not marked cancelled: %+v", toolMsg)
	}

	// No survivor process: the whole group (shell + sleep) was killed.
	time.Sleep(100 * time.Millisecond)
	if out, err := exec.Command("pgrep", "-f", "^sleep 60$").Output(); err == nil {
		t.Errorf("sleep 60 survivor still running after cancel: %q", strings.TrimSpace(string(out)))
	}

	// The reload view renders the cancelled result with the label.
	vres, err := http.Get(srv.URL + "/api/sessions/" + url.PathEscape(info.ID) + "/view")
	if err != nil {
		t.Fatalf("GET /view: %v", err)
	}
	vbody, _ := io.ReadAll(vres.Body)
	vres.Body.Close()
	if !strings.Contains(string(vbody), "cancelled") {
		t.Errorf("/view after cancellation missing 'cancelled' label; body:\n%s", vbody)
	}
}

// blockingStream is a tool-output stream that returns EOF only once its context
// is cancelled, and closes done so a test can observe the client received the
// cancel signal. It models a long-running command on the execution host.
type blockingStream struct {
	ctx  context.Context
	done chan struct{}
	once sync.Once
}

func (s *blockingStream) Read(p []byte) (int, error) {
	<-s.ctx.Done()
	s.once.Do(func() { close(s.done) })
	return 0, io.EOF
}

func (s *blockingStream) Close() error { return nil }

// TestCancelStopsRemoteExecTool verifies cancellation reaches a connected
// execution client: the server tells the client to stop its running command
// (its per-command context is cancelled on the client), then emits
// tool_cancelled, removes the run, commits a cancelled tool message, and ends
// the turn.
func TestCancelStopsRemoteExecTool(t *testing.T) {
	srv := newTestServer(t, toolThenReplyLLM())
	c := client.New(srv.URL)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	info, err := c.Create(ctx)
	if err != nil {
		t.Fatalf("Create: %v", err)
	}

	// Register as the execution provider. The dispatch runs a command that
	// blocks until the client cancels it (the server's Cancel signal), and
	// records when that happens.
	bs := &blockingStream{ctx: nil, done: make(chan struct{})}
	dispatch := func(dctx context.Context, name string, args []byte) (io.ReadCloser, error) {
		bs.ctx = dctx
		return bs, nil
	}
	go func() { _ = c.ServeExec(ctx, info.ID, dispatch) }()
	time.Sleep(100 * time.Millisecond)

	// Subscribe to observe the terminal envelope kinds.
	var busMu sync.Mutex
	var envelopes []api.Envelope
	subCtx, subCancel := context.WithCancel(context.Background())
	defer subCancel()
	subDone := make(chan struct{})
	go func() {
		defer close(subDone)
		_ = c.Subscribe(subCtx, info.ID, info.Seq, func(env api.Envelope) {
			busMu.Lock()
			envelopes = append(envelopes, env)
			busMu.Unlock()
		}, func(env api.Envelope) bool { return env.Kind == api.KindTurnDone })
	}()

	if err := c.Append(ctx, info.ID, "run it"); err != nil {
		t.Fatalf("Append: %v", err)
	}

	var callID string
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		rr, err := c.Runs(ctx, info.ID)
		if err != nil {
			t.Fatalf("Runs: %v", err)
		}
		if len(rr.Runs) > 0 {
			callID = rr.Runs[0].CallID
			break
		}
		time.Sleep(20 * time.Millisecond)
	}
	if callID == "" {
		t.Fatal("tool never appeared on /runs")
	}

	if err := c.Cancel(ctx, info.ID, callID); err != nil {
		t.Fatalf("Cancel: %v", err)
	}

	// The client must receive the cancel signal and stop its command.
	select {
	case <-bs.done:
	case <-time.After(5 * time.Second):
		t.Fatal("execution client never saw the cancel signal")
	}

	// The turn ends and the run is reconciled as cancelled.
	select {
	case <-subDone:
	case <-time.After(10 * time.Second):
		t.Fatal("turn never completed after cancel")
	}
	subCancel()

	var sawCancelled bool
	for _, env := range envelopes {
		if env.Kind == api.KindToolCancelled && env.ToolCallID == callID {
			sawCancelled = true
		}
	}
	if !sawCancelled {
		t.Errorf("bus missing tool_cancelled; got %+v", envelopes)
	}

	rr, err := c.Runs(ctx, info.ID)
	if err != nil {
		t.Fatalf("Runs: %v", err)
	}
	if len(rr.Runs) != 0 {
		t.Errorf("runs after cancel = %+v, want empty", rr.Runs)
	}

	h, err := c.History(ctx, info.ID)
	if err != nil {
		t.Fatalf("History: %v", err)
	}
	var toolMsg *llm.ChatMessage
	for i := range h.History {
		if h.History[i].Role == "tool" {
			toolMsg = &h.History[i]
		}
	}
	if toolMsg == nil {
		t.Fatalf("history missing committed tool message; history = %+v", h.History)
	}
	if !toolMsg.Cancelled {
		t.Errorf("committed tool message not marked cancelled: %+v", toolMsg)
	}
}

// TestCancelSilentToolThenRequeue guards the no-output-cancel regression end to
// end: cancelling a tool that produced no output (sleep) must not wedge the
// session. The committed cancelled tool message must carry content, because the
// next user turn's LLM request includes it and providers (e.g. deepseek) reject
// a role-"tool" message whose content field is missing — which would leave
// every subsequent message unanswerable. The fake LLM below enforces exactly
// that validation, so the test fails if the agent commits an empty cancelled
// result.
func TestCancelSilentToolThenRequeue(t *testing.T) {
	var mu sync.Mutex
	n := 0
	srv := newTestServer(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Enforce the provider contract: every role-"tool" message must carry
		// content. This is what real providers (deepseek via litellm) reject
		// with a 400 when it is missing.
		var req struct {
			Messages []struct {
				Role    string  `json:"role"`
				Content *string `json:"content"`
			} `json:"messages"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Errorf("decode request: %v", err)
		}
		for _, m := range req.Messages {
			if m.Role == "tool" && (m.Content == nil) {
				t.Errorf("provider rejected: tool message missing content field")
				http.Error(w, `{"error":{"message":"missing field content"}}`, http.StatusBadRequest)
				return
			}
		}

		w.Header().Set("Content-Type", "text/event-stream")
		mu.Lock()
		n++
		call := n
		mu.Unlock()
		if call == 1 {
			fmt.Fprint(w,
				`data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1","type":"function","function":{"name":"shell","arguments":"{\"command\":\"sleep 60\"}"}}]},"finish_reason":"tool_calls"}]}`+"\n\n"+
					`data: [DONE]`+"\n")
			return
		}
		fmt.Fprint(w,
			`data: {"choices":[{"delta":{"content":"done"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1}}`+"\n\n"+
				`data: [DONE]`+"\n")
	}))
	defer exec.Command("pkill", "-f", "^sleep 60$").Run() // tidy any stray survivor

	c := client.New(srv.URL)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	info, err := c.Create(ctx)
	if err != nil {
		t.Fatalf("Create: %v", err)
	}

	if err := c.Append(ctx, info.ID, "first"); err != nil {
		t.Fatalf("Append first: %v", err)
	}

	// Wait for the (silent) run to appear.
	var callID string
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		rr, err := c.Runs(ctx, info.ID)
		if err != nil {
			t.Fatalf("Runs: %v", err)
		}
		if len(rr.Runs) > 0 {
			callID = rr.Runs[0].CallID
			break
		}
		time.Sleep(20 * time.Millisecond)
	}
	if callID == "" {
		t.Fatal("tool never appeared on /runs")
	}

	// Cancel it: sleep 60 produces no output, so this is the empty-result case.
	if err := c.Cancel(ctx, info.ID, callID); err != nil {
		t.Fatalf("Cancel: %v", err)
	}
	deadline = time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		rr, err := c.Runs(ctx, info.ID)
		if err != nil {
			t.Fatalf("Runs: %v", err)
		}
		if len(rr.Runs) == 0 {
			break
		}
		time.Sleep(20 * time.Millisecond)
	}

	// The next queued message must still get answered: the cancelled tool
	// message is committed with content, so the LLM request is well-formed.
	if err := c.Append(ctx, info.ID, "second"); err != nil {
		t.Fatalf("Append second: %v", err)
	}
	deadline = time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		h, err := c.History(ctx, info.ID)
		if err != nil {
			t.Fatalf("History: %v", err)
		}
		for _, m := range h.History {
			if m.Role == "assistant" && m.Content == "done" {
				return // second turn completed with a reply
			}
		}
		time.Sleep(20 * time.Millisecond)
	}
	h, _ := c.History(ctx, info.ID)
	t.Fatalf("second turn never completed; history = %+v", h.History)
}

// TestTurnErrorSurfacesOnBus verifies a failed turn (the LLM provider returns
// an error, e.g. a 400) carries the underlying error on the turn_completed
// envelope instead of silently completing: clients render the marker's Error
// field to tell the user the turn failed rather than mistaking it for a turn
// with nothing to say. No assistant reply is committed on a failed turn.
func TestTurnErrorSurfacesOnBus(t *testing.T) {
	srv := newTestServer(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// The provider rejects the request outright (the model never runs).
		http.Error(w, `{"error":{"message":"missing field content"}}`, http.StatusBadRequest)
	}))
	c := client.New(srv.URL)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	info, err := c.Create(ctx)
	if err != nil {
		t.Fatalf("Create: %v", err)
	}
	if err := c.Append(ctx, info.ID, "hello"); err != nil {
		t.Fatalf("Append: %v", err)
	}

	var got []api.Envelope
	if err := c.Subscribe(ctx, info.ID, info.Seq, func(env api.Envelope) { got = append(got, env) },
		func(env api.Envelope) bool { return env.Kind == api.KindTurnDone }); err != nil {
		t.Fatalf("Subscribe: %v", err)
	}

	var turnErr string
	for _, env := range got {
		if env.Kind == api.KindTurnDone {
			turnErr = env.Error
		}
	}
	if turnErr == "" {
		t.Fatalf("turn_completed carried no error; got %+v", got)
	}
	if !strings.Contains(turnErr, "400") || !strings.Contains(turnErr, "missing field content") {
		t.Errorf("turn error = %q, want the provider's 400 body", turnErr)
	}

	// The turn failed before producing a reply; no assistant message committed.
	h, err := c.History(ctx, info.ID)
	if err != nil {
		t.Fatalf("History: %v", err)
	}
	for _, m := range h.History {
		if m.Role == "assistant" {
			t.Errorf("assistant message committed on a failed turn: %+v", m)
		}
	}
}

// TestIndexRendersTurnError verifies the chat page is wired to surface a failed
// turn: the turn_completed SSE handler checks env.error, appends an error
// message block (msg-error), and dedups it by seq so an SSE reconnect replay
// doesn't duplicate the block.
func TestIndexRendersTurnError(t *testing.T) {
	srv := newTestServer(t, plainLLM())
	c := client.New(srv.URL)
	ctx := context.Background()
	info, err := c.Create(ctx)
	if err != nil {
		t.Fatalf("Create: %v", err)
	}

	resp, err := http.Get(srv.URL + "/?session=" + info.ID)
	if err != nil {
		t.Fatalf("GET /: %v", err)
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	s := string(body)
	for _, want := range []string{
		`env.error`,          // the turn_completed handler checks the error
		`renderedTurnErrors`, // dedup on reconnect replay
		`msg-error`,          // the error block class (JS + CSS)
	} {
		if !strings.Contains(s, want) {
			t.Errorf("index missing %q", want)
		}
	}
}

// TestTurnUsageSurvivesRestartViaView is the persistence proof for the query
// model: a tool turn's per-request usage is persisted (one query row per model
// request), so after a restart — when the live bus no longer holds the
// turn_completed marker — /view still renders the "(N in, M out tokens)" line
// in the same place the live stream drew it.
func TestTurnUsageSurvivesRestartViaView(t *testing.T) {
	dbPath := filepath.Join(t.TempDir(), "porter.db")

	// Server 1: run a tool turn. The tool call has no usage; the final reply
	// has usage 1/1, so the derived turn total is 1 in / 1 out.
	srv1, ts1 := startServerDB(t, dbPath, toolThenReplyLLM())
	c1 := client.New(ts1.URL)
	ctx1, cancel1 := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel1()
	info, err := c1.Create(ctx1)
	if err != nil {
		t.Fatalf("Create: %v", err)
	}
	if err := c1.Append(ctx1, info.ID, "run it"); err != nil {
		t.Fatalf("Append: %v", err)
	}
	var done api.Envelope
	if err := c1.Subscribe(ctx1, info.ID, info.Seq, func(env api.Envelope) {
		if env.Kind == api.KindTurnDone {
			done = env
		}
	}, func(env api.Envelope) bool { return env.Kind == api.KindTurnDone }); err != nil {
		t.Fatalf("Subscribe: %v", err)
	}
	// The live marker carries the summed usage and the turn's user-message seq
	// (the identity /view tags its footers with, so the live client can dedup).
	if done.CachedInput != 0 || done.UncachedInput != 1 || done.Output != 1 {
		t.Errorf("turn_completed usage = %d cached/%d miss/%d out, want 0/1/1", done.CachedInput, done.UncachedInput, done.Output)
	}
	if done.TurnSeq == 0 {
		t.Errorf("turn_completed turn_seq = 0, want the user message's seq")
	}
	srv1.Close() // stop schedulers + close the database: the process "restarts"

	// Server 2 on the same database: /view must derive the token line from the
	// persisted queries — the bus no longer has the completion marker.
	_, ts2 := startServerDB(t, dbPath, plainLLM())
	defer ts2.Close()
	resp, err := http.Get(ts2.URL + "/api/sessions/" + info.ID + "/view")
	if err != nil {
		t.Fatalf("GET view: %v", err)
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	view := string(body)
	for _, want := range []string{
		`(1 in, 1 out tokens)`, // the turn's aggregated usage
		`token-line`,           // the metadata class
		`data-turn-seq="`,      // the dedup identity the live client checks
	} {
		if !strings.Contains(view, want) {
			t.Errorf("view after restart missing %q:\n%s", want, view)
		}
	}
}

// TestTurnErrorSurvivesRestartViaView verifies a failed turn's error is
// persisted on its query row, so /view renders the error block after a restart
// exactly where the live stream did — the same reload guarantee as token
// usage, for the query that failed.
func TestTurnErrorSurvivesRestartViaView(t *testing.T) {
	dbPath := filepath.Join(t.TempDir(), "porter.db")

	srv1, ts1 := startServerDB(t, dbPath, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, `{"error":{"message":"missing field content"}}`, http.StatusBadRequest)
	}))
	c1 := client.New(ts1.URL)
	ctx1, cancel1 := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel1()
	info, err := c1.Create(ctx1)
	if err != nil {
		t.Fatalf("Create: %v", err)
	}
	if err := c1.Append(ctx1, info.ID, "hello"); err != nil {
		t.Fatalf("Append: %v", err)
	}
	if err := c1.Subscribe(ctx1, info.ID, info.Seq, nil, func(env api.Envelope) bool {
		return env.Kind == api.KindTurnDone
	}); err != nil {
		t.Fatalf("Subscribe: %v", err)
	}
	srv1.Close() // restart

	_, ts2 := startServerDB(t, dbPath, plainLLM())
	defer ts2.Close()
	resp, err := http.Get(ts2.URL + "/api/sessions/" + info.ID + "/view")
	if err != nil {
		t.Fatalf("GET view: %v", err)
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	view := string(body)
	for _, want := range []string{
		`msg-error`,             // the error block class
		`missing field content`, // the persisted provider error
		`data-turn-seq="`,       // the dedup identity
	} {
		if !strings.Contains(view, want) {
			t.Errorf("view after restart missing %q:\n%s", want, view)
		}
	}
}

// cacheSplitLLM serves a plain reply with a cache split (7 of 10 prompt tokens
// served from cache), so the derived turn renders the explicit
// "(X cached + Y miss in, ...)" line.
func cacheSplitLLM() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w,
			`data: {"choices":[{"delta":{"content":"hi"},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":3,"prompt_tokens_details":{"cached_tokens":7}}}`+"\n\n"+
				`data: [DONE]`+"\n")
	}
}

// TestTurnCacheSplitRendersInView verifies the explicit cached/uncached split
// reaches the web render: the turn_completed marker carries it live, and /view
// renders "(7 cached + 3 miss in, 3 out tokens)" on reload from the persisted
// query rows.
func TestTurnCacheSplitRendersInView(t *testing.T) {
	srv := newTestServer(t, cacheSplitLLM())
	ts := srv
	c := client.New(ts.URL)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	info, err := c.Create(ctx)
	if err != nil {
		t.Fatalf("Create: %v", err)
	}
	if err := c.Append(ctx, info.ID, "run it"); err != nil {
		t.Fatalf("Append: %v", err)
	}
	var done api.Envelope
	if err := c.Subscribe(ctx, info.ID, info.Seq, func(env api.Envelope) {
		if env.Kind == api.KindTurnDone {
			done = env
		}
	}, func(env api.Envelope) bool { return env.Kind == api.KindTurnDone }); err != nil {
		t.Fatalf("Subscribe: %v", err)
	}
	if done.CachedInput != 7 || done.UncachedInput != 3 || done.Output != 3 {
		t.Fatalf("turn_completed usage = %d cached/%d miss/%d out, want 7/3/3", done.CachedInput, done.UncachedInput, done.Output)
	}

	resp, err := http.Get(ts.URL + "/api/sessions/" + info.ID + "/view")
	if err != nil {
		t.Fatalf("GET view: %v", err)
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	// The '+' is HTML-escaped to &#43; by html/template's auto-escaper (it
	// still renders as '+' in the browser).
	for _, want := range []string{
		`(7 cached &#43; 3 miss in, 3 out tokens)`,
		`token-line`,
	} {
		if !strings.Contains(string(body), want) {
			t.Errorf("view missing %q:\n%s", want, body)
		}
	}
}

// runTurnDoneSeq appends a user message to the given session and subscribes
// from `since` until the turn completes, returning the turn_completed envelope
// and the highest bus seq observed. Passing the returned seq as the next turn's
// `since` skips the previous turn's completion marker — a fixed `since` would
// replay it and stop on the wrong turn.
func runTurnDoneSeq(t *testing.T, c *client.Client, id string, since uint64) (api.Envelope, uint64) {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := c.Append(ctx, id, "run it"); err != nil {
		t.Fatalf("Append: %v", err)
	}
	var done api.Envelope
	var last uint64
	if err := c.Subscribe(ctx, id, since, func(env api.Envelope) {
		if env.Seq > last {
			last = env.Seq
		}
	}, func(env api.Envelope) bool {
		if env.Kind == api.KindTurnDone {
			done = env
			return true
		}
		return false
	}); err != nil {
		t.Fatalf("Subscribe: %v", err)
	}
	if done.Kind != api.KindTurnDone {
		t.Fatalf("no turn_completed observed")
	}
	return done, last
}

// TestTurnDoneCarriesSessionTotals verifies the session total accumulates
// across turns and rides on each turn_completed marker, so the web client can
// show a session total below the input box by setting (never adding) from the
// marker's totals. Each cacheSplit turn is 7 cached + 3 miss in, 3 out; two
// turns total 14 cached + 6 miss in, 6 out.
func TestTurnDoneCarriesSessionTotals(t *testing.T) {
	srv := newTestServer(t, cacheSplitLLM())
	c := client.New(srv.URL)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	info, err := c.Create(ctx)
	if err != nil {
		t.Fatalf("Create: %v", err)
	}

	// A fresh session has no usage: the page renders the session-total element
	// hidden, and the live handler is wired to the marker's total fields.
	resp0, err := http.Get(srv.URL + "/?session=" + info.ID)
	if err != nil {
		t.Fatalf("GET / (empty): %v", err)
	}
	body0, _ := io.ReadAll(resp0.Body)
	resp0.Body.Close()
	for _, want := range []string{
		`id="session-total"`,     // the element exists so JS can unhide it
		`hidden`,                 // zero totals -> hidden at render
		`env.total_cached_input`, // the live handler reads the marker totals
		`session-total`,          // the element + JS + CSS class
	} {
		if !strings.Contains(string(body0), want) {
			t.Errorf("empty index missing %q:\n%s", want, body0)
		}
	}

	d1, since := runTurnDoneSeq(t, c, info.ID, info.Seq)
	if d1.TotalCachedInput != 7 || d1.TotalUncachedInput != 3 || d1.TotalOutput != 3 {
		t.Errorf("turn 1 session total = %d cached/%d miss/%d out, want 7/3/3", d1.TotalCachedInput, d1.TotalUncachedInput, d1.TotalOutput)
	}

	d2, _ := runTurnDoneSeq(t, c, info.ID, since)
	if d2.TotalCachedInput != 14 || d2.TotalUncachedInput != 6 || d2.TotalOutput != 6 {
		t.Errorf("turn 2 session total = %d cached/%d miss/%d out, want 14/6/6", d2.TotalCachedInput, d2.TotalUncachedInput, d2.TotalOutput)
	}

	// The page seeds the session-total line from the session's running totals.
	resp, err := http.Get(srv.URL + "/?session=" + info.ID)
	if err != nil {
		t.Fatalf("GET /: %v", err)
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	for _, want := range []string{
		`id="session-total"`,
		`(14 cached &#43; 6 miss in, 6 out tokens)`,
	} {
		if !strings.Contains(string(body), want) {
			t.Errorf("index missing %q:\n%s", want, body)
		}
	}
}

// TestSessionTotalsSurviveRestartViaPage verifies the session total is rebuilt
// from the persisted query rows on a restart — the same guarantee as the
// per-turn token lines — so the page renders the accumulated session total even
// though turn_completed markers are live-only.
func TestSessionTotalsSurviveRestartViaPage(t *testing.T) {
	dbPath := filepath.Join(t.TempDir(), "porter.db")

	// Server 1: run two cacheSplit turns (7 cached + 3 miss in, 3 out each).
	srv1, ts1 := startServerDB(t, dbPath, cacheSplitLLM())
	c1 := client.New(ts1.URL)
	ctx1, cancel1 := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel1()
	info, err := c1.Create(ctx1)
	if err != nil {
		t.Fatalf("Create: %v", err)
	}
	since := info.Seq
	for i := 0; i < 2; i++ {
		_, since = runTurnDoneSeq(t, c1, info.ID, since)
	}
	srv1.Close() // stop schedulers + close the database: the process "restarts"

	// Server 2 on the same database: the page must rebuild the session total
	// from the persisted queries.
	_, ts2 := startServerDB(t, dbPath, plainLLM())
	defer ts2.Close()
	resp, err := http.Get(ts2.URL + "/?session=" + info.ID)
	if err != nil {
		t.Fatalf("GET /: %v", err)
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	for _, want := range []string{
		`id="session-total"`,
		`(14 cached &#43; 6 miss in, 6 out tokens)`,
	} {
		if !strings.Contains(string(body), want) {
			t.Errorf("index after restart missing %q:\n%s", want, body)
		}
	}
}

// TestExecContextAndStatus verifies the exec-context registration endpoint and
// the status endpoint: status is local before a provider connects, remote (with
// the reported context) once it does, and local again after it disconnects.
func TestExecContextAndStatus(t *testing.T) {
	srv := newTestServer(t, plainLLM())
	c := client.New(srv.URL)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	info, err := c.Create(ctx)
	if err != nil {
		t.Fatalf("Create: %v", err)
	}

	execCtx := api.ExecContext{System: "linux/amd64", CWD: "/work", Files: []string{"README.md"}}
	if err := c.PostExecContext(ctx, info.ID, execCtx); err != nil {
		t.Fatalf("PostExecContext: %v", err)
	}

	// No provider connected yet: status is local (context not yet active).
	st, err := c.ExecStatus(ctx, info.ID)
	if err != nil {
		t.Fatalf("ExecStatus: %v", err)
	}
	if st.Connected || st.Kind != "local" {
		t.Errorf("status before connect = %+v, want local/disconnected", st)
	}

	// Connect as the execution provider under its own context so we can cancel
	// the exec subscription without killing the status queries below.
	execCtx2, execCancel := context.WithCancel(ctx)
	go func() { _ = c.ServeExec(execCtx2, info.ID, tools.NewDispatcher().Run) }()
	time.Sleep(150 * time.Millisecond)

	st, err = c.ExecStatus(ctx, info.ID)
	if err != nil {
		t.Fatalf("ExecStatus after connect: %v", err)
	}
	if !st.Connected || st.Kind != "remote" {
		t.Errorf("status after connect = %+v, want remote/connected", st)
	}
	if st.Context == nil || st.Context.CWD != "/work" || st.Context.System != "linux/amd64" {
		t.Errorf("status context = %+v, want the reported context", st.Context)
	}

	// Disconnect (cancel the exec subscription): back to local.
	execCancel()
	time.Sleep(150 * time.Millisecond)
	st, err = c.ExecStatus(ctx, info.ID)
	if err != nil {
		t.Fatalf("ExecStatus after disconnect: %v", err)
	}
	if st.Connected || st.Kind != "local" {
		t.Errorf("status after disconnect = %+v, want local/disconnected", st)
	}
}

// capturingLLM records each request's messages and tools, then replies plainly.
func capturingLLM(captured *[][]json.RawMessage, toolsSeen *[]bool) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		var req struct {
			Messages []json.RawMessage `json:"messages"`
			Tools    []json.RawMessage `json:"tools"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			return
		}
		*captured = append(*captured, req.Messages)
		*toolsSeen = append(*toolsSeen, len(req.Tools) > 0)
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w,
			`data: {"choices":[{"delta":{"content":"hi"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":2}}`+"\n\n"+
				`data: [DONE]`+"\n")
	}
}

// TestExecProviderContextInjectedIntoModel is the end-to-end path: an execution
// provider registers its context (with a skill), and the server injects it as a
// system message and exposes load_skill to the model.
func TestExecProviderContextInjectedIntoModel(t *testing.T) {
	var captured [][]json.RawMessage
	var toolsSeen []bool
	srv := newTestServer(t, capturingLLM(&captured, &toolsSeen))
	c := client.New(srv.URL)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	info, err := c.Create(ctx)
	if err != nil {
		t.Fatalf("Create: %v", err)
	}

	execCtx := api.ExecContext{
		System: "linux/arm64",
		CWD:    "/work",
		Files:  []string{"README.md", "main.go"},
		Skills: []api.Skill{{Name: "my-skill", Description: "does things", Path: "/work/.agents/skills/my-skill/SKILL.md"}},
	}
	if err := c.PostExecContext(ctx, info.ID, execCtx); err != nil {
		t.Fatalf("PostExecContext: %v", err)
	}
	go func() { _ = c.ServeExec(ctx, info.ID, tools.NewDispatcher().Run) }()
	time.Sleep(150 * time.Millisecond)

	if err := c.Append(ctx, info.ID, "hi"); err != nil {
		t.Fatalf("Append: %v", err)
	}
	done := false
	if err := c.Subscribe(ctx, info.ID, info.Seq, nil, func(env api.Envelope) bool {
		if env.Kind == api.KindTurnDone {
			done = true
			return true
		}
		return false
	}); err != nil {
		t.Fatalf("Subscribe: %v", err)
	}
	if !done {
		t.Fatal("no turn_completed observed")
	}

	if len(captured) == 0 {
		t.Fatal("no LLM request captured")
	}
	var first llm.ChatMessage
	if err := json.Unmarshal(captured[0][0], &first); err != nil {
		t.Fatalf("unmarshal first message: %v", err)
	}
	if first.Role != "system" {
		t.Errorf("first message role = %q, want system (injected context)", first.Role)
	}
	for _, want := range []string{"/work", "README.md", "my-skill: does things"} {
		if !strings.Contains(first.Content, want) {
			t.Errorf("injected context missing %q:\n%s", want, first.Content)
		}
	}
	if !toolsSeen[0] {
		t.Error("model was not offered any tools")
	}
}

// TestWebRendersExecStatusAndSystemNotices verifies the web UI surfaces the new
// execution-provider pieces: the SSE stream subscribes to exec_status, and /view
// renders committed role-"system" provider notices as a dim block.
func TestWebRendersExecStatusAndSystemNotices(t *testing.T) {
	srv := newTestServer(t, plainLLM())
	c := client.New(srv.URL)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	info, err := c.Create(ctx)
	if err != nil {
		t.Fatalf("Create: %v", err)
	}
	// Register an exec provider with context, so a connect notice is committed.
	execCtx := api.ExecContext{System: "linux/amd64", CWD: "/work"}
	if err := c.PostExecContext(ctx, info.ID, execCtx); err != nil {
		t.Fatalf("PostExecContext: %v", err)
	}
	go func() { _ = c.ServeExec(ctx, info.ID, tools.NewDispatcher().Run) }()
	time.Sleep(150 * time.Millisecond)
	// Disconnect to keep the server from holding a connection at cleanup.
	cancel()
	time.Sleep(150 * time.Millisecond)

	// Index page: the SSE processor must subscribe to exec_status.
	resp, err := http.Get(srv.URL + "/?session=" + info.ID)
	if err != nil {
		t.Fatalf("GET /: %v", err)
	}
	body, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	if !strings.Contains(string(body), "exec_status") {
		t.Errorf("index missing exec_status in the SSE stream; got:\n%s", string(body)[:400])
	}

	// /view: the committed provider notice renders as a system block.
	resp, err = http.Get(srv.URL + "/api/sessions/" + info.ID + "/view")
	if err != nil {
		t.Fatalf("GET /view: %v", err)
	}
	vbody, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	v := string(vbody)
	if !strings.Contains(v, "msg-system") || !strings.Contains(v, "execution provider connected") {
		t.Errorf("/view missing the system notice; got:\n%s", v[:600])
	}
}

// TestStopStopsRunningTurn is the end-to-end Stop story: a model stream that
// produces partial text then hangs; the user stops the turn via the HTTP
// endpoint; the server commits the partial reply (marked interrupted), ends the
// turn with a stopped marker (not an error), calls the model no further, and
// renders the same on reload.
func TestStopStopsRunningTurn(t *testing.T) {
	var mu sync.Mutex
	n := 0
	srv := newTestServer(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		n++
		call := n
		mu.Unlock()
		w.Header().Set("Content-Type", "text/event-stream")
		if call == 1 {
			// Stream a partial reply, then hold the connection open (no
			// finish_reason, no [DONE]) until the stop cancels it.
			fl, _ := w.(http.Flusher)
			fmt.Fprint(w, `data: {"choices":[{"delta":{"content":"The answer is "},"finish_reason":null}]}`+"\n\n")
			if fl != nil {
				fl.Flush()
			}
			<-r.Context().Done()
			return
		}
		// The turn must end on stop; a second model round-trip means the
		// partial was fed back instead of stopping.
		t.Errorf("model was called again after stop")
		fmt.Fprint(w,
			`data: {"choices":[{"delta":{"content":"unexpected"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1}}`+"\n\n"+
				`data: [DONE]`+"\n")
	}))

	c := client.New(srv.URL)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	info, err := c.Create(ctx)
	if err != nil {
		t.Fatalf("Create: %v", err)
	}

	// Subscribe so we can observe the streamed partial and the terminal marker.
	var busMu sync.Mutex
	var envelopes []api.Envelope
	partialSeen := make(chan struct{}, 1)
	subCtx, subCancel := context.WithCancel(context.Background())
	defer subCancel()
	subDone := make(chan struct{})
	go func() {
		defer close(subDone)
		_ = c.Subscribe(subCtx, info.ID, info.Seq, func(env api.Envelope) {
			busMu.Lock()
			envelopes = append(envelopes, env)
			busMu.Unlock()
			if env.Kind == api.KindLLM && env.Event != nil && env.Event.Type == codec.TypeMessageDelta {
				select {
				case partialSeen <- struct{}{}:
				default:
				}
			}
		}, func(env api.Envelope) bool { return env.Kind == api.KindTurnDone })
	}()

	if err := c.Append(ctx, info.ID, "run it"); err != nil {
		t.Fatalf("Append: %v", err)
	}

	// Wait for the partial text to stream, then stop the turn.
	select {
	case <-partialSeen:
	case <-time.After(5 * time.Second):
		t.Fatal("partial content never streamed")
	}
	if err := c.Stop(ctx, info.ID); err != nil {
		t.Fatalf("Stop: %v", err)
	}
	select {
	case <-subDone:
	case <-time.After(10 * time.Second):
		t.Fatal("turn never completed after stop")
	}
	subCancel()

	// The bus carries turn_completed marked stopped, with no error.
	var sawStopped bool
	for _, env := range envelopes {
		if env.Kind == api.KindTurnDone {
			if !env.Stopped {
				t.Errorf("turn_completed not marked stopped: %+v", env)
			}
			if env.Error != "" {
				t.Errorf("turn_completed carries an error: %+v", env)
			}
			sawStopped = true
		}
	}
	if !sawStopped {
		t.Fatalf("bus missing turn_completed; got %+v", envelopes)
	}

	// History gained the committed partial assistant message, marked
	// interrupted.
	h, err := c.History(ctx, info.ID)
	if err != nil {
		t.Fatalf("History: %v", err)
	}
	var partial *llm.ChatMessage
	for i := range h.History {
		if h.History[i].Role == "assistant" {
			partial = &h.History[i]
		}
	}
	if partial == nil {
		t.Fatalf("history missing committed assistant message; history = %+v", h.History)
	}
	want := "The answer is\n\n... [interrupted]"
	if partial.Content != want {
		t.Errorf("committed partial = %q, want %q", partial.Content, want)
	}
	if len(partial.ToolCalls) != 0 {
		t.Errorf("committed partial carries tool calls: %+v", partial.ToolCalls)
	}

	// The reload view renders the partial message and the stopped footer.
	vres, err := http.Get(srv.URL + "/api/sessions/" + url.PathEscape(info.ID) + "/view")
	if err != nil {
		t.Fatalf("GET /view: %v", err)
	}
	vbody, _ := io.ReadAll(vres.Body)
	vres.Body.Close()
	if !strings.Contains(string(vbody), "... [interrupted]") {
		t.Errorf("/view missing interrupted marker; body:\n%s", vbody)
	}
	if !strings.Contains(string(vbody), `class="turn-stopped"`) {
		t.Errorf("/view missing stopped footer; body:\n%s", vbody)
	}
}

// TestStoppedTurnResumesOnNextMessage verifies a stopped chat is resumable: the
// next user message starts a normal new turn whose model request carries the
// committed partial (with the interrupted marker) in history, so the model
// knows the previous reply was cut off.
func TestStoppedTurnResumesOnNextMessage(t *testing.T) {
	var mu sync.Mutex
	n := 0
	var bodies [][]byte
	srv := newTestServer(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		mu.Lock()
		n++
		call := n
		bodies = append(bodies, body)
		mu.Unlock()
		w.Header().Set("Content-Type", "text/event-stream")
		if call == 1 {
			// First turn: stream a partial reply, then hold until stopped.
			fl, _ := w.(http.Flusher)
			fmt.Fprint(w, `data: {"choices":[{"delta":{"content":"The answer is "},"finish_reason":null}]}`+"\n\n")
			if fl != nil {
				fl.Flush()
			}
			<-r.Context().Done()
			return
		}
		// Second turn (the resume): reply fully.
		fmt.Fprint(w,
			`data: {"choices":[{"delta":{"content":"The answer is 42."},"finish_reason":"stop"}],"usage":{"prompt_tokens":4,"completion_tokens":5}}`+"\n\n"+
				`data: [DONE]`+"\n")
	}))

	c := client.New(srv.URL)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	info, err := c.Create(ctx)
	if err != nil {
		t.Fatalf("Create: %v", err)
	}

	partialSeen := make(chan struct{}, 1)
	turnDone := make(chan struct{}, 8)
	subCtx, subCancel := context.WithCancel(context.Background())
	defer subCancel()
	go func() {
		_ = c.Subscribe(subCtx, info.ID, info.Seq, func(env api.Envelope) {
			if env.Kind == api.KindLLM && env.Event != nil && env.Event.Type == codec.TypeMessageDelta {
				select {
				case partialSeen <- struct{}{}:
				default:
				}
			}
			if env.Kind == api.KindTurnDone {
				select {
				case turnDone <- struct{}{}:
				default:
				}
			}
		}, nil)
	}()

	// Turn 1: partial reply, then stop.
	if err := c.Append(ctx, info.ID, "what is 6*7?"); err != nil {
		t.Fatalf("Append 1: %v", err)
	}
	select {
	case <-partialSeen:
	case <-time.After(5 * time.Second):
		t.Fatal("partial content never streamed")
	}
	if err := c.Stop(ctx, info.ID); err != nil {
		t.Fatalf("Stop: %v", err)
	}
	select {
	case <-turnDone:
	case <-time.After(10 * time.Second):
		t.Fatal("first turn never completed after stop")
	}

	// Turn 2: resume with a plain message; the reply arrives.
	if err := c.Append(ctx, info.ID, "continue"); err != nil {
		t.Fatalf("Append 2: %v", err)
	}
	select {
	case <-turnDone:
	case <-time.After(10 * time.Second):
		t.Fatal("second turn never completed")
	}

	mu.Lock()
	defer mu.Unlock()
	if len(bodies) != 2 {
		t.Fatalf("model requests = %d, want 2 (partial turn + resume)", len(bodies))
	}
	// The resumed request's history carries the committed partial with the
	// interrupted marker, so the model knows the previous reply was cut off.
	var req struct {
		Messages []struct {
			Role    string `json:"role"`
			Content string `json:"content"`
		} `json:"messages"`
	}
	if err := json.Unmarshal(bodies[1], &req); err != nil {
		t.Fatalf("decode resumed request: %v", err)
	}
	found := false
	for _, m := range req.Messages {
		if m.Role == "assistant" && strings.Contains(m.Content, "... [interrupted]") {
			found = true
		}
	}
	if !found {
		t.Errorf("resumed request missing the interrupted partial; messages = %+v", req.Messages)
	}
}

// startServerMCP is startServerDB with an explicit MCP config path, so tests
// can exercise the MCP hub through a real turn.
func startServerMCP(t *testing.T, dbPath, mcpPath string, llmHandler http.HandlerFunc) (*Server, *httptest.Server) {
	t.Helper()
	llmSrv := httptest.NewServer(llmHandler)
	s, err := newServer(config.Config{BaseURL: llmSrv.URL + "/v1", Model: "m", APIKey: "k"}, dbPath, mcpPath)
	if err != nil {
		t.Fatalf("newServer: %v", err)
	}
	ts := httptest.NewServer(s.Handler())
	t.Cleanup(func() {
		s.Close()
		ts.Close()
		llmSrv.Close()
	})
	return s, ts
}

// mockMCPHandler is a minimal streamable-HTTP MCP server for tests: it
// handshakes, lists one echo tool, and echoes tool calls back as text.
func mockMCPHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		var req struct {
			ID     any             `json:"id"`
			Method string          `json:"method"`
			Params json.RawMessage `json:"params"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, "bad request", http.StatusBadRequest)
			return
		}
		resp := map[string]any{"jsonrpc": "2.0", "id": req.ID}
		switch req.Method {
		case "initialize":
			resp["result"] = map[string]any{
				"protocolVersion": "2025-06-18",
				"capabilities":    map[string]any{},
				"serverInfo":      map[string]any{"name": "mock", "version": "1"},
			}
		case "notifications/initialized":
			w.WriteHeader(http.StatusAccepted)
			return
		case "tools/list":
			resp["result"] = map[string]any{
				"tools": []map[string]any{{"name": "echo", "description": "Echo text back", "inputSchema": map[string]any{
					"type": "object", "properties": map[string]any{"text": map[string]any{"type": "string"}}, "required": []string{"text"},
				}}},
			}
		case "tools/call":
			var p struct {
				Name      string         `json:"name"`
				Arguments map[string]any `json:"arguments"`
			}
			_ = json.Unmarshal(req.Params, &p)
			resp["result"] = map[string]any{
				"content": []map[string]any{{"type": "text", "text": "echo: " + fmt.Sprintf("%v", p.Arguments["text"])}},
				"isError": false,
			}
		default:
			http.Error(w, "unknown method", http.StatusBadRequest)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(resp)
	}
}

// mcpThenReplyLLM asks for a FindMCP call, then a CallMCP call, then replies
// plainly.
func mcpThenReplyLLM() http.HandlerFunc {
	var mu sync.Mutex
	n := 0
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		mu.Lock()
		n++
		call := n
		mu.Unlock()
		switch call {
		case 1:
			fmt.Fprint(w,
				`data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1","type":"function","function":{"name":"FindMCP","arguments":"{\"server_name\":\"mock\"}"}}]},"finish_reason":"tool_calls"}]}`+"\n\n"+
					`data: [DONE]`+"\n")
		case 2:
			fmt.Fprint(w,
				`data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c2","type":"function","function":{"name":"CallMCP","arguments":"{\"server_name\":\"mock\",\"tool_name\":\"echo\",\"args\":{\"text\":\"hi\"}}"}}]},"finish_reason":"tool_calls"}]}`+"\n\n"+
					`data: [DONE]`+"\n")
		default:
			fmt.Fprint(w,
				`data: {"choices":[{"delta":{"content":"done"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1}}`+"\n\n"+
					`data: [DONE]`+"\n")
		}
	}
}

// TestMCPTurn drives a real turn that discovers an MCP server with FindMCP,
// calls a tool on it with CallMCP, and verifies both results land in history.
func TestMCPTurn(t *testing.T) {
	mcpSrv := httptest.NewServer(mockMCPHandler())
	defer mcpSrv.Close()

	cfgPath := filepath.Join(t.TempDir(), "porter.mcp.json")
	data, _ := json.Marshal(map[string]any{"servers": []map[string]any{{
		"name": "mock", "description": "Mock server", "url": mcpSrv.URL,
	}}})
	if err := os.WriteFile(cfgPath, data, 0o644); err != nil {
		t.Fatalf("write mcp config: %v", err)
	}

	_, ts := startServerMCP(t, filepath.Join(t.TempDir(), "porter.db"), cfgPath, mcpThenReplyLLM())
	_, hist := runOneTurn(t, ts.URL, "use the mock server")

	var toolResults []string
	for _, m := range hist.History {
		if m.Role == "tool" {
			toolResults = append(toolResults, m.Content)
		}
	}
	if len(toolResults) != 2 {
		t.Fatalf("tool results = %d, want 2: %v", len(toolResults), toolResults)
	}
	if !strings.Contains(toolResults[0], "server mock (1 tools)") || !strings.Contains(toolResults[0], "echo: Echo text back") {
		t.Errorf("FindMCP result = %q", toolResults[0])
	}
	if toolResults[1] != "echo: hi" {
		t.Errorf("CallMCP result = %q, want 'echo: hi'", toolResults[1])
	}
}

// TestTruncationEndToEnd runs a real shell command with output larger than the
// model-view head+tail budget through the whole HTTP stack, and verifies the
// split: the DB and the /view render hold the full output, the bus carries the
// full output plus truncation metadata (rendered as a badge in the UI), and
// the committed message envelope carries the metadata for a reconnecting
// client. The model request itself gets the truncated form — that is covered
// by the agent tests; here we assert the server surfaces what the UI needs.
func TestTruncationEndToEnd(t *testing.T) {
	var mu sync.Mutex
	n := 0
	llmHandler := func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		mu.Lock()
		n++
		call := n
		mu.Unlock()
		if call == 1 {
			io.WriteString(w,
				`data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1","type":"function","function":{"name":"shell","arguments":"{\"command\":\"awk 'BEGIN{for(i=0;i<3000;i++) printf \\\"line %d\\\\n\\\", i}'\"}"}}]},"finish_reason":"tool_calls"}]}`+"\n\n"+
					`data: [DONE]`+"\n")
			return
		}
		io.WriteString(w,
			`data: {"choices":[{"delta":{"content":"done"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1}}`+"\n\n"+
				`data: [DONE]`+"\n")
	}
	ts := newTestServer(t, llmHandler)

	id, got, h := runOneTurnID(t, ts.URL, "run it")

	// The committed history (DB-backed) holds the full output.
	var full string
	for _, m := range h.History {
		if m.Role == "tool" {
			full = m.Content
		}
	}
	if len(full) < 2000 {
		t.Fatalf("expected a big tool output, got %d bytes", len(full))
	}

	// The terminal tool_result envelope carries the full output plus metadata,
	// and every committed role-"tool" envelope carries the metadata too (for a
	// reconnecting client to render the badge).
	var meta *llm.ToolOutputMeta
	committedMetaSeen := false
	for _, env := range got {
		if env.Kind == api.KindToolResult && env.ToolCallID == "c1" {
			if env.Result != full {
				t.Errorf("bus tool_result should carry the full output")
			}
			meta = env.ToolOutput
		}
		if env.Kind == api.KindMessage && env.Message != nil && env.Message.Role == "tool" {
			if env.ToolOutput != nil {
				committedMetaSeen = true
			}
		}
	}
	if meta == nil || !meta.Truncated || meta.TotalBytes != len(full) || meta.ShownBytes != recall.HeadBytes+recall.TailBytes {
		t.Errorf("tool_result metadata = %+v, want truncated total=%d shown=%d", meta, len(full), recall.HeadBytes+recall.TailBytes)
	}
	if !committedMetaSeen {
		t.Errorf("committed message envelope missing tool_output metadata")
	}

	// /view renders the full output plus the truncation badge.
	resp, err := http.Get(ts.URL + "/api/sessions/" + url.PathEscape(id) + "/view")
	if err != nil {
		t.Fatalf("GET /view: %v", err)
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	html := string(body)
	if !strings.Contains(html, "tool-output-badge") || !strings.Contains(html, "model saw") {
		t.Errorf("/view missing truncation badge:\n%s", html)
	}
	// /view renders the whole output (newlines as <br>): the first line,
	// a middle line the truncated model view would omit, and the last line.
	for _, probe := range []string{"line 0", "line 1500", "line 2999"} {
		if !strings.Contains(html, probe) {
			t.Errorf("/view must render the full tool output (missing %q)", probe)
		}
	}
	if strings.Contains(html, "[tool output:") {
		t.Errorf("/view must show the full output, not the truncated model-view form")
	}
}

func TestArchiveUnarchiveSession(t *testing.T) {
	srv := newTestServer(t, plainLLM())
	c := client.New(srv.URL)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	info, err := c.Create(ctx)
	if err != nil {
		t.Fatalf("Create: %v", err)
	}

	// A fresh session is active (no archived_at in the list).
	var list api.SessionsResponse
	if err := fetchJSON(srv.URL+api.SessionsPath, &list); err != nil {
		t.Fatalf("list: %v", err)
	}
	if len(list.Sessions) != 1 || list.Sessions[0].ArchivedAt != 0 {
		t.Fatalf("fresh session = %+v, want one active row", list.Sessions)
	}

	// Archive: 204, and the row carries a nonzero archived_at.
	if err := c.Archive(ctx, info.ID); err != nil {
		t.Fatalf("Archive: %v", err)
	}
	list = api.SessionsResponse{}
	if err := fetchJSON(srv.URL+api.SessionsPath, &list); err != nil {
		t.Fatalf("list after archive: %v", err)
	}
	if len(list.Sessions) != 1 || list.Sessions[0].ArchivedAt == 0 {
		t.Fatalf("archived session = %+v, want archived_at set", list.Sessions)
	}

	// Archiving again is idempotent (no error).
	if err := c.Archive(ctx, info.ID); err != nil {
		t.Fatalf("re-Archive: %v", err)
	}

	// Unarchive: 204, and the row is active again.
	if err := c.Unarchive(ctx, info.ID); err != nil {
		t.Fatalf("Unarchive: %v", err)
	}
	list = api.SessionsResponse{}
	if err := fetchJSON(srv.URL+api.SessionsPath, &list); err != nil {
		t.Fatalf("list after unarchive: %v", err)
	}
	if len(list.Sessions) != 1 || list.Sessions[0].ArchivedAt != 0 {
		t.Fatalf("unarchived session = %+v, want archived_at cleared", list.Sessions)
	}

	// Unknown session: 404.
	resp, err := http.Post(srv.URL+"/api/sessions/session_9999/archive", "", nil)
	if err != nil {
		t.Fatalf("archive unknown: %v", err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusNotFound {
		t.Errorf("archive unknown status = %d, want 404", resp.StatusCode)
	}
}

func TestAppendUnarchivesSession(t *testing.T) {
	srv := newTestServer(t, plainLLM())
	c := client.New(srv.URL)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	info, err := c.Create(ctx)
	if err != nil {
		t.Fatalf("Create: %v", err)
	}
	if err := c.Archive(ctx, info.ID); err != nil {
		t.Fatalf("Archive: %v", err)
	}

	// Chatting with the archived session pulls it out of archive: after the
	// append, the list reports it active again.
	if err := c.Append(ctx, info.ID, "still relevant"); err != nil {
		t.Fatalf("Append: %v", err)
	}
	var list api.SessionsResponse
	if err := fetchJSON(srv.URL+api.SessionsPath, &list); err != nil {
		t.Fatalf("list: %v", err)
	}
	if len(list.Sessions) != 1 || list.Sessions[0].ArchivedAt != 0 {
		t.Fatalf("session after append = %+v, want unarchived", list.Sessions)
	}
}

// fetchJSON GETs url and decodes the JSON response into out, failing the test
// on transport, status, or decode errors. Callers must pass a fresh struct:
// encoding/json does not zero the target, so reusing one across decodes keeps
// omitempty fields (like archived_at on an active session) from a prior
// decode.
func fetchJSON(url string, out any) error {
	resp, err := http.Get(url)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("status %d", resp.StatusCode)
	}
	return json.NewDecoder(resp.Body).Decode(out)
}

func TestIndexArchiveButtonState(t *testing.T) {
	srv := newTestServer(t, plainLLM())
	c := client.New(srv.URL)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	get := func(id string) string {
		t.Helper()
		resp, err := http.Get(srv.URL + "/?session=" + id)
		if err != nil {
			t.Fatalf("GET /: %v", err)
		}
		defer resp.Body.Close()
		body, _ := io.ReadAll(resp.Body)
		return string(body)
	}

	// An active session renders the Archive state (data-archived="false").
	info, err := c.Create(ctx)
	if err != nil {
		t.Fatalf("Create: %v", err)
	}
	s := get(info.ID)
	if !strings.Contains(s, `data-archived="false"`) {
		t.Errorf("active session page missing data-archived=\"false\": %q", s)
	}
	if !strings.Contains(s, "Archive") {
		t.Errorf("active session page should show the Archive action")
	}

	// An archived session renders the Restore state (data-archived="true").
	if err := c.Archive(ctx, info.ID); err != nil {
		t.Fatalf("Archive: %v", err)
	}
	s = get(info.ID)
	if !strings.Contains(s, `data-archived="true"`) {
		t.Errorf("archived session page missing data-archived=\"true\": %q", s)
	}
	if !strings.Contains(s, "Restore") {
		t.Errorf("archived session page should show the Restore action")
	}
}

// TestExecSelectEndpoint verifies POST /exec/select end to end: two named
// clients connect, the second takes over (as before the registry), and
// selecting the first via the endpoint switches the active provider without
// either client disconnecting.
func TestExecSelectEndpoint(t *testing.T) {
	srv := newTestServer(t, plainLLM())
	c := client.New(srv.URL)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	info, err := c.Create(ctx)
	if err != nil {
		t.Fatalf("Create: %v", err)
	}

	// Two execution providers with distinct identities and contexts. Each is
	// connected and verified in turn, so the takeover order is deterministic.
	laptopCtx, laptopCancel := context.WithCancel(ctx)
	defer laptopCancel()
	go func() {
		_ = c.ServeExec(laptopCtx, info.ID, tools.NewDispatcher().Run,
			client.ExecConn{ID: "laptop", Name: "laptop", Kind: "remote"})
	}()
	time.Sleep(150 * time.Millisecond)

	st, err := c.ExecStatus(ctx, info.ID)
	if err != nil {
		t.Fatalf("ExecStatus: %v", err)
	}
	if st.ActiveID != "laptop" || !st.Connected {
		t.Fatalf("status after laptop connect = %+v, want laptop active/connected", st)
	}

	serverCtx, serverCancel := context.WithCancel(ctx)
	defer serverCancel()
	go func() {
		_ = c.ServeExec(serverCtx, info.ID, tools.NewDispatcher().Run,
			client.ExecConn{ID: "server", Name: "server", Kind: "remote"})
	}()
	time.Sleep(150 * time.Millisecond)

	// The second connection (server) takes over, and the status lists both.
	st, err = c.ExecStatus(ctx, info.ID)
	if err != nil {
		t.Fatalf("ExecStatus: %v", err)
	}
	if st.ActiveID != "server" || !st.Connected {
		t.Errorf("status after connects = %+v, want server active/connected", st)
	}
	if len(st.Clients) != 3 {
		t.Errorf("clients = %d, want 3 (local + laptop + server)", len(st.Clients))
	}

	// Select the laptop through the endpoint.
	body, _ := json.Marshal(api.ExecSelectRequest{ID: "laptop"})
	req, err := http.NewRequestWithContext(ctx, http.MethodPost,
		srv.URL+"/api/sessions/"+info.ID+"/exec/select", strings.NewReader(string(body)))
	if err != nil {
		t.Fatalf("build select request: %v", err)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("select request: %v", err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("select status = %d, want 200", resp.StatusCode)
	}

	st, err = c.ExecStatus(ctx, info.ID)
	if err != nil {
		t.Fatalf("ExecStatus after select: %v", err)
	}
	if st.ActiveID != "laptop" || !st.Connected {
		t.Errorf("status after select = %+v, want laptop active/connected", st)
	}
	if len(st.Clients) != 3 {
		t.Errorf("clients after select = %d, want 3 (both remotes still connected)", len(st.Clients))
	}

	// Selecting an unknown provider is rejected.
	body, _ = json.Marshal(api.ExecSelectRequest{ID: "ghost"})
	req, _ = http.NewRequestWithContext(ctx, http.MethodPost,
		srv.URL+"/api/sessions/"+info.ID+"/exec/select", strings.NewReader(string(body)))
	resp, err = http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("ghost select request: %v", err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusNotFound {
		t.Errorf("ghost select status = %d, want 404", resp.StatusCode)
	}
}

// TestWebRendersExecPicker verifies the chat page renders the execution
// picker: the "Commands run on" bar below the chat, seeded by the status
// endpoint's registry (local + connected clients with the active id).
func TestWebRendersExecPicker(t *testing.T) {
	srv := newTestServer(t, plainLLM())
	c := client.New(srv.URL)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	info, err := c.Create(ctx)
	if err != nil {
		t.Fatalf("Create: %v", err)
	}
	// A named remote provider with context, so the picker has something to show
	// besides local.
	execCtx := api.ExecContext{ID: "laptop", Name: "laptop", System: "darwin/arm64", CWD: "/Users/mark"}
	if err := c.PostExecContext(ctx, info.ID, execCtx); err != nil {
		t.Fatalf("PostExecContext: %v", err)
	}
	execCtx2, execCancel := context.WithCancel(ctx)
	go func() {
		_ = c.ServeExec(execCtx2, info.ID, tools.NewDispatcher().Run, client.ExecConn{ID: "laptop", Name: "laptop", Kind: "remote"})
	}()
	time.Sleep(150 * time.Millisecond)
	defer func() { execCancel(); time.Sleep(50 * time.Millisecond) }()

	// The status endpoint carries the registry the picker renders.
	st, err := c.ExecStatus(ctx, info.ID)
	if err != nil {
		t.Fatalf("ExecStatus: %v", err)
	}
	if st.ActiveID != "laptop" || len(st.Clients) != 2 {
		t.Errorf("status = %+v, want laptop active with local + laptop clients", st)
	}

	// The chat page renders the picker bar and button.
	resp, err := http.Get(srv.URL + "/?session=" + info.ID)
	if err != nil {
		t.Fatalf("GET /: %v", err)
	}
	body, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	page := string(body)
	for _, want := range []string{"Commands run on", "exec-picker-btn", "exec-picker", "exec/status", "exec/select"} {
		if !strings.Contains(page, want) {
			t.Errorf("index missing %q in the picker; got:\n%s", want, page[:600])
		}
	}
}
