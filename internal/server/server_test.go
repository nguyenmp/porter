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
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"porter/internal/api"
	"porter/internal/client"
	"porter/internal/codec"
	"porter/internal/config"
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

// newTestServer stands up a real porter server backed by the given fake LLM.
func newTestServer(t *testing.T, llmHandler http.HandlerFunc) *httptest.Server {
	t.Helper()
	llmSrv := httptest.NewServer(llmHandler)
	s, err := New(config.Config{BaseURL: llmSrv.URL + "/v1", Model: "m", APIKey: "k"})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	ts := httptest.NewServer(s.Handler())
	t.Cleanup(func() { ts.Close(); llmSrv.Close() })
	return ts
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
			if env.Input != 1 || env.Output != 2 {
				t.Errorf("turn done usage = %d/%d, want 1/2", env.Input, env.Output)
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
