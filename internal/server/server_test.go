package server

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"porter/internal/api"
	"porter/internal/client"
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

func TestIndexAutoCreatesSession(t *testing.T) {
	srv := newTestServer(t, plainLLM())

	// GET / with no session param should redirect (302) to /?session=<id>.
	// Don't follow redirects so we can inspect the Location.
	client := &http.Client{CheckRedirect: func(req *http.Request, via []*http.Request) error {
		return http.ErrUseLastResponse
	}}
	resp, err := client.Get(srv.URL + "/")
	if err != nil {
		t.Fatalf("GET /: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusFound {
		t.Fatalf("status = %d, want 302", resp.StatusCode)
	}
	loc := resp.Header.Get("Location")
	if !strings.HasPrefix(loc, "/?session=sess-") {
		t.Errorf("Location = %q, want /?session=sess-<id>", loc)
	}

	// Following the redirect should serve the full HTML page.
	resp2, err := http.Get(srv.URL + "/")
	if err != nil {
		t.Fatalf("GET / (follow): %v", err)
	}
	defer resp2.Body.Close()
	if resp2.StatusCode != http.StatusOK {
		t.Errorf("status = %d, want 200 after redirect", resp2.StatusCode)
	}
	ct := resp2.Header.Get("Content-Type")
	if !strings.HasPrefix(ct, "text/html") {
		t.Errorf("Content-Type = %q, want text/html", ct)
	}
	body, _ := io.ReadAll(resp2.Body)
	s := string(body)
	if !strings.Contains(s, "<html") {
		t.Errorf("response does not contain <html")
	}
	if !strings.Contains(s, "<title>porter</title>") {
		t.Errorf("response does not contain <title>porter</title>")
	}
	if !strings.Contains(s, `id="chat"`) {
		t.Errorf("response does not contain #chat div")
	}
}

func TestIndexPassesSessionParam(t *testing.T) {
	srv := newTestServer(t, plainLLM())
	resp, err := http.Get(srv.URL + "/?session=sess-42")
	if err != nil {
		t.Fatalf("GET /: %v", err)
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	s := string(body)
	if !strings.Contains(s, "sess-42") {
		t.Errorf("response does not contain session id 'sess-42'")
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

func TestIndexPollsWhenSessionSet(t *testing.T) {
	srv := newTestServer(t, plainLLM())
	resp, err := http.Get(srv.URL + "/?session=sess-42")
	if err != nil {
		t.Fatalf("GET /: %v", err)
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	s := string(body)
	if !strings.Contains(s, `hx-get="/api/sessions/sess-42/view"`) {
		t.Errorf("index does not contain hx-get for session view")
	}
	if !strings.Contains(s, `hx-trigger="every 1s"`) {
		t.Errorf("index does not contain hx-trigger for polling")
	}
}

func TestIndexRedirectCreatesValidSession(t *testing.T) {
	srv := newTestServer(t, plainLLM())

	// The redirect should point at a session that actually exists in the store.
	// Don't follow redirects so we can inspect the Location.
	client := &http.Client{CheckRedirect: func(req *http.Request, via []*http.Request) error {
		return http.ErrUseLastResponse
	}}
	resp, err := client.Get(srv.URL + "/")
	if err != nil {
		t.Fatalf("GET /: %v", err)
	}
	defer resp.Body.Close()
	loc := resp.Header.Get("Location")
	if loc == "" {
		t.Fatalf("no Location header in redirect")
	}
	// Extract the session id from the Location header.
	sessionID := strings.TrimPrefix(loc, "/?session=")
	if sessionID == loc {
		t.Fatalf("could not extract session id from Location %q", loc)
	}
	// Verify the session is real by fetching its history.
	histResp, err := http.Get(srv.URL + "/api/sessions/" + sessionID)
	if err != nil {
		t.Fatalf("GET history: %v", err)
	}
	defer histResp.Body.Close()
	if histResp.StatusCode != http.StatusOK {
		t.Errorf("history for redirected session: status = %d, want 200", histResp.StatusCode)
	}
}

func TestViewShowsTokenUsage(t *testing.T) {
	srv := newTestServer(t, plainLLM())
	id, _, _ := runOneTurnID(t, srv.URL, "hello")

	resp, err := http.Get(srv.URL + "/api/sessions/" + id + "/view")
	if err != nil {
		t.Fatalf("GET view: %v", err)
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	s := string(body)
	if !strings.Contains(s, "turn-usage") {
		t.Errorf("view does not contain turn-usage section")
	}
	if !strings.Contains(s, "1 in / 2 out") {
		t.Errorf("view does not show token usage '1 in / 2 out'; got: %s", s)
	}
}
