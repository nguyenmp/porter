package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"porter/internal/api"
	"porter/internal/client"
	"porter/internal/config"
	"porter/internal/server"
)

func TestRunStreamsJSONL(t *testing.T) {
	// Fake upstream LLM streams a two-part delta "Hello" with usage.
	llmSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w,
			`data: {"choices":[{"delta":{"content":"Hel"},"finish_reason":null}]}`+"\n\n"+
				`data: {"choices":[{"delta":{"content":"lo"},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":20}}`+"\n\n"+
				`data: [DONE]`+"\n")
	}))
	s, err := server.New(config.Config{BaseURL: llmSrv.URL + "/v1", Model: "m", APIKey: "k"})
	if err != nil {
		t.Fatalf("server.New: %v", err)
	}
	srv := httptest.NewServer(s.Handler())
	defer func() { srv.Close(); llmSrv.Close() }()

	cfg := config.ClientConfig{ServerURL: srv.URL}
	var out bytes.Buffer
	err = run(context.Background(), cfg, "hey", &out)
	if err != nil {
		t.Fatalf("run: %v", err)
	}

	got := out.String()
	for _, want := range []string{
		`"type":"message_delta"`,
		`"delta":"Hel"`,
		`"delta":"lo"`,
		`"type":"usage"`,
		`"input_tokens":10`,
		`"output_tokens":20`,
	} {
		if !strings.Contains(got, want) {
			t.Errorf("output missing %q; got:\n%s", want, got)
		}
	}
	if strings.Contains(got, "data:") {
		t.Errorf("raw SSE leaked through; got:\n%s", got)
	}
}

// TestRunResumesSession verifies the one-shot CLI appends to an existing
// session when cfg.Session is set instead of creating a new one: the prompt
// lands in that session's history, and no second session is created.
func TestRunResumesSession(t *testing.T) {
	llmSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w,
			`data: {"choices":[{"delta":{"content":"hi"},"finish_reason":"stop"}],"usage":{"prompt_tokens":4,"completion_tokens":5}}`+"\n\n"+
				`data: [DONE]`+"\n")
	}))
	defer llmSrv.Close()
	s, err := server.New(config.Config{BaseURL: llmSrv.URL + "/v1", Model: "m", APIKey: "k"})
	if err != nil {
		t.Fatalf("server.New: %v", err)
	}
	srv := httptest.NewServer(s.Handler())
	defer func() { srv.Close(); s.Close() }()

	cfg := config.ClientConfig{ServerURL: srv.URL}
	c := client.New(cfg.ServerURL, client.BasicAuth{})

	// The one session we create explicitly is the only one the resumed run may
	// touch: the session count must grow by exactly one across the whole test.
	before := countTestSessions(t, srv.URL)
	info, err := c.Create(context.Background())
	if err != nil {
		t.Fatalf("create session: %v", err)
	}

	// Second run attaches to the session we just created.
	cfg.Session = info.ID
	var out bytes.Buffer
	if err := run(context.Background(), cfg, "hello", &out); err != nil {
		t.Fatalf("run: %v", err)
	}

	// The prompt (and its reply) landed in the existing session's history.
	h, err := c.History(context.Background(), info.ID)
	if err != nil {
		t.Fatalf("history: %v", err)
	}
	var sawUser, sawReply bool
	for _, m := range h.History {
		switch m.Role {
		case "user":
			sawUser = sawUser || m.Content == "hello"
		case "assistant":
			sawReply = true
		}
	}
	if !sawUser {
		t.Errorf("existing session history missing the appended prompt; got:\n%+v", h.History)
	}
	if !sawReply {
		t.Errorf("existing session history missing the assistant reply; got:\n%+v", h.History)
	}

	// And no second session was created server-side: the count grew by exactly
	// the one session we created explicitly.
	if after := countTestSessions(t, srv.URL); after != before+1 {
		t.Errorf("resumed run created a new session: sessions before=%d after=%d (want %d)", before, after, before+1)
	}
}

// countTestSessions returns how many sessions the server currently has.
func countTestSessions(t *testing.T, base string) int {
	t.Helper()
	resp, err := http.Get(base + api.SessionsPath)
	if err != nil {
		t.Fatalf("list sessions: %v", err)
	}
	defer resp.Body.Close()
	var list api.SessionsResponse
	if err := json.NewDecoder(resp.Body).Decode(&list); err != nil {
		t.Fatalf("decode sessions: %v", err)
	}
	return len(list.Sessions)
}

// TestRunReportsTurnError verifies the one-shot CLI bubbles up a failed turn:
// when the LLM provider rejects the request (e.g. a 400), run returns an error
// carrying the provider's message instead of printing "stream complete" and
// exiting zero as if the turn succeeded.
func TestRunReportsTurnError(t *testing.T) {
	llmSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, `{"error":{"message":"missing field content"}}`, http.StatusBadRequest)
	}))
	s, err := server.New(config.Config{BaseURL: llmSrv.URL + "/v1", Model: "m", APIKey: "k"})
	if err != nil {
		t.Fatalf("server.New: %v", err)
	}
	srv := httptest.NewServer(s.Handler())
	defer func() { srv.Close(); llmSrv.Close() }()

	cfg := config.ClientConfig{ServerURL: srv.URL}
	var out bytes.Buffer
	err = run(context.Background(), cfg, "hey", &out)
	if err == nil {
		t.Fatal("run returned nil on a failed turn, want the provider error surfaced")
	}
	if !strings.Contains(err.Error(), "missing field content") {
		t.Errorf("run error = %q, want the provider's message", err)
	}
}
