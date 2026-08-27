package repl

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"porter/internal/api"
	"porter/internal/codec"
	"porter/internal/config"
	"porter/internal/llm"
	"porter/internal/server"
)

// newReplServer stands up a real porter server whose fake upstream LLM replies
// "reply <len(messages)>" with usage 4/5.
func newReplServer(t *testing.T) *httptest.Server {
	t.Helper()
	llmSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req struct {
			Messages []json.RawMessage `json:"messages"`
		}
		_ = json.NewDecoder(r.Body).Decode(&req)
		reply := fmt.Sprintf("reply %d", len(req.Messages))
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprintf(w,
			`data: {"choices":[{"delta":{"content":%q},"finish_reason":"stop"}],"usage":{"prompt_tokens":4,"completion_tokens":5}}`+"\n\n"+
				`data: [DONE]`+"\n", reply)
	}))
	s, err := server.New(config.Config{BaseURL: llmSrv.URL + "/v1", Model: "m", APIKey: "k"})
	if err != nil {
		t.Fatalf("server.New: %v", err)
	}
	ts := httptest.NewServer(s.Handler())
	t.Cleanup(func() { ts.Close(); llmSrv.Close() })
	return ts
}

func TestRunReadsAndQuits(t *testing.T) {
	srv := newReplServer(t)
	var out, jsonl bytes.Buffer
	err := Run(context.Background(), config.ClientConfig{ServerURL: srv.URL}, strings.NewReader("quit\n"), &out, &jsonl)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if !strings.Contains(out.String(), "> ") {
		t.Errorf("no prompt printed; got:\n%s", out.String())
	}
}

func TestRunEOF(t *testing.T) {
	srv := newReplServer(t)
	if err := Run(context.Background(), config.ClientConfig{ServerURL: srv.URL}, strings.NewReader(""), &bytes.Buffer{}, &bytes.Buffer{}); err != nil {
		t.Fatalf("Run on EOF: %v", err)
	}
}

func TestRunStreamsMultiTurn(t *testing.T) {
	srv := newReplServer(t)

	cfg := config.ClientConfig{ServerURL: srv.URL}
	var out, jsonl bytes.Buffer
	// Two turns: first "hello", then "again". The REPL is the session's
	// execution provider, so every request sees the injected environment
	// context (1 system message) plus the committed provider-connect notice
	// (1 system message): first turn = 3 messages, second = 5.
	err := Run(context.Background(), cfg, strings.NewReader("hello\nagain\nquit\n"), &out, &jsonl)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}

	got := out.String()
	if !strings.Contains(got, "reply 3") {
		t.Errorf("first turn should see 3 messages (context + connect notice + user); got:\n%s", got)
	}
	if !strings.Contains(got, "reply 5") {
		t.Errorf("second turn should see 5 messages (history grew); got:\n%s", got)
	}
	if !strings.Contains(got, "execution provider connected:") {
		t.Errorf("committed provider-connect notice missing from the view; got:\n%s", got)
	}
	if !strings.Contains(got, "(4 in, 5 out tokens)") {
		t.Errorf("missing token line; got:\n%s", got)
	}
	// Structured JSONL went to the jsonl writer, not stdout.
	if strings.Contains(got, `"type":"message_delta"`) {
		t.Errorf("JSONL leaked into stdout view; got:\n%s", got)
	}
	if !strings.Contains(jsonl.String(), `"type":"message_delta"`) {
		t.Errorf("jsonl writer missing events; got:\n%s", jsonl.String())
	}
}

func TestRunLogFileKeepsTerminalQuiet(t *testing.T) {
	srv := newReplServer(t)

	logPath := filepath.Join(t.TempDir(), "porter.log")
	cfg := config.ClientConfig{ServerURL: srv.URL, LogFile: logPath}

	var out, jsonl bytes.Buffer
	if err := Run(context.Background(), cfg, strings.NewReader("hello\nquit\n"), &out, &jsonl); err != nil {
		t.Fatalf("Run: %v", err)
	}

	if jsonl.Len() != 0 {
		t.Errorf("jsonl writer should be unused when LogFile set; got:\n%s", jsonl.String())
	}
	raw, err := os.ReadFile(logPath)
	if err != nil {
		t.Fatalf("read log file: %v", err)
	}
	if !strings.Contains(string(raw), `"type":"message_delta"`) {
		t.Errorf("log file missing events; got:\n%s", raw)
	}
}

// TestLiveViewRendersCommittedAssistantWhenLiveMissed is a deterministic
// regression test for the flaky multi-turn race: the server streams live LLM
// deltas in real time only (never replayed), so a subscriber that connects
// after a fast reply has streamed misses every delta and only sees the
// committed messages via replay. The view must render the committed assistant
// copy (and record it on the structured stream) in that case, and must not
// double-render when the live stream WAS seen.
func TestLiveViewRendersCommittedAssistantWhenLiveMissed(t *testing.T) {
	var out, jsonl bytes.Buffer
	v := &liveView{out: &out, jsonl: &jsonl}

	// Late subscriber: only the committed messages replay, with no live deltas.
	// The assistant reply must still be shown.
	u1 := llm.UserMessage("hello")
	a1 := llm.AssistantMessage("reply 1", "", nil)
	v.emit(api.Envelope{Kind: api.KindMessage, Message: &u1})
	v.emit(api.Envelope{Kind: api.KindMessage, Message: &a1})
	if !strings.Contains(out.String(), "reply 1") {
		t.Errorf("committed assistant reply lost when live stream was missed; got:\n%s", out.String())
	}
	// The missed reply must also be captured on the structured stream (the
	// log-file path), as the terminal TypeMessage event the live decoder emits.
	if !strings.Contains(jsonl.String(), `"type":"message"`) || !strings.Contains(jsonl.String(), "reply 1") {
		t.Errorf("missed reply not recorded on structured stream; got:\n%s", jsonl.String())
	}

	// Subscriber that caught the live stream: deltas render as they arrive and
	// the committed copy must not double-render.
	out.Reset()
	u2 := llm.UserMessage("again")
	a2 := llm.AssistantMessage("reply 2", "", nil)
	v.emit(api.Envelope{Kind: api.KindMessage, Message: &u2})
	v.emit(api.Envelope{Kind: api.KindLLM, Event: &codec.Event{Type: codec.TypeMessageDelta, Delta: "reply 2"}})
	v.emit(api.Envelope{Kind: api.KindMessage, Message: &a2})
	if got := out.String(); got != "reply 2" {
		t.Errorf("assistant reply after live stream = %q, want %q (no double render)", got, "reply 2")
	}

	// A fully streamed tool round renders its live deltas, and the committed
	// assistant-with-calls copy is not duplicated.
	out.Reset()
	u3 := llm.UserMessage("run it")
	a3 := llm.AssistantMessage("", "", []llm.ToolCall{
		{ID: "call_1", Type: "function", Function: llm.ToolFunction{Name: "shell", Arguments: `{"command":"echo hi"}`}},
	})
	v.emit(api.Envelope{Kind: api.KindMessage, Message: &u3})
	v.emit(api.Envelope{Kind: api.KindLLM, Event: &codec.Event{Type: codec.TypeToolCall, Name: "shell", Arguments: `{"command":"echo hi"}`}})
	v.emit(api.Envelope{Kind: api.KindMessage, Message: &a3})
	if !strings.Contains(out.String(), "shell") {
		t.Errorf("live tool round missing tool call; got:\n%s", out.String())
	}
}

// TestLiveViewRendersTurnError ensures a failed turn surfaces its error on the
// human view and records it on the structured stream, instead of ending the
// turn looking like a successful reply with no tokens.
func TestLiveViewRendersTurnError(t *testing.T) {
	var out, jsonl bytes.Buffer
	v := &liveView{out: &out, jsonl: &jsonl}
	v.emit(api.Envelope{Kind: api.KindTurnDone, TurnID: 1, Error: "server 400: missing field content"})

	if !strings.Contains(out.String(), "error: server 400: missing field content") {
		t.Errorf("turn error not shown on the human view; got:\n%s", out.String())
	}
	if !strings.Contains(jsonl.String(), `"kind":"turn_completed"`) || !strings.Contains(jsonl.String(), "missing field content") {
		t.Errorf("turn error not recorded on the structured stream; got:\n%s", jsonl.String())
	}
}

// TestRunShowsCacheSplit verifies the REPL renders the explicit cached/uncached
// input split when the provider reports cache hits, instead of the plain
// "(N in, ...)" line.
func TestRunShowsCacheSplit(t *testing.T) {
	llmSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w,
			`data: {"choices":[{"delta":{"content":"hi"},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":5,"prompt_tokens_details":{"cached_tokens":6}}}`+"\n\n"+
				`data: [DONE]`+"\n")
	}))
	defer llmSrv.Close()
	s, err := server.New(config.Config{BaseURL: llmSrv.URL + "/v1", Model: "m", APIKey: "k"})
	if err != nil {
		t.Fatalf("server.New: %v", err)
	}
	ts := httptest.NewServer(s.Handler())
	defer func() { ts.Close(); s.Close() }()

	var out, jsonl bytes.Buffer
	err = Run(context.Background(), config.ClientConfig{ServerURL: ts.URL}, strings.NewReader("hi\nquit\n"), &out, &jsonl)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if !strings.Contains(out.String(), "(6 cached + 4 miss in, 5 out tokens)") {
		t.Errorf("missing cached/uncached token line; got:\n%s", out.String())
	}
}
