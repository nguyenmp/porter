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

	"porter/internal/config"
)

func TestRunReadsAndQuits(t *testing.T) {
	var out, jsonl bytes.Buffer
	cfg := config.Config{BaseURL: "http://unused.invalid/v1", Model: "m"}
	err := Run(context.Background(), cfg, strings.NewReader("quit\n"), &out, &jsonl)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if !strings.Contains(out.String(), "> ") {
		t.Errorf("no prompt printed; got:\n%s", out.String())
	}
}

func TestRunEOF(t *testing.T) {
	if err := Run(context.Background(), config.Config{}, strings.NewReader(""), &bytes.Buffer{}, &bytes.Buffer{}); err != nil {
		t.Fatalf("Run on EOF: %v", err)
	}
}

func TestRunStreamsMultiTurn(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Echo the number of messages the client sent, proving history grows.
		var req struct {
			Messages []json.RawMessage `json:"messages"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Errorf("decode request: %v", err)
		}
		n := len(req.Messages)
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprintf(w,
			`data: {"choices":[{"delta":{"reasoning_content":"think"},"finish_reason":null}]}`+"\n\n"+
				`data: {"choices":[{"delta":{"content":"reply "},"finish_reason":null}]}`+"\n\n"+
				`data: {"choices":[{"delta":{"content":"%d"},"finish_reason":"stop"}],"usage":{"prompt_tokens":4,"completion_tokens":5}}`+"\n\n"+
				`data: [DONE]`+"\n", n,
		)
	}))
	defer server.Close()

	cfg := config.Config{BaseURL: server.URL + "/v1", Model: "test-model", APIKey: "k"}
	var out, jsonl bytes.Buffer
	// Two turns: first "hello", then "again". History should grow 1 -> 3.
	err := Run(context.Background(), cfg, strings.NewReader("hello\nagain\nquit\n"), &out, &jsonl)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}

	got := out.String()
	if !strings.Contains(got, "reply 1") {
		t.Errorf("first turn should see 1 message; got:\n%s", got)
	}
	if !strings.Contains(got, "reply 3") {
		t.Errorf("second turn should see 3 messages (history grew); got:\n%s", got)
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

func TestRunToolCallAcrossTurns(t *testing.T) {
	var calls int
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls++
		w.Header().Set("Content-Type", "text/event-stream")
		switch calls {
		case 1: // first turn: ask for a tool call
			fmt.Fprintf(w,
				`data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1","type":"function","function":{"name":"shell","arguments":"{\"command\":\"echo hi\"}"}}]},"finish_reason":"tool_calls"}]}`+"\n\n"+
					`data: [DONE]`+"\n")
		case 2: // first turn: final reply after tool result
			fmt.Fprintf(w, `data: {"choices":[{"delta":{"content":"done1"},"finish_reason":"stop"}]}`+"\n\n"+`data: [DONE]`+"\n")
		default: // second turn
			fmt.Fprintf(w, `data: {"choices":[{"delta":{"content":"done2"},"finish_reason":"stop"}]}`+"\n\n"+`data: [DONE]`+"\n")
		}
	}))
	defer server.Close()

	cfg := config.Config{BaseURL: server.URL + "/v1", Model: "test-model", APIKey: "k"}
	var out, jsonl bytes.Buffer
	err := Run(context.Background(), cfg, strings.NewReader("first\nsecond\nquit\n"), &out, &jsonl)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}

	got := out.String()
	if !strings.Contains(got, "done1") || !strings.Contains(got, "done2") {
		t.Errorf("expected both turns' replies; got:\n%s", got)
	}
	if !strings.Contains(got, "shell") {
		t.Errorf("tool call should surface in stdout view; got:\n%s", got)
	}
	if !strings.Contains(jsonl.String(), `"type":"tool_result"`) {
		t.Errorf("jsonl missing tool_result; got:\n%s", jsonl.String())
	}
}

func TestRunLogFileKeepsTerminalQuiet(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprintf(w, `data: {"choices":[{"delta":{"content":"hi"},"finish_reason":"stop"}]}`+"\n\n"+`data: [DONE]`+"\n")
	}))
	defer server.Close()

	logPath := filepath.Join(t.TempDir(), "porter.log")
	cfg := config.Config{BaseURL: server.URL + "/v1", Model: "test-model", APIKey: "k", LogFile: logPath}

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
