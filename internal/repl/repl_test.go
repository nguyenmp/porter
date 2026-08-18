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
	"porter/internal/config"
	"porter/internal/llm"
)

// streamServer serves /api/stream. For each request it streams a reply whose
// text is the number of messages the client sent, then a completion carrying
// that reply and the extended history.
func streamServer(t *testing.T) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != api.StreamPath {
			t.Errorf("unexpected path %s", r.URL.Path)
		}
		var req api.StreamRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Errorf("decode request: %v", err)
		}
		n := len(req.History)
		reply := fmt.Sprintf("reply %d", n)
		history := append(append([]llm.ChatMessage{}, req.History...), llm.AssistantMessage(reply, nil))
		w.Header().Set("Content-Type", "application/x-ndjson")
		enc := json.NewEncoder(w)
		_ = enc.Encode(map[string]string{"type": "message_delta", "role": "assistant", "delta": reply})
		_ = enc.Encode(api.Completion{Completed: true, Text: reply, Input: 4, Output: 5, History: history})
	}))
	return srv
}

func TestRunReadsAndQuits(t *testing.T) {
	var out, jsonl bytes.Buffer
	cfg := config.ClientConfig{ServerURL: "http://unused.invalid"}
	err := Run(context.Background(), cfg, strings.NewReader("quit\n"), &out, &jsonl)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if !strings.Contains(out.String(), "> ") {
		t.Errorf("no prompt printed; got:\n%s", out.String())
	}
}

func TestRunEOF(t *testing.T) {
	if err := Run(context.Background(), config.ClientConfig{}, strings.NewReader(""), &bytes.Buffer{}, &bytes.Buffer{}); err != nil {
		t.Fatalf("Run on EOF: %v", err)
	}
}

func TestRunStreamsMultiTurn(t *testing.T) {
	server := streamServer(t)
	defer server.Close()

	cfg := config.ClientConfig{ServerURL: server.URL}
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
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req api.StreamRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Errorf("decode request: %v", err)
		}
		calls++
		reply := fmt.Sprintf("done%d", calls)
		history := append(append([]llm.ChatMessage{}, req.History...), llm.AssistantMessage(reply, nil))

		w.Header().Set("Content-Type", "application/x-ndjson")
		enc := json.NewEncoder(w)
		// A tool call + result, then the final reply.
		_ = enc.Encode(map[string]string{"type": "tool_call", "tool_call_id": "c1", "name": "shell", "arguments": `{"command":"echo hi"}`})
		_ = enc.Encode(map[string]string{"type": "tool_result", "tool_call_id": "c1", "name": "shell", "result": "exit code: 0"})
		_ = enc.Encode(map[string]string{"type": "message_delta", "role": "assistant", "delta": reply})
		_ = enc.Encode(api.Completion{Completed: true, Text: reply, History: history})
	}))
	defer srv.Close()

	cfg := config.ClientConfig{ServerURL: srv.URL}
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
	server := streamServer(t)
	defer server.Close()

	logPath := filepath.Join(t.TempDir(), "porter.log")
	cfg := config.ClientConfig{ServerURL: server.URL, LogFile: logPath}

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