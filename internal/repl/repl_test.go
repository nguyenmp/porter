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