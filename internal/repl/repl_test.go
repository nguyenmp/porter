package repl

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
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
