package agent

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"sync"
	"testing"

	"porter/internal/api"
	"porter/internal/config"
	"porter/internal/llm"
	"porter/internal/tools"
)

// toolServer serves one tool-calling turn then a plain-text reply, capturing
// each request's messages and whether tools were declared.
func toolServer(t *testing.T) (*httptest.Server, func() ([][]json.RawMessage, []bool)) {
	t.Helper()
	var mu sync.Mutex
	var captured [][]json.RawMessage
	var hadTools []bool

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req struct {
			Messages []json.RawMessage `json:"messages"`
			Tools    []json.RawMessage `json:"tools"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Errorf("decode request: %v", err)
		}
		mu.Lock()
		captured = append(captured, req.Messages)
		hadTools = append(hadTools, len(req.Tools) > 0)
		mu.Unlock()

		w.Header().Set("Content-Type", "text/event-stream")
		n := len(hadTools)
		if n == 1 {
			// First turn: ask for a tool call (arguments stream in pieces).
			fmt.Fprintf(w,
				`data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"shell","arguments":"{\"command\":\""}}]},"finish_reason":null}]}`+"\n\n"+
					`data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"echo hi\"}"}}]},"finish_reason":"tool_calls"}]}`+"\n\n"+
					`data: [DONE]`+"\n")
		} else {
			// Second turn: reply plainly, with usage.
			fmt.Fprintf(w,
				`data: {"choices":[{"delta":{"content":"done"},"finish_reason":"stop"}],"usage":{"prompt_tokens":2,"completion_tokens":3}}`+"\n\n"+
					`data: [DONE]`+"\n")
		}
	}))
	return srv, func() ([][]json.RawMessage, []bool) {
		mu.Lock()
		defer mu.Unlock()
		return captured, hadTools
	}
}

func TestRunTurnExecutesToolAndLoops(t *testing.T) {
	srv, snapshot := toolServer(t)
	defer srv.Close()

	cfg := config.Config{BaseURL: srv.URL + "/v1", Model: "test", APIKey: "k"}
	client := llm.NewClient(cfg, nil)
	var text, jsonl bytes.Buffer

	emit := func(env api.Envelope) {
		switch env.Kind {
		case api.KindLLM:
			if env.Event == nil {
				return
			}
			ev := *env.Event
			EncodeJSON(&jsonl)(ev)
			Render(&text, false)(ev)
		case api.KindToolResult:
			data, _ := json.Marshal(env)
			jsonl.Write(append(data, '\n'))
		}
	}

	res, err := RunTurn(context.Background(), client, []llm.ChatMessage{llm.UserMessage("run it")}, tools.NewDispatcher(), emit, nil)
	if err != nil {
		t.Fatalf("RunTurn: %v", err)
	}

	if res.Text != "done" {
		t.Errorf("final text = %q, want %q", res.Text, "done")
	}
	if res.Usage.Input != 2 || res.Usage.Output != 3 {
		t.Errorf("usage = %+v, want 2 in / 3 out", res.Usage)
	}

	captured, hadTools := snapshot()
	if len(captured) != 2 {
		t.Fatalf("expected 2 requests, got %d", len(captured))
	}
	if !hadTools[0] || !hadTools[1] {
		t.Errorf("both requests should declare tools; got %v", hadTools)
	}

	// Second request must feed the tool result back in.
	found := false
	for _, m := range captured[1] {
		var msg struct {
			Role       string `json:"role"`
			ToolCallID string `json:"tool_call_id"`
			Content    string `json:"content"`
		}
		if err := json.Unmarshal(m, &msg); err != nil {
			t.Fatalf("unmarshal message: %v", err)
		}
		if msg.Role == "tool" && msg.ToolCallID == "call_1" && strings.Contains(msg.Content, "exit code: 0") {
			found = true
		}
	}
	if !found {
		t.Errorf("second request missing tool result; got %s", captured[1])
	}

	if !strings.Contains(text.String(), "shell") {
		t.Errorf("text view missing tool indicator; got:\n%s", text.String())
	}
	if !strings.Contains(jsonl.String(), `"type":"tool_call"`) || !strings.Contains(jsonl.String(), `"kind":"tool_result"`) {
		t.Errorf("jsonl missing tool events; got:\n%s", jsonl.String())
	}
}

// TestRunTurnOnMessage commits each finalized message (assistant-with-calls,
// tool result, final reply) as it completes, in order.
func TestRunTurnOnMessage(t *testing.T) {
	srv, _ := toolServer(t)
	defer srv.Close()

	cfg := config.Config{BaseURL: srv.URL + "/v1", Model: "test", APIKey: "k"}
	client := llm.NewClient(cfg, nil)

	var got []llm.ChatMessage
	res, err := RunTurn(context.Background(), client, []llm.ChatMessage{llm.UserMessage("run it")}, tools.NewDispatcher(), nil, func(m llm.ChatMessage) {
		got = append(got, m)
	})
	if err != nil {
		t.Fatalf("RunTurn: %v", err)
	}

	var roles []string
	for _, m := range got {
		roles = append(roles, m.Role)
	}
	want := []string{"assistant", "tool", "assistant"}
	if len(roles) != len(want) {
		t.Fatalf("onMessage calls = %v, want %v", roles, want)
	}
	for i, w := range want {
		if roles[i] != w {
			t.Errorf("message[%d] role = %q, want %q", i, roles[i], w)
		}
	}
	// The first committed message carries the tool call the model requested.
	if len(got[0].ToolCalls) != 1 || got[0].ToolCalls[0].ID != "call_1" {
		t.Errorf("first committed message missing tool call; got %+v", got[0])
	}
	// The tool result is keyed to that call.
	if got[1].Role != "tool" || got[1].ToolCallID != "call_1" {
		t.Errorf("tool result = %+v, want tool_call_id call_1", got[1])
	}
	// The final message is the plain answer.
	if got[2].Content != "done" || len(got[2].ToolCalls) != 0 {
		t.Errorf("final message = %+v, want content done", got[2])
	}
	// onMessage output must match the assembled turn result.
	if !reflect.DeepEqual(got, res.History[1:]) {
		t.Errorf("onMessage order does not match turn history\ncommitted: %+v\nhistory:   %+v", got, res.History[1:])
	}
}