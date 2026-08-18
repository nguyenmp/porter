package server

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"porter/internal/api"
	"porter/internal/codec"
	"porter/internal/config"
	"porter/internal/llm"
)

func TestHandleStream(t *testing.T) {
	// Fake upstream LLM that replies plainly with usage.
	llmSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if got := r.Header.Get("Authorization"); got != "Bearer k" {
			t.Errorf("authorization = %q", got)
		}
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w,
			`data: {"choices":[{"delta":{"content":"hi"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":2}}`+"\n\n"+
				`data: [DONE]`+"\n")
	}))
	defer llmSrv.Close()

	s, err := New(config.Config{BaseURL: llmSrv.URL + "/v1", Model: "m", APIKey: "k"})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	srv := httptest.NewServer(s.handler())
	defer srv.Close()

	reqBody, _ := json.Marshal(api.StreamRequest{History: []llm.ChatMessage{llm.UserMessage("hello")}})
	resp, err := http.Post(srv.URL+api.StreamPath, "application/json", bytes.NewReader(reqBody))
	if err != nil {
		t.Fatalf("POST: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %s", resp.Status)
	}

	var sawEvent bool
	var comp api.Completion
	sc := bufio.NewScanner(resp.Body)
	for sc.Scan() {
		var probe struct{ Completed bool `json:"completed"` }
		if err := json.Unmarshal(sc.Bytes(), &probe); err != nil {
			t.Fatalf("decode: %v", err)
		}
		if probe.Completed {
			if err := json.Unmarshal(sc.Bytes(), &comp); err != nil {
				t.Fatalf("decode completion: %v", err)
			}
			continue
		}
		var ev codec.Event
		if err := json.Unmarshal(sc.Bytes(), &ev); err != nil {
			t.Fatalf("decode event: %v", err)
		}
		if ev.Type == codec.TypeMessageDelta && ev.Delta == "hi" {
			sawEvent = true
		}
	}
	if err := sc.Err(); err != nil {
		t.Fatalf("scan: %v", err)
	}

	if !sawEvent {
		t.Errorf("did not stream the message delta")
	}
	if !comp.Completed {
		t.Errorf("missing completion trailer")
	}
	if comp.Text != "hi" {
		t.Errorf("completion text = %q, want hi", comp.Text)
	}
	if comp.Input != 1 || comp.Output != 2 {
		t.Errorf("completion usage = %d/%d, want 1/2", comp.Input, comp.Output)
	}
	// The completion history must include the user message plus the reply.
	if len(comp.History) != 2 {
		t.Fatalf("completion history length = %d, want 2", len(comp.History))
	}
	if comp.History[0].Role != "user" || !strings.Contains(comp.History[1].Content, "hi") {
		t.Errorf("unexpected completion history: %+v", comp.History)
	}
}