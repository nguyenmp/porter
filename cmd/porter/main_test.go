package main

import (
	"bytes"
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"porter/internal/config"
)

func TestRunStreamsJSONL(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" {
			t.Errorf("unexpected path %s", r.URL.Path)
		}
		if got := r.Header.Get("Authorization"); got != "Bearer test-key" {
			t.Errorf("authorization = %q", got)
		}
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		fmt.Fprint(w,
			`data: {"choices":[{"delta":{"content":"Hel"},"finish_reason":null}]}`+"\n\n"+
				`data: {"choices":[{"delta":{"content":"lo"},"finish_reason":null}]}`+"\n\n"+
				`data: {"choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":20}}`+"\n\n"+
				`data: [DONE]`+"\n",
		)
	}))
	defer server.Close()

	cfg := config.Config{
		BaseURL: server.URL + "/v1",
		Model:   "test-model",
		APIKey:  "test-key",
	}
	var out bytes.Buffer
	err := run(context.Background(), cfg, "hey", &out)
	if err != nil {
		t.Fatalf("run: %v", err)
	}

	got := out.String()
	for _, want := range []string{
		`"type":"message_delta"`,
		`"delta":"Hel"`,
		`"delta":"lo"`,
		`"type":"message"`,
		`"content":"Hello"`,
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
