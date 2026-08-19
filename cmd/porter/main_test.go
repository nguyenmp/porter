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