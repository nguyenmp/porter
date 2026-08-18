package main

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"porter/internal/api"
	"porter/internal/codec"
	"porter/internal/config"
)

func TestRunStreamsJSONL(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != api.StreamPath {
			t.Errorf("unexpected path %s", r.URL.Path)
		}
		w.Header().Set("Content-Type", "application/x-ndjson")
		enc := json.NewEncoder(w)
		_ = enc.Encode(codec.Event{Type: codec.TypeMessageDelta, Role: "assistant", Delta: "Hel"})
		_ = enc.Encode(codec.Event{Type: codec.TypeMessageDelta, Role: "assistant", Delta: "lo"})
		_ = enc.Encode(codec.Event{Type: codec.TypeMessage, Role: "assistant", Content: "Hello"})
		_ = enc.Encode(codec.Event{Type: codec.TypeUsage, InputTokens: 10, OutputTokens: 20})
		_ = enc.Encode(api.Completion{Completed: true, Text: "Hello", Input: 10, Output: 20})
	}))
	defer server.Close()

	cfg := config.ClientConfig{ServerURL: server.URL}
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