package llm

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"porter/internal/config"
)

// TestStreamRequestsUsage ensures the streaming request asks the provider for
// token usage via stream_options.include_usage. OpenAI-compatible providers
// (and LiteLLM in front of them) omit `usage` from a streamed response unless
// this flag is set, so without it turn metadata can never report input/output
// token counts.
func TestStreamRequestsUsage(t *testing.T) {
	var got []byte
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		got, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "text/event-stream")
		io.WriteString(w, "data: [DONE]\n")
	}))
	defer srv.Close()

	client := NewClient(config.Config{BaseURL: srv.URL + "/v1", Model: "m"}, nil)
	body, err := client.Stream(t.Context(), []ChatMessage{UserMessage("hi")}, nil)
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}
	body.Close()

	var req struct {
		StreamOptions *struct {
			IncludeUsage bool `json:"include_usage"`
		} `json:"stream_options"`
	}
	if err := json.Unmarshal(got, &req); err != nil {
		t.Fatalf("unmarshal request body %q: %v", got, err)
	}
	if req.StreamOptions == nil || !req.StreamOptions.IncludeUsage {
		t.Fatalf("request body missing stream_options.include_usage=true; got: %s", got)
	}
}
