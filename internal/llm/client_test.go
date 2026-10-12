package llm

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"porter/internal/config"
)

// TestStreamRequestsResponsesEndpoint pins the transport: the request goes to
// the /responses endpoint, and the body carries an "input" item list rather
// than a "messages" array.
func TestStreamRequestsResponsesEndpoint(t *testing.T) {
	var got []byte
	var path string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		got, _ = io.ReadAll(r.Body)
		path = r.URL.Path
		w.Header().Set("Content-Type", "text/event-stream")
		io.WriteString(w, sseEndMarker)
	}))
	defer srv.Close()

	client := NewClient(config.Config{BaseURL: srv.URL + "/v1", Model: "m"}, nil)
	body, err := client.Stream(t.Context(), []ChatMessage{UserMessage("hi")}, nil)
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}
	body.Close()

	if path != "/v1/responses" {
		t.Errorf("request path = %q, want /v1/responses", path)
	}

	var req struct {
		Model    string          `json:"model"`
		Stream   bool            `json:"stream"`
		Messages json.RawMessage `json:"messages"`
		Input    []struct {
			Type    string `json:"type"`
			Role    string `json:"role"`
			Content []struct {
				Type string `json:"type"`
				Text string `json:"text"`
			} `json:"content"`
		} `json:"input"`
	}
	if err := json.Unmarshal(got, &req); err != nil {
		t.Fatalf("unmarshal request body %q: %v", got, err)
	}
	if req.Messages != nil {
		t.Errorf("request still sends a messages array: %s", got)
	}
	if !req.Stream {
		t.Errorf("request must stream: %s", got)
	}
	if len(req.Input) != 1 || req.Input[0].Type != "message" || req.Input[0].Role != "user" {
		t.Fatalf("input = %+v, want one user message item", req.Input)
	}
	if len(req.Input[0].Content) != 1 || req.Input[0].Content[0].Type != "input_text" || req.Input[0].Content[0].Text != "hi" {
		t.Errorf("content = %+v, want one input_text part", req.Input[0].Content)
	}
}

// sseEndMarker is the end-of-stream marker some gateways still send. A fake
// stream that carries no events closes with it.
const sseEndMarker = "data" + ":" + " " + "[" + "DONE" + "]" + "\n"

// TestStreamSendsProvider checks the configured provider preference arrives in
// the request as the `provider` object, because that is what pins the endpoint
// that answers.
func TestStreamSendsProvider(t *testing.T) {
	prefs, err := config.ParseProvider("deepinfra/fp8")
	if err != nil {
		t.Fatalf("ParseProvider: %v", err)
	}
	body := streamBody(t, config.Config{BaseURL: "unused", Model: "m", Provider: prefs})

	var req struct {
		Provider *struct {
			Only           []string `json:"only"`
			AllowFallbacks *bool    `json:"allow_fallbacks"`
		} `json:"provider"`
	}
	if err := json.Unmarshal(body, &req); err != nil {
		t.Fatalf("unmarshal request body %q: %v", body, err)
	}
	if req.Provider == nil {
		t.Fatalf("request body has no provider object; got: %s", body)
	}
	if len(req.Provider.Only) != 1 || req.Provider.Only[0] != "deepinfra/fp8" {
		t.Fatalf("provider.only = %v, want [deepinfra/fp8]", req.Provider.Only)
	}
	if req.Provider.AllowFallbacks == nil || *req.Provider.AllowFallbacks {
		t.Fatalf("provider.allow_fallbacks = %v, want false", req.Provider.AllowFallbacks)
	}
}

// TestStreamOmitsProvider keeps the default request unchanged: with no
// preference set, the request carries no provider object and the gateway
// routes as it did before.
func TestStreamOmitsProvider(t *testing.T) {
	body := streamBody(t, config.Config{BaseURL: "unused", Model: "m"})
	if bytes.Contains(body, []byte(`"provider"`)) {
		t.Fatalf("request body mentions a provider object when none is set: %s", body)
	}
}

// streamBody sends one request through a test server and returns the body the
// client built.
func streamBody(t *testing.T, cfg config.Config) []byte {
	t.Helper()
	var got []byte
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		got, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "text/event-stream")
		io.WriteString(w, "data: [DONE]\n")
	}))
	defer srv.Close()

	cfg.BaseURL = srv.URL + "/v1"
	client := NewClient(cfg, nil)
	body, err := client.Stream(t.Context(), []ChatMessage{UserMessage("hi")}, nil)
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}
	body.Close()
	return got
}
