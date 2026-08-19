// Package server runs the agent loop behind a single HTTP endpoint. It is
// stateless: it holds no conversation history, only the LLM client and the tool
// provider the loop runs. A thin client sends the full history each turn and
// gets back a stream of codec.Event lines ending in an api.Completion trailer
// with the assembled result.
package server

import (
	"encoding/json"
	"log"
	"net/http"

	"porter/internal/agent"
	"porter/internal/api"
	"porter/internal/config"
	"porter/internal/llm"
	"porter/internal/tools"
)

// Server answers /api/stream.
type Server struct {
	addr     string
	client   *llm.Client
	provider tools.Provider
}

// New validates cfg and builds a Server with its own LLM client and local tool
// provider.
func New(cfg config.Config) (*Server, error) {
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	return &Server{
		addr:     cfg.Addr,
		client:   llm.NewClient(cfg, nil),
		provider: tools.NewDispatcher(),
	}, nil
}

// ListenAndServe serves the stream endpoint until the process stops.
func (s *Server) ListenAndServe() error {
	mux := http.NewServeMux()
	mux.HandleFunc(api.StreamPath, s.handleStream)
	return http.ListenAndServe(s.addr, mux)
}

// handler returns the stream endpoint as an http.Handler for testing.
func (s *Server) handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc(api.StreamPath, s.handleStream)
	return mux
}

// flushWriter flushes after every Write so streamed events reach the client the
// moment they are produced, rather than when the handler returns.
type flushWriter struct {
	w http.ResponseWriter
}

func (f flushWriter) Write(p []byte) (int, error) {
	n, err := f.w.Write(p)
	if err == nil {
		if fl, ok := f.w.(http.Flusher); ok {
			fl.Flush()
		}
	}
	return n, err
}

// handleStream runs one turn for the submitted history, streaming every
// codec.Event as NDJSON followed by an api.Completion trailer.
func (s *Server) handleStream(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	var req api.StreamRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "invalid request: "+err.Error(), http.StatusBadRequest)
		return
	}

	w.Header().Set("Content-Type", "application/x-ndjson")
	w.WriteHeader(http.StatusOK)

	res, err := agent.RunTurn(r.Context(), s.client, req.History, s.provider, agent.EncodeJSON(flushWriter{w}), nil)
	if err != nil {
		// Send a completion so the client stops cleanly; it reports the error
		// from the empty result.
		enc := json.NewEncoder(flushWriter{w})
		_ = enc.Encode(api.Completion{Completed: true})
		return
	}
	_ = json.NewEncoder(flushWriter{w}).Encode(api.Completion{
		Completed: true,
		Text:      res.Text,
		Input:     res.Usage.Input,
		Output:    res.Usage.Output,
		History:   res.History,
	})
}

// Serve runs the server, logging the listen address.
func Serve(cfg config.Config) error {
	s, err := New(cfg)
	if err != nil {
		return err
	}
	log.Printf("porter server listening on %s", cfg.Addr)
	return s.ListenAndServe()
}