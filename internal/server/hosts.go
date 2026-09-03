package server

import (
	"encoding/json"
	"net/http"

	"github.com/go-chi/chi/v5"

	"porter/internal/api"
)

// handleHosts returns every registered execution host (persistent agents that
// can provision execution contexts, e.g. the laptop agent), for the web UI's
// "new chat on" picker.
func (s *Server) handleHosts(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(api.HostsResponse{Hosts: s.store.Hosts()})
}

// handleHostExec registers a client as an execution host and holds the
// connection open, pushing each provision request the server needs down it as
// NDJSON. The host's id is its URL path segment (e.g. the hostname); name and
// kind ride as query params, matching a session's exec connection. The
// deferred UnregisterHost removes exactly this registration.
func (s *Server) handleHostExec(w http.ResponseWriter, r *http.Request) {
	ch := make(chan api.HostRequest, 8)
	q := r.URL.Query()
	conn := s.store.RegisterHost(ch, chi.URLParam(r, "host_id"), q.Get("name"), q.Get("kind"))
	defer s.store.UnregisterHost(conn)

	w.Header().Set("Content-Type", "application/x-ndjson")
	w.WriteHeader(http.StatusOK)
	// Flush the 200 now so the host's ServeHost sees the connection accepted
	// immediately, instead of only when the first provision request arrives
	// (an idle host could otherwise never observe its own registration).
	if f, ok := w.(http.Flusher); ok {
		f.Flush()
	}

	enc := json.NewEncoder(flushWriter{w})
	for {
		select {
		case req := <-ch:
			if err := enc.Encode(req); err != nil {
				return
			}
		case <-r.Context().Done():
			return
		}
	}
}

// handleHostContext registers the base environment context of the connected
// execution host (system, default working directory, files, skills). The host
// agent posts it when it connects so the "new chat on" picker can show where
// a host runs and what skills it has.
func (s *Server) handleHostContext(w http.ResponseWriter, r *http.Request) {
	var ctx api.ExecContext
	if err := json.NewDecoder(r.Body).Decode(&ctx); err != nil {
		http.Error(w, "invalid host context: "+err.Error(), http.StatusBadRequest)
		return
	}
	s.store.SetHostContext(ctx)
	w.WriteHeader(http.StatusOK)
}

// handleHostProviderError reports that a host failed to provision a provider
// (e.g. the requested working directory does not exist). The waiting
// session-create request gets the error; the session itself keeps its local
// fallback.
func (s *Server) handleHostProviderError(w http.ResponseWriter, r *http.Request) {
	var req api.ProviderErrorRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "invalid provider error: "+err.Error(), http.StatusBadRequest)
		return
	}
	if err := s.store.HostProviderError(chi.URLParam(r, "host_id"), chi.URLParam(r, "provider_id"), req.Error); err != nil {
		http.Error(w, err.Error(), http.StatusNotFound)
		return
	}
	w.WriteHeader(http.StatusOK)
}
