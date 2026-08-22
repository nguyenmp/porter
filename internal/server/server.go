// Package server owns conversation state and runs the agent loop behind a small
// HTTP API. Clients are stateless: they create a session, append user messages,
// poll history, and subscribe to an event bus. Sessions serialize their own
// history (single writer per session) and pace turn execution through a queue,
// so the server is the serialization point for every conversation.
package server

import (
	"bytes"
	"embed"
	"encoding/json"
	"fmt"
	"html/template"
	"log"
	"net/http"
	"strconv"
	"strings"

	"github.com/go-chi/chi/v5"
	"github.com/yuin/goldmark"
	"github.com/yuin/goldmark/extension"

	"porter/internal/api"
	"porter/internal/config"
	"porter/internal/llm"
	"porter/internal/session"
)

//go:embed web
var webFS embed.FS

// nl2br converts newlines to <br> tags after HTML-escaping. Used in the view
// template to render plaintext (user messages) with line breaks. Returns
// template.HTML so the template engine does not double-escape the output.
func nl2br(s string) template.HTML {
	s = template.HTMLEscapeString(s)
	return template.HTML(strings.ReplaceAll(s, "\n", "<br>"))
}

// renderMarkdown converts markdown text to HTML using goldmark with the GFM
// table extension. goldmark handles its own HTML escaping, so the output is
// safe to return as template.HTML (the template engine will not re-escape it).
func renderMarkdown(s string) template.HTML {
	var buf bytes.Buffer
	if err := goldmark.New(goldmark.WithExtensions(extension.GFM)).Convert([]byte(s), &buf); err != nil {
		// On error, fall back to escaped plaintext with line breaks.
		return nl2br(s)
	}
	return template.HTML(buf.String())
}

// templates are parsed once at startup from the embedded web directory.
var templates = template.Must(template.New("").Funcs(template.FuncMap{
	"nl2br":          nl2br,
	"renderMarkdown": renderMarkdown,
}).ParseFS(webFS, "web/*.tmpl"))

// pageData is the data passed to every page template.
type pageData struct {
	Title   string
	Session string
}

// viewData is passed to the view fragment template. Messages is the session's
// committed history; Turns carries per-turn metadata (token usage, errors) for
// rendering at the bottom of the view.
type viewData struct {
	Messages []llm.ChatMessage
	Turns    []session.TurnMeta
}

// Server owns the LLM client and all sessions.
type Server struct {
	addr   string
	client *llm.Client
	store  *session.Store
}

// New validates cfg and builds a Server with its own LLM client and session
// store. Each session resolves its own execution provider at runtime,
// defaulting to local execution.
func New(cfg config.Config) (*Server, error) {
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	return &Server{
		addr:   cfg.Addr,
		client: llm.NewClient(cfg, nil),
		store:  session.NewStore(),
	}, nil
}

// ListenAndServe serves the API until the process stops.
func (s *Server) ListenAndServe() error {
	return http.ListenAndServe(s.addr, s.Handler())
}

// Handler returns the HTTP routes as an http.Handler.
func (s *Server) Handler() http.Handler {
	r := chi.NewRouter()
	r.Get("/", s.handleIndex)
	r.Post(api.SessionsPath, s.handleCreate)
	r.Post(api.SessionMessagesPath, s.handleAppend)
	r.Get(api.SessionHistoryPath, s.handleHistory)
	r.Get(api.SessionViewPath, s.handleView)
	r.Get(api.SessionEventsPath, s.handleEvents)
	r.Get(api.SessionExecPath, s.handleExec)
	r.Post(api.SessionExecResultPath, s.handleExecResult)
	return r
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

// handleCreate makes a new session and returns its id, history, and resume seq.
func (s *Server) handleCreate(w http.ResponseWriter, r *http.Request) {
	ses := s.store.Create(s.client)
	snap := ses.Snapshot()
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(api.SessionInfo{
		ID:      ses.ID(),
		History: snap.History,
		Seq:     snap.Seq,
	})
}

// handleAppend queues a user message for the session's scheduler. It accepts
// both JSON (api.AppendRequest) and form-encoded ("content" field) bodies so
// HTMX forms can post directly without a JS shim. On success it sets the
// HX-Trigger response header to "refresh" so the chat div polls immediately
// instead of waiting for the next 1s interval.
func (s *Server) handleAppend(w http.ResponseWriter, r *http.Request) {
	ses, ok := s.store.Get(chi.URLParam(r, "id"))
	if !ok {
		http.Error(w, "session not found", http.StatusNotFound)
		return
	}
	content, err := readAppendContent(r)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	if strings.TrimSpace(content) == "" {
		http.Error(w, "empty message", http.StatusBadRequest)
		return
	}
	ses.Enqueue(content)
	w.Header().Set("HX-Trigger", "refresh")
	w.WriteHeader(http.StatusAccepted)
}

// readAppendContent extracts the user's message from either a JSON body
// (api.AppendRequest) or a form-encoded body ("content" field), depending on
// the request's Content-Type.
func readAppendContent(r *http.Request) (string, error) {
	ct := r.Header.Get("Content-Type")
	if strings.HasPrefix(ct, "application/x-www-form-urlencoded") {
		if err := r.ParseForm(); err != nil {
			return "", fmt.Errorf("invalid form: %w", err)
		}
		return r.PostForm.Get("content"), nil
	}
	var req api.AppendRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		return "", fmt.Errorf("invalid request: %w", err)
	}
	return req.Content, nil
}

// handleHistory returns the session's authoritative history and seq.
func (s *Server) handleHistory(w http.ResponseWriter, r *http.Request) {
	ses, ok := s.store.Get(chi.URLParam(r, "id"))
	if !ok {
		http.Error(w, "session not found", http.StatusNotFound)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(ses.Snapshot())
}

// handleEvents subscribes a client to the session's event bus, replaying from
// the caller's `since` then streaming live as NDJSON.
func (s *Server) handleEvents(w http.ResponseWriter, r *http.Request) {
	ses, ok := s.store.Get(chi.URLParam(r, "id"))
	if !ok {
		http.Error(w, "session not found", http.StatusNotFound)
		return
	}
	since, _ := strconv.ParseUint(r.URL.Query().Get("since"), 10, 64)

	w.Header().Set("Content-Type", "application/x-ndjson")
	w.WriteHeader(http.StatusOK)

	enc := json.NewEncoder(flushWriter{w})
	for env := range ses.From(r.Context(), since) {
		_ = enc.Encode(env)
		if env.Kind == api.KindResync {
			return
		}
	}
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

// handleExec registers a client as the session's execution provider and holds
// the connection open, pushing each tool call the agent makes down it as NDJSON.
func (s *Server) handleExec(w http.ResponseWriter, r *http.Request) {
	ses, ok := s.store.Get(chi.URLParam(r, "id"))
	if !ok {
		http.Error(w, "session not found", http.StatusNotFound)
		return
	}
	ch := make(chan api.ExecRequest, 8)
	ses.RegisterExec(ch)
	defer ses.UnregisterExec()

	w.Header().Set("Content-Type", "application/x-ndjson")
	w.WriteHeader(http.StatusOK)

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

// handleExecResult streams a client's tool output back to the in-flight call.
func (s *Server) handleExecResult(w http.ResponseWriter, r *http.Request) {
	ses, ok := s.store.Get(chi.URLParam(r, "id"))
	if !ok {
		http.Error(w, "session not found", http.StatusNotFound)
		return
	}
	if err := ses.ExecResult(chi.URLParam(r, "call_id"), r.Body); err != nil {
		http.Error(w, err.Error(), http.StatusNotFound)
		return
	}
	w.WriteHeader(http.StatusOK)
}

// render executes the named template with the given data and writes the result
// to w as text/html. It is the single entry point for page rendering; handlers
// pass template name and data, nothing else.
func render(w http.ResponseWriter, name string, data any) {
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	if err := templates.ExecuteTemplate(w, name, data); err != nil {
		// The response has likely already started streaming; log the error
		// rather than overwriting a partial write with http.Error.
		log.Printf("render %q: %v", name, err)
	}
}

// handleView renders a session's committed history as an HTML fragment. This is
// the target of the page's HTMX polling: every second the chat div issues
// hx-get to this endpoint and swaps the returned innerHTML.
func (s *Server) handleView(w http.ResponseWriter, r *http.Request) {
	ses, ok := s.store.Get(chi.URLParam(r, "id"))
	if !ok {
		http.Error(w, "session not found", http.StatusNotFound)
		return
	}
	render(w, "view.tmpl", viewData{
		Messages: ses.Snapshot().History,
		Turns:    ses.Turns(),
	})
}

// handleIndex serves the chat page. When no ?session= param is present, it
// creates a new session and redirects to /?session=<id> so the page always has
// something to poll. When the param is present, it renders the page directly.
func (s *Server) handleIndex(w http.ResponseWriter, r *http.Request) {
	sessID := r.URL.Query().Get("session")
	if sessID == "" {
		ses := s.store.Create(s.client)
		http.Redirect(w, r, "/?session="+ses.ID(), http.StatusFound)
		return
	}
	render(w, "layout.tmpl", pageData{
		Title:   "porter",
		Session: sessID,
	})
}
