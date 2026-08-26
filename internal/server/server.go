// Package server owns conversation state and runs the agent loop behind a small
// HTTP API. Clients are stateless: they create a session, append user messages,
// poll history, and subscribe to an event bus. Sessions serialize their own
// history (single writer per session) and pace turn execution through a queue,
// so the server is the serialization point for every conversation.
package server

import (
	"embed"
	"encoding/json"
	"fmt"
	"html/template"
	"log"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/go-chi/chi/v5"

	"porter/internal/api"
	"porter/internal/config"
	"porter/internal/db"
	"porter/internal/llm"
	mdr "porter/internal/render"
	"porter/internal/session"
)

//go:embed web
var webFS embed.FS

// nl2br converts newlines to <br> tags after HTML-escaping. Used in the view
// template to render plaintext (user messages) with line breaks. It is a thin
// wrapper over mdr.Plaintext so the template and the live SSE stream share
// one implementation; returns template.HTML so the template engine does not
// double-escape the output.
func nl2br(s string) template.HTML {
	return template.HTML(mdr.Plaintext(s))
}

// renderMarkdown converts markdown text to HTML via the shared render package.
// It returns template.HTML so the template engine does not re-escape the
// already-safe goldmark output. The same mdr.Markdown is used on the live
// SSE stream, so a message looks identical whether it streamed in or was
// rendered from history on load.
func renderMarkdown(s string) template.HTML {
	return template.HTML(mdr.Markdown(s))
}

// templates are parsed once at startup from the embedded web directory.
// fmtDur renders a tool-run duration for the view: tenths of a second below
// 10s, whole seconds at/above 10s (matching the live client's granularity).
func fmtDur(start, end int64) string {
	if end <= start || start == 0 {
		return "0s"
	}
	ms := end - start
	if s := float64(ms) / 1000; s < 10 {
		return fmt.Sprintf("%.1fs", s)
	}
	return fmt.Sprintf("%ds", (ms+500)/1000)
}

// fmtClock renders a server epoch-ms timestamp as local wall-clock time for the
// view's reload summary tooltip.
func fmtClock(ms int64) string {
	if ms <= 0 {
		return ""
	}
	return time.UnixMilli(ms).Format("15:04:05")
}

// toolExitCode extracts the exit-status line the shell tool appends to its
// output ("exit code: 0") and returns it in the summary label form
// ("exit_code: 0"). It returns "" when the result carries no such line (e.g. a
// tool that failed before it started), so the status can be omitted. Mirrored
// by exitCodeLabel in the web client so live and reload render identically.
func toolExitCode(result string) string {
	s := strings.TrimSpace(result)
	if i := strings.LastIndexByte(s, '\n'); i >= 0 {
		s = s[i+1:]
	}
	s = strings.TrimSpace(s)
	code := strings.TrimPrefix(s, "exit code: ")
	if code == s {
		return "" // no "exit code: " prefix on the final line
	}
	return "exit_code: " + code
}

// argsSnippet flattens a tool call's JSON arguments into the short single-line
// form used on the tool-call/result summary, truncating past 60 chars with an
// ellipsis. Mirrors argsSnippet in the web client so live and reload render
// identically.
func argsSnippet(args string) string {
	flat := strings.Join(strings.Fields(args), " ")
	if len(flat) > 60 {
		return flat[:60] + "…"
	}
	return flat
}

var templates = template.Must(template.New("").Funcs(template.FuncMap{
	"nl2br":          nl2br,
	"renderMarkdown": renderMarkdown,
	"dur":            fmtDur,
	"clock":          fmtClock,
	"toolExitCode":   toolExitCode,
	"argsSnippet":    argsSnippet,
}).ParseFS(webFS, "web/*.tmpl"))

// pageData is the data passed to every page template.
type pageData struct {
	Title   string
	Session string
	Seq     uint64
	// Running and Queue seed the connection/turn status indicator at render
	// time, so a page loaded mid-turn (or with turns queued) shows the correct
	// state before the SSE stream catches it up.
	Running bool
	Queue   int
}

// ToolRunInfo carries the display details of one tool call for the view,
// looked up by call_id so a committed role-"tool" result message can render the
// tool name and arguments it shares with its calling assistant message.
type ToolRunInfo struct {
	Name      string
	Arguments string
}

// viewData is passed to the view fragment template. Messages is the session's
// committed history; Tools maps each tool call_id to its name/arguments so tool
// results render with their call context.
type viewData struct {
	Messages []llm.ChatMessage
	Tools    map[string]ToolRunInfo
}

// Server owns the LLM client and all sessions.
type Server struct {
	addr   string
	client *llm.Client
	store  *session.Store
}

// New validates cfg and builds a Server with its own LLM client and session
// store, opening the server-owned session database at porter.db in the working
// directory and loading every persisted session so history, the session list,
// and bus positions survive restarts. Each session resolves its own execution
// provider at runtime, defaulting to local execution.
func New(cfg config.Config) (*Server, error) {
	return newServer(cfg, "porter.db")
}

// newServer is New with an explicit database path, so tests (and future
// configuration) can point the server at a database other than ./porter.db.
func newServer(cfg config.Config, dbPath string) (*Server, error) {
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	d, err := db.Open(dbPath)
	if err != nil {
		return nil, err
	}
	client := llm.NewClient(cfg, nil)
	store := session.NewStore(d)
	if err := store.Load(client); err != nil {
		_ = d.Close()
		return nil, fmt.Errorf("load persisted sessions: %w", err)
	}
	return &Server{addr: cfg.Addr, client: client, store: store}, nil
}

// Close stops the session schedulers and closes the session database. It is
// used on shutdown and by tests that simulate a restart.
func (s *Server) Close() { s.store.Close() }

// ListenAndServe serves the API until the process stops.
func (s *Server) ListenAndServe() error {
	return http.ListenAndServe(s.addr, s.Handler())
}

// Handler returns the HTTP routes as an http.Handler.
func (s *Server) Handler() http.Handler {
	r := chi.NewRouter()
	r.Get("/", s.handleIndex)
	r.Get(api.SessionsPath, s.handleList)
	r.Post(api.SessionsPath, s.handleCreate)
	r.Post(api.SessionMessagesPath, s.handleAppend)
	r.Get(api.SessionHistoryPath, s.handleHistory)
	r.Get(api.SessionViewPath, s.handleView)
	r.Get(api.SessionEventsPath, s.handleEvents)
	r.Get(api.SessionStreamPath, s.handleStream)
	r.Get(api.SessionRunsPath, s.handleRuns)
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

// handleList returns every live session, newest first, for the web sidebar.
// The server is the source of truth for which sessions exist, so a client can
// render the list from here instead of keeping its own registry.
func (s *Server) handleList(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(api.SessionsResponse{Sessions: s.store.List()})
}

// handleCreate makes a new session and returns its id, history, and resume seq.
func (s *Server) handleCreate(w http.ResponseWriter, r *http.Request) {
	ses, err := s.store.Create(s.client)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
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
// HTMX forms can post directly without a JS shim. Live updates reach the page
// over the SSE stream, so nothing else is needed here beyond accepting the
// message.
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

// handleRuns returns the session's in-flight tool runs plus the server's clock,
// so a client that connects or reconnects mid-run can reconstruct running
// blocks with server-accurate elapsed times.
func (s *Server) handleRuns(w http.ResponseWriter, r *http.Request) {
	ses, ok := s.store.Get(chi.URLParam(r, "id"))
	if !ok {
		http.Error(w, "session not found", http.StatusNotFound)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(api.RunsResponse{
		Now:  time.Now().UnixMilli(),
		Runs: ses.Runs(),
	})
}

// sseEventLine formats one envelope as an SSE event line.
func sseEventLine(env api.Envelope) []byte {
	data, _ := json.Marshal(env)
	return []byte(fmt.Sprintf("event: %s\ndata: %s\n\n", env.Kind, data))
}

// handleStream subscribes a client to the session's event bus, replaying from
// the caller's `since` then streaming live as Server-Sent Events. Each
// envelope is sent as an SSE event whose event name is the envelope kind and
// whose data is the JSON-encoded envelope.
func (s *Server) handleStream(w http.ResponseWriter, r *http.Request) {
	ses, ok := s.store.Get(chi.URLParam(r, "id"))
	if !ok {
		http.Error(w, "session not found", http.StatusNotFound)
		return
	}
	since, _ := strconv.ParseUint(r.URL.Query().Get("since"), 10, 64)

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.WriteHeader(http.StatusOK)

	fw := flushWriter{w}
	for env := range ses.From(r.Context(), since) {
		_, _ = fw.Write(sseEventLine(env))
		if env.Kind == api.KindResync {
			return
		}
	}
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
	messages := ses.Snapshot().History
	tools := make(map[string]ToolRunInfo, len(messages))
	for _, m := range messages {
		for _, c := range m.ToolCalls {
			tools[c.ID] = ToolRunInfo{Name: c.Function.Name, Arguments: c.Function.Arguments}
		}
	}
	render(w, "view.tmpl", viewData{
		Messages: messages,
		Tools:    tools,
	})
}

// handleIndex serves the chat page. When no ?session= param is present it
// renders the empty state; creating a chat is now an explicit action (the New
// chat button posts to /api/sessions), not a side effect of visiting the page.
// When the param is present it renders the page directly and passes the current
// seq so the SSE stream can resume without a gap.
func (s *Server) handleIndex(w http.ResponseWriter, r *http.Request) {
	sessID := r.URL.Query().Get("session")
	if sessID == "" {
		render(w, "layout.tmpl", pageData{Title: "porter"})
		return
	}
	ses, ok := s.store.Get(sessID)
	if !ok {
		http.Error(w, "session not found", http.StatusNotFound)
		return
	}
	render(w, "layout.tmpl", pageData{
		Title:   "porter",
		Session: sessID,
		Seq:     ses.Snapshot().Seq,
		Running: ses.Running(),
		Queue:   ses.QueueDepth(),
	})
}
