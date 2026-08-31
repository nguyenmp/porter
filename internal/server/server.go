// Package server owns conversation state and runs the agent loop behind a small
// HTTP API. Clients are stateless: they create a session, append user messages,
// poll history, and subscribe to an event bus. Sessions serialize their own
// history (single writer per session) and pace turn execution through a queue,
// so the server is the serialization point for every conversation.
package server

import (
	"embed"
	"encoding/json"
	"errors"
	"fmt"
	"html/template"
	"io"
	"io/fs"
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
	"porter/internal/mcp"
	mdr "porter/internal/render"
	"porter/internal/session"
)

//go:embed web
var webFS embed.FS

// mustSub returns an fs.FS rooted at sub within webFS, panicking on failure.
// webFS is embedded at build time, so the only failure mode is a typo in the
// subdirectory name, which should fail loudly at startup.
func mustSub(fsys embed.FS, dir string) fs.FS {
	sub, err := fs.Sub(fsys, dir)
	if err != nil {
		panic(err)
	}
	return sub
}

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

// fmtBytes renders a byte count as a short human-readable size for the
// tool-output badge ("1.0 MB", "1.5 KB", "512 B"). Mirrored by fmtBytes in the
// web client so live and reload render identically.
func fmtBytes(n int) string {
	switch {
	case n >= 1<<30:
		return fmt.Sprintf("%.1f GB", float64(n)/(1<<30))
	case n >= 1<<20:
		return fmt.Sprintf("%.1f MB", float64(n)/(1<<20))
	case n >= 1<<10:
		return fmt.Sprintf("%.1f KB", float64(n)/(1<<10))
	default:
		return fmt.Sprintf("%d B", n)
	}
}

// tokenLine renders a turn's token usage for display. When the provider
// reported cache hits, the input split is shown explicitly so cache savings are
// visible: "(X cached + Y miss in, Z out tokens)". Otherwise it is the plain
// "(N in, M out tokens)" line. It is the single Go source of truth for the
// format; the web client mirrors it in JS for the live stream.
func tokenLine(cached, uncached, output int) string {
	if cached > 0 {
		return fmt.Sprintf("(%d cached + %d miss in, %d out tokens)", cached, uncached, output)
	}
	return fmt.Sprintf("(%d in, %d out tokens)", uncached, output)
}

var templates = template.Must(template.New("").Funcs(template.FuncMap{
	"nl2br":          nl2br,
	"renderMarkdown": renderMarkdown,
	"dur":            fmtDur,
	"clock":          fmtClock,
	"toolExitCode":   toolExitCode,
	"argsSnippet":    argsSnippet,
	"tokenLine":      tokenLine,
	"fmtBytes":       fmtBytes,
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
	// TotalCached/TotalUncached/TotalOutput seed the session-total line below
	// the input box at page render. They are the session's accumulated usage
	// across completed turns; live turn_completed markers carry the
	// authoritative updated totals, which the client sets (never adds) from.
	TotalCached   int
	TotalUncached int
	TotalOutput   int
	// Archived reports whether the current session is archived, so the chat
	// page's archive/restore button can render the correct initial state
	// before any JS runs.
	Archived bool
}

// ToolRunInfo carries the display details of one tool call for the view,
// looked up by call_id so a committed role-"tool" result message can render the
// tool name and arguments it shares with its calling assistant message.
type ToolRunInfo struct {
	Name      string
	Arguments string
}

// viewFooter is one turn's outcome for the view: the aggregated token usage
// and, when any of the turn's queries failed, the error (or when the user
// stopped the turn, Stopped). It is rendered after the turn's final message,
// matching where the live client appends it on turn_completed. UserSeq (the
// turn's user-message seq) tags the element so the live client can dedup its
// own render against what /view produced.
type viewFooter struct {
	UserSeq       uint64
	CachedInput   int
	UncachedInput int
	Output        int
	Error         string
	Stopped       bool
}

// viewItem is one element of the rendered history: either a committed message
// or a turn footer placed after the turn's final message. Precomputing the
// interleaving on the server keeps the template a simple flat range and
// guarantees a reload renders the turn outcomes in the same positions the live
// stream does.
type viewItem struct {
	Message *llm.ChatMessage
	Footer  *viewFooter
}

// viewData is passed to the view fragment template. Items is the session's
// committed history interleaved with turn footers; Tools maps each tool
// call_id to its name/arguments so tool results render with their call context.
type viewData struct {
	Items []viewItem
	Tools map[string]ToolRunInfo
}

// Server owns the LLM client and all sessions.
type Server struct {
	addr   string
	client *llm.Client
	store  *session.Store
}

// New validates cfg and builds a Server with its own LLM client and session
// store, opening the server-owned session database at porter.db in the working
// directory, loading every persisted session so history, the session list, and
// bus positions survive restarts, and loading the MCP hub from porter.mcp.json
// so sessions can serve MCP tools. Each session resolves its own execution
// provider at runtime, defaulting to local execution.
func New(cfg config.Config) (*Server, error) {
	return newServer(cfg, "porter.db", "porter.mcp.json")
}

// newServer is New with explicit database and MCP config paths, so tests (and
// future configuration) can point the server at a database other than
// ./porter.db and skip or redirect MCP loading.
func newServer(cfg config.Config, dbPath, mcpPath string) (*Server, error) {
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	d, err := db.Open(dbPath)
	if err != nil {
		return nil, err
	}
	// Load the MCP hub before serving: each configured server's tool list is
	// fetched once at boot and cached in memory (see mcp.Load). A missing
	// config file is fine — no MCP — while a malformed one fails startup like
	// a bad env. Credentials live only here, on the server.
	hub, err := mcp.Load(mcpPath, nil)
	if err != nil {
		_ = d.Close()
		return nil, err
	}
	client := llm.NewClient(cfg, nil)
	store := session.NewStore(d, hub)
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
	// Static assets for the web UI (vendored client libraries). webFS embeds
	// the whole web/ directory, so markdown-it ships inside the binary; /web/*
	// serves it with the directory prefix stripped.
	r.Handle("/web/*", http.StripPrefix("/web/", http.FileServer(http.FS(mustSub(webFS, "web")))))
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
	r.Post(api.SessionCancelPath, s.handleCancel)
	r.Post(api.SessionStopPath, s.handleStop)
	r.Post(api.SessionArchivePath, s.handleArchive)
	r.Post(api.SessionUnarchivePath, s.handleUnarchive)
	r.Post(api.SessionExecContextPath, s.handleExecContext)
	r.Get(api.SessionExecStatusPath, s.handleExecStatus)
	r.Post(api.SessionExecSelectPath, s.handleExecSelect)
	r.Get(api.HostsPath, s.handleHosts)
	r.Get(api.HostExecPath, s.handleHostExec)
	r.Post(api.HostContextPath, s.handleHostContext)
	r.Post(api.HostProviderErrorPath, s.handleHostProviderError)
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

// handleCreate makes a new session and returns its id, history, and resume
// seq. When the request names an execution host ({"host": "macbook"}), the
// server asks that host to provision an execution context for the session and
// waits (bounded) for its provider to register, so the first message in a
// chat created "on" a host runs there. A provisioning failure is not fatal:
// the session is returned with a Warning and keeps its local fallback until
// the host's provider (if ever) connects.
func (s *Server) handleCreate(w http.ResponseWriter, r *http.Request) {
	req, err := readCreateRequest(r)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	ses, err := s.store.Create(s.client)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	warning := ""
	if req.Host != "" && req.Host != "local" {
		if err := s.store.Provision(r.Context(), ses.ID(), req.Host, api.HostRequest{
			CWD:    req.CWD,
			Repo:   req.Repo,
			Branch: req.Branch,
		}); err != nil {
			warning = fmt.Sprintf("could not create execution context on %s: %v", req.Host, err)
		}
	}
	snap := ses.Snapshot()
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(api.SessionInfo{
		ID:      ses.ID(),
		History: snap.History,
		Seq:     snap.Seq,
		Warning: warning,
	})
}

// readCreateRequest extracts the optional create body from either a JSON body
// (api.CreateRequest) or a form-encoded body ("host" field), matching
// handleAppend's dual-format handling so the HTMX new-chat form can post
// directly without a JS shim.
func readCreateRequest(r *http.Request) (api.CreateRequest, error) {
	var req api.CreateRequest
	ct := r.Header.Get("Content-Type")
	if strings.HasPrefix(ct, "application/x-www-form-urlencoded") {
		if err := r.ParseForm(); err != nil {
			return req, fmt.Errorf("invalid form: %w", err)
		}
		req.Host = r.PostForm.Get("host")
		req.CWD = r.PostForm.Get("cwd")
		req.Repo = r.PostForm.Get("repo")
		req.Branch = r.PostForm.Get("branch")
		return req, nil
	}
	if r.Body == nil {
		return req, nil
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil && err != io.EOF {
		return req, fmt.Errorf("invalid create request: %w", err)
	}
	return req, nil
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
	// Chatting with an archived session pulls it out of archive: sending any
	// message unarchives it server-side, so every client (web, REPL, script)
	// gets the same behavior with no per-client logic. The DB write is
	// fail-fast; a failure surfaces as 500 rather than silently leaving the
	// session archived while the message queues.
	if ses.Archived() {
		if err := s.store.Unarchive(ses.ID()); err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
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
// The client identifies itself on the connection (?id=&name=&kind=) so the
// session can register it as a named provider in its registry; a client that
// doesn't (legacy binaries) is registered under a server-generated id. The
// returned id is what the deferred UnregisterExec names, so the cleanup always
// removes exactly this registration.
func (s *Server) handleExec(w http.ResponseWriter, r *http.Request) {
	ses, ok := s.store.Get(chi.URLParam(r, "id"))
	if !ok {
		http.Error(w, "session not found", http.StatusNotFound)
		return
	}
	ch := make(chan api.ExecRequest, 8)
	q := r.URL.Query()
	id := ses.RegisterExec(ch, q.Get("id"), q.Get("name"), q.Get("kind"))
	// A provider that registers in response to a provision request (the
	// persistent host agent's flow) resolves the session-create wait; a
	// plain REPL connection has no pending provision and this is a no-op.
	s.store.ProvisionRegistered(ses.ID(), id)
	defer ses.UnregisterExec(id)

	w.Header().Set("Content-Type", "application/x-ndjson")
	w.WriteHeader(http.StatusOK)
	// Flush the 200 now so the client's ServeExec sees the connection accepted
	// without waiting for the first tool call to arrive on the stream.
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

// handleCancel cancels an in-flight tool run: the session stops the running
// command (killing a local process group or signalling a remote execution
// client) and ends the turn. It is the backend for the UI's Cancel button on a
// running tool block.
func (s *Server) handleCancel(w http.ResponseWriter, r *http.Request) {
	ses, ok := s.store.Get(chi.URLParam(r, "id"))
	if !ok {
		http.Error(w, "session not found", http.StatusNotFound)
		return
	}
	if err := ses.CancelRun(chi.URLParam(r, "call_id")); err != nil {
		http.Error(w, err.Error(), http.StatusNotFound)
		return
	}
	w.WriteHeader(http.StatusOK)
}

// handleStop aborts the session's currently running turn: the model stream is
// cancelled (committing any partial reply, marked interrupted), any running
// tool is stopped, and the turn ends with a stopped marker. It is the backend
// for the UI's Stop button.
func (s *Server) handleStop(w http.ResponseWriter, r *http.Request) {
	ses, ok := s.store.Get(chi.URLParam(r, "id"))
	if !ok {
		http.Error(w, "session not found", http.StatusNotFound)
		return
	}
	if err := ses.Stop(); err != nil {
		// No turn is running (idle or already finished): reject, mirroring
		// handleCancel's unknown-run behavior. The UI reconciles from the
		// turn_completed envelope, so a late click is harmless.
		http.Error(w, err.Error(), http.StatusNotFound)
		return
	}
	w.WriteHeader(http.StatusOK)
}

// handleArchive marks the session archived, folding it out of the active
// sidebar list into the Archived folder (most recently archived first). The
// chat itself is unaffected — history, running turns, and streaming all
// continue. For a session provisioned on an execution host as a worktree
// sandbox, archiving releases the sandbox: the host removes the worktree (so
// the repo does not accumulate abandoned sandboxes). Idempotent: archiving an
// already-archived session is a no-op success.
func (s *Server) handleArchive(w http.ResponseWriter, r *http.Request) {
	id := chi.URLParam(r, "id")
	if err := s.store.Archive(id); err != nil {
		if errors.Is(err, db.ErrNotFound) {
			http.Error(w, "session not found", http.StatusNotFound)
			return
		}
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	// Best-effort, non-blocking: the archive already succeeded; a missed
	// release only leaks the sandbox until the host's next startup cleanup.
	s.store.ReleaseSession(id)
	w.WriteHeader(http.StatusOK)
}

// handleUnarchive restores an archived session to the active list. It is the
// explicit backend for the chat page's Restore button; sending any message to
// an archived session unarchives it through the same store path. Idempotent.
func (s *Server) handleUnarchive(w http.ResponseWriter, r *http.Request) {
	if err := s.store.Unarchive(chi.URLParam(r, "id")); err != nil {
		if errors.Is(err, db.ErrNotFound) {
			http.Error(w, "session not found", http.StatusNotFound)
			return
		}
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	w.WriteHeader(http.StatusOK)
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

// handleExecContext registers the environment context of the connected
// execution provider (system, working directory, files, skills). The REPL
// posts it when it connects so the session can inject it into the model
// and expose load_skill for the reported skills.
func (s *Server) handleExecContext(w http.ResponseWriter, r *http.Request) {
	ses, ok := s.store.Get(chi.URLParam(r, "id"))
	if !ok {
		http.Error(w, "session not found", http.StatusNotFound)
		return
	}
	var ctx api.ExecContext
	if err := json.NewDecoder(r.Body).Decode(&ctx); err != nil {
		http.Error(w, "invalid exec context: "+err.Error(), http.StatusBadRequest)
		return
	}
	ses.SetExecContext(ctx)
	w.WriteHeader(http.StatusOK)
}

// handleExecStatus returns the session's current execution provider status,
// so a client can show which provider is active (and swap in a new one)
// without polling the event bus.
func (s *Server) handleExecStatus(w http.ResponseWriter, r *http.Request) {
	ses, ok := s.store.Get(chi.URLParam(r, "id"))
	if !ok {
		http.Error(w, "session not found", http.StatusNotFound)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(ses.ExecStatus())
}

// handleExecSelect switches the session's active execution provider to the
// client named in the body ("local" for the server process). The deselected
// client stays connected, so it can be selected back without reconnecting. It
// is the backend for the web picker.
func (s *Server) handleExecSelect(w http.ResponseWriter, r *http.Request) {
	ses, ok := s.store.Get(chi.URLParam(r, "id"))
	if !ok {
		http.Error(w, "session not found", http.StatusNotFound)
		return
	}
	var req api.ExecSelectRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "invalid exec select: "+err.Error(), http.StatusBadRequest)
		return
	}
	if err := ses.SelectExec(req.ID); err != nil {
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
// the target of the page's initial load: the chat div issues hx-get to this
// endpoint and swaps the returned innerHTML. Each turn's outcome (token usage,
// or an error) is derived from the persisted query records and interleaved
// after the turn's final message, so a full reload renders the same token and
// error lines the live stream produces — including after a server restart,
// when the live bus no longer holds the turn-completion markers.
func (s *Server) handleView(w http.ResponseWriter, r *http.Request) {
	ses, ok := s.store.Get(chi.URLParam(r, "id"))
	if !ok {
		http.Error(w, "session not found", http.StatusNotFound)
		return
	}
	ps, err := ses.Persisted()
	if err != nil {
		log.Printf("load session %s for view: %v", ses.ID(), err)
		http.Error(w, "internal error", http.StatusInternalServerError)
		return
	}
	turns := session.DeriveTurns(ps)
	turnByUser := make(map[uint64]session.Turn, len(turns))
	for _, t := range turns {
		turnByUser[t.UserSeq] = t
	}
	tools := make(map[string]ToolRunInfo, len(ps.Messages))
	items := make([]viewItem, 0, len(ps.Messages)+len(turns))
	var curTurn uint64 // the user-message seq of the turn currently open
	for i, m := range ps.Messages {
		if m.Role == "user" {
			curTurn = m.Seq
		}
		cm := m.ChatMessage
		items = append(items, viewItem{Message: &cm})
		for _, c := range m.ToolCalls {
			tools[c.ID] = ToolRunInfo{Name: c.Function.Name, Arguments: c.Function.Arguments}
		}
		// A turn ends at the last message before the next user message (or the
		// final message of history); render its footer there so a reload places
		// it exactly where the live turn_completed marker would.
		if i == len(ps.Messages)-1 || ps.Messages[i+1].Role == "user" {
			if t, ok := turnByUser[curTurn]; ok && (t.CachedInput > 0 || t.UncachedInput > 0 || t.Output > 0 || t.Error != "" || t.Stopped) {
				items = append(items, viewItem{Footer: &viewFooter{
					UserSeq:       t.UserSeq,
					CachedInput:   t.CachedInput,
					UncachedInput: t.UncachedInput,
					Output:        t.Output,
					Error:         t.Error,
					Stopped:       t.Stopped,
				}})
			}
		}
	}
	render(w, "view.tmpl", viewData{
		Items: items,
		Tools: tools,
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
	cached, uncached, output := ses.Totals()
	render(w, "layout.tmpl", pageData{
		Title:         "porter",
		Session:       sessID,
		Seq:           ses.Snapshot().Seq,
		Running:       ses.Running(),
		Queue:         ses.QueueDepth(),
		TotalCached:   cached,
		TotalUncached: uncached,
		TotalOutput:   output,
		Archived:      ses.Archived(),
	})
}
