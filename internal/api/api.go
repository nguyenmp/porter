// Package api defines how a thin porter client talks to the server that owns
// conversation state. The server is the source of truth: clients are stateless,
// rendering from a history poll and a live event bus, and the only thing they
// send is a command (create a session, append a user message).
package api

import (
	"porter/internal/codec"
	"porter/internal/llm"
)

// Routes the client can reach. All are served by the server; {id} is a session
// id substituted at request time.
const (
	// SessionsPath is the session collection: POST creates a session,
	// GET lists them (newest first).
	SessionsPath = "/api/sessions"
	// SessionHistoryPath returns a session's authoritative history and seq: GET.
	SessionHistoryPath = "/api/sessions/{id}"
	// SessionViewPath renders a session's committed history as an HTML
	// fragment for the chat page. GET.
	SessionViewPath = "/api/sessions/{id}/view"
	// SessionMessagesPath appends a user message and queues a turn: POST.
	SessionMessagesPath = "/api/sessions/{id}/messages"
	// SessionEventsPath subscribes to a session's event bus: GET (NDJSON).
	SessionEventsPath = "/api/sessions/{id}/events"
	// SessionStreamPath streams a session's event bus as Server-Sent Events.
	SessionStreamPath = "/api/sessions/{id}/stream"
	// SessionRunsPath lists a session's in-flight tool runs (those started but
	// not yet finished), so a client that connects or reconnects mid-run can
	// reconstruct running blocks from the server's authoritative state: GET.
	SessionRunsPath = "/api/sessions/{id}/runs"
	// SessionLivePath returns a session's in-flight LLM stream tail — every
	// live LLM envelope since the last commit, plus the newest live position —
	// so a client whose /view render wiped a live bubble (or that otherwise
	// missed the stream) can re-seed it instead of waiting for the turn's next
	// commit: GET.
	SessionLivePath = "/api/sessions/{id}/live"
	// SessionExecPath registers a client as the session's execution provider
	// and holds the connection open for exec requests: GET (NDJSON requests).
	SessionExecPath = "/api/sessions/{id}/exec"
	// SessionExecResultPath streams a tool call's output back to the server,
	// after the client runs it: POST (streaming body).
	SessionExecResultPath = "/api/sessions/{id}/exec/{call_id}"
	// SessionCancelPath cancels an in-flight tool run, stopping the running
	// command and ending the turn: POST.
	SessionCancelPath = "/api/sessions/{id}/cancel/{call_id}"
	// SessionStopPath aborts the session's currently running turn: the model
	// stream is cancelled (committing any partial reply, marked interrupted),
	// any running tool is stopped, and the turn ends with a stopped marker.
	// POST.
	SessionStopPath = "/api/sessions/{id}/stop"
	// SessionArchivePath marks a session archived, folding it out of the
	// active sidebar list and into the Archived folder: POST. Idempotent.
	SessionArchivePath = "/api/sessions/{id}/archive"
	// SessionUnarchivePath restores an archived session to the active list:
	// POST. Idempotent. Sending any message to an archived session also
	// unarchives it server-side, so this endpoint is the explicit button.
	SessionUnarchivePath = "/api/sessions/{id}/unarchive"
	// SessionRenamePath sets (or clears) a session's custom display name:
	// POST. Names are display-only; an empty (or whitespace-only) name clears
	// back to the first-message preview. Renaming an archived session keeps it
	// archived.
	SessionRenamePath = "/api/sessions/{id}/rename"
	// SessionExecContextPath registers the environment context of the connected
	// execution provider (system, working directory, files, skills): POST.
	SessionExecContextPath = "/api/sessions/{id}/exec/context"
	// SessionExecStatusPath returns the session's current execution provider
	// status (connected/local + its reported context): GET.
	SessionExecStatusPath = "/api/sessions/{id}/exec/status"
	// SessionExecSelectPath switches the session's active execution provider
	// to the client named in the body ("local" for the server process): POST.
	SessionExecSelectPath = "/api/sessions/{id}/exec/select"

	// HostsPath lists the registered execution hosts — persistent agents (e.g.
	// on a laptop) that can provision execution contexts for new sessions: GET.
	HostsPath = "/api/hosts"
	// HostExecPath is a host's persistent connection: the server pushes
	// provision requests down it as NDJSON. The host's id is its path segment,
	// like a session's exec connection: GET.
	HostExecPath = "/api/hosts/{host_id}/exec"
	// HostContextPath registers a host's base environment context (system,
	// default working directory, files, skills): POST.
	HostContextPath = "/api/hosts/{host_id}/context"
	// HostProviderErrorPath reports that a host failed to provision a provider
	// (e.g. the requested working directory does not exist): POST.
	HostProviderErrorPath = "/api/hosts/{host_id}/providers/{provider_id}/error"
)

// RepoRef names one local git repository to sandbox a session in, and the
// base branch ("" = the repo's HEAD) to check out. A sandboxed provision
// carries one RepoRef per repo: the host creates a git worktree for each — a
// fresh branch porter/<provider_id> (the second worktree of the same repo gets
// porter/<provider_id>-2, and so on) — inside a container directory that
// becomes the session's working directory, so a chat can work across several
// repos at once without trampling anyone. Repeating the same repo with
// different branches compares branches side by side.
type RepoRef struct {
	Path   string `json:"path"`
	Branch string `json:"branch,omitempty"`
}

// HostRequest is one message the server pushes to a registered execution host
// over its persistent exec connection. Kind "provision" asks the host to
// create an execution environment (a sandbox) for a session and register
// itself as that session's execution provider — the roadmap's Execution Host:
// provisioning creates an environment and returns an Execution Provider. Kind
// "release" asks the host to tear down a sandbox it created earlier (sent
// when the session that owns it is archived). CWD names a working directory
// for the environment ("" = the host's default). Repos names the local git
// repositories to sandbox the session in (a git worktree each); with no
// Repos, the environment is just the working directory.
type HostRequest struct {
	Kind string `json:"kind"` // "provision" | "release"
	// ProviderID is the id the host's provisioned provider registers under on
	// the session. The server generates it so it is unique and safe.
	ProviderID string `json:"provider_id,omitempty"`
	// SessionID is the session the provider will serve.
	SessionID string    `json:"session_id,omitempty"`
	CWD       string    `json:"cwd,omitempty"`
	Repos     []RepoRef `json:"repos,omitempty"`
}

// ExecRequest is one message the server pushes to a session's execution
// provider over its exec subscription. A normal request carries a tool call to
// run (CallID/Name/Arguments); a cancellation request carries only Cancel=true
// and tells the client to stop its currently running command (the agent runs
// tools one at a time, so there is at most one).
type ExecRequest struct {
	CallID    string `json:"call_id"`
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
	Cancel    bool   `json:"cancel,omitempty"`
}

// SessionInfo is returned by POST /api/sessions. It carries the new session's
// id, its (empty) history, and the seq to resume the bus from.
type SessionInfo struct {
	ID      string            `json:"id"`
	History []llm.ChatMessage `json:"history"`
	Seq     uint64            `json:"seq"`
	// Warning, when set, reports a non-fatal problem from session creation —
	// today: a requested execution host could not be provisioned, so the
	// session falls back to local execution until the host connects.
	Warning string `json:"warning,omitempty"`
}

// CreateRequest is the optional body of POST /api/sessions. Host names an
// execution host to provision the new session's execution provider on ("" or
// "local" uses the server process); CWD and Repos are passed to the host so
// it can create the right environment (Repos = one git worktree per repo).
type CreateRequest struct {
	Host  string    `json:"host,omitempty"`
	CWD   string    `json:"cwd,omitempty"`
	Repos []RepoRef `json:"repos,omitempty"`
}

// HostSummary describes one registered execution host for the web UI's "new
// chat on" picker: a persistent agent (e.g. on a laptop) that can provision
// execution contexts.
type HostSummary struct {
	ID        string       `json:"id"`
	Name      string       `json:"name,omitempty"`
	Kind      string       `json:"kind"` // "host"
	Connected bool         `json:"connected"`
	Context   *ExecContext `json:"context,omitempty"`
}

// HostsResponse is returned by GET /api/hosts.
type HostsResponse struct {
	Hosts []HostSummary `json:"hosts"`
}

// ProviderErrorRequest is the body of POST .../providers/{provider_id}/error.
type ProviderErrorRequest struct {
	Error string `json:"error"`
}

// AppendRequest is the body of POST /api/sessions/{id}/messages.
type AppendRequest struct {
	Content string `json:"content"`
}

// RenameRequest is the body of POST /api/sessions/{id}/rename. Name is the
// session's new custom display name; empty (or whitespace-only) clears it
// back to the first-message preview.
type RenameRequest struct {
	Name string `json:"name"`
}

// SessionHistory is the authoritative conversation state returned by
// GET /api/sessions/{id}. History is exactly the committed messages with
// seq <= Seq; a client that replays the bus with `since=seq` gets the rest with
// no gap and no overlap.
type SessionHistory struct {
	History []llm.ChatMessage `json:"history"`
	Seq     uint64            `json:"seq"`
}

// RunInfo describes one in-flight tool run. Output is the partial result
// accumulated so far (the server is the source of truth for what has been
// produced while the client was disconnected). StartedAt is the server clock.
type RunInfo struct {
	CallID    string `json:"call_id"`
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
	StartedAt int64  `json:"started_at"`
	Output    string `json:"output"`
}

// RunsResponse is returned by GET /api/sessions/{id}/runs. Now is the server's
// clock at response time, so a client can compute a server-accurate elapsed
// time for each run regardless of clock skew.
type RunsResponse struct {
	Now  int64     `json:"now"`
	Runs []RunInfo `json:"runs"`
}

// LiveResponse is returned by GET /api/sessions/{id}/live. Events is the
// session's in-flight LLM stream tail (every live LLM envelope since the last
// commit, in order); Seq is the live position of the newest one. A client that
// re-seeds its streaming view applies Events, skips live envelopes it already
// applied (live_seq <= Seq), and continues from there.
type LiveResponse struct {
	Seq    uint64     `json:"seq"`
	Events []Envelope `json:"events"`
}

// Envelope kinds carried on a session's event bus. An Envelope is the union of
// everything a subscriber can receive: a live LLM Event, a system-side fact
// (tool results, and later subagent or execution notices), or a session
// lifecycle marker.
const (
	// KindLLM wraps a codec.Event from the running turn, for real-time rendering.
	KindLLM = "llm"
	// KindToolResult reports that the agent ran a tool and got this result. It
	// comes from our system, not the model.
	KindToolResult = "tool_result"
	// KindToolResultDelta reports one live chunk of a running tool's output,
	// streamed as the tool produces it. It comes from our system, not the
	// model. The terminal KindToolResult (full result) still follows once the
	// tool exits, so subscribers reconcile to one complete record and the REPL
	// keeps a single audit line.
	KindToolResultDelta = "tool_result_delta"
	// KindToolStarted marks that a tool run began, carrying the server's start
	// clock and the call's arguments. It is emitted once the tool starts (not
	// when it first produces output), so clients can show an honest
	// queued -> running transition with an elapsed timer even for silent
	// tools. The terminal KindToolResult then carries start/finish clocks so
	// the final duration is server-derived.
	KindToolStarted = "tool_started"
	// KindToolCancelled reports that a running tool was aborted (e.g. the user
	// clicked Cancel in the UI). It carries the partial output accumulated so
	// far plus the server start/finish clocks, and is terminal like
	// KindToolResult: the run leaves the in-flight set and the turn ends. The
	// committed role-"tool" message still follows, marked cancelled, so history
	// is transparent.
	KindToolCancelled = "tool_cancelled"
	// KindMessage marks a message the server just committed to history, stamped
	// with its seq. This is what reconciles a subscriber with history.
	KindMessage = "message_committed"
	// KindTurnDone marks that a turn finished, carrying its usage split into
	// cached/uncached input and output tokens.
	KindTurnDone = "turn_completed"
	// KindResync tells a subscriber its `since` is too old to bridge to live;
	// it must refetch history and resubscribe. No further lines follow.
	KindResync = "resync"
	// KindExecStatus broadcasts the session's current execution provider
	// status (connected/local + its reported context) whenever it changes. It
	// comes from our system, not the model.
	KindExecStatus = "exec_status"
	// KindSessionRenamed broadcasts that the session's custom display name
	// changed (set or cleared). It is live-only, like KindExecStatus — never
	// replayed, because a freshly loaded page gets the name from GET
	// /api/sessions.
	KindSessionRenamed = "session_renamed"
)

// Envelope is a single NDJSON line on a session's event bus. Kind selects which
// fields are meaningful.
type Envelope struct {
	Kind string `json:"kind"`
	Seq  uint64 `json:"seq,omitempty"` // KindMessage, KindTurnDone
	// LiveSeq is the session's monotonic live-stream position, stamped on
	// every KindLLM envelope (and only those) as it is published. It is what
	// lets a client deduplicate the live tail when it is replayed — on an SSE
	// reconnect, or a /live re-seed after a /view swap: an envelope whose
	// LiveSeq is not newer than the last one the client applied is skipped, so
	// streamed text never double-appends. Unlike Seq (committed bus positions),
	// live positions are not persisted; they are rebuilt from zero at startup,
	// when no turns are in flight.
	LiveSeq uint64           `json:"live_seq,omitempty"` // KindLLM
	Event   *codec.Event     `json:"event,omitempty"`    // KindLLM
	Message *llm.ChatMessage `json:"message,omitempty"`  // KindMessage
	// MessageHTML is the server-rendered HTML for a committed assistant
	// message (KindMessage). It is set only for assistant messages with
	// content, so the SSE client can render the committed copy exactly as the
	// /view endpoint would (markdown) instead of approximating it client-side.
	// Plaintext messages (user, tool) are excluded; the client already renders
	// those identically to /view.
	MessageHTML string `json:"message_html,omitempty"` // KindMessage
	ToolCallID  string `json:"tool_call_id,omitempty"` // KindToolResult, KindToolResultDelta, KindToolStarted
	Name        string `json:"name,omitempty"`         // KindToolResult, KindToolResultDelta, KindToolStarted
	Arguments   string `json:"arguments,omitempty"`    // KindToolResult, KindToolStarted
	Result      string `json:"result,omitempty"`       // KindToolResult
	// ToolOutput is structured metadata about a tool result's size and model-view
	// presentation (total/shown bytes, truncation, recall). It is set on
	// KindToolResult (normal, cancelled, and read_output) and carried on the
	// KindMessage commit, so the live UI renders the same badge /view renders
	// from the persisted copy.
	ToolOutput *llm.ToolOutputMeta `json:"tool_output,omitempty"` // KindToolResult, KindMessage
	Delta      string              `json:"delta,omitempty"`       // KindToolResultDelta
	StartedAt  int64               `json:"started_at,omitempty"`  // KindToolStarted, KindToolResult
	FinishedAt int64               `json:"finished_at,omitempty"` // KindToolResult
	TurnID     int64               `json:"turn_id,omitempty"`     // KindTurnDone
	TurnSeq    uint64              `json:"turn_seq,omitempty"`    // KindTurnDone (the user message seq that started the turn)
	// CachedInput/UncachedInput are the turn's prompt-token split (cache hits
	// vs misses); Output is its completion tokens. Total input is their sum,
	// derived where display needs it. KindTurnDone.
	CachedInput   int `json:"cached_input,omitempty"`
	UncachedInput int `json:"uncached_input,omitempty"`
	Output        int `json:"output,omitempty"` // KindTurnDone
	// TotalCachedInput/TotalUncachedInput/TotalOutput are the session's running
	// token totals — the sum over every completed turn, not just this one — so
	// a client can show a session total below the input box without re-deriving
	// it. The value is authoritative at the marker's bus position: events are
	// ordered, so a client that always sets its total from these fields
	// converges on the true session total even across replays (setting, never
	// adding, so the page-load baseline can't be double-counted). KindTurnDone.
	TotalCachedInput   int    `json:"total_cached_input,omitempty"`
	TotalUncachedInput int    `json:"total_uncached_input,omitempty"`
	TotalOutput        int    `json:"total_output,omitempty"` // KindTurnDone
	Error              string `json:"error,omitempty"`        // KindTurnDone
	// Stopped reports that the user aborted the turn (the Stop button) rather
	// than completing or failing. A stopped turn's partial reply (if any) is
	// already committed, marked interrupted. KindTurnDone.
	Stopped bool `json:"stopped,omitempty"`
	// Queue is the number of turns still waiting in the server's queue behind
	// the message this envelope commits. It is set on user KindMessage
	// envelopes (the server reports, at each turn start, how many messages are
	// queued after it) so the web client can show a live queue-depth indicator
	// without polling.
	// ExecStatus is the session's current execution provider status, published
	// whenever it changes (a client connected, disconnected, or swapped in).
	// KindExecStatus.
	ExecStatus *ExecStatus `json:"exec_status,omitempty"` // KindExecStatus
	// SessionName is the session's custom display name after a rename ("" when
	// cleared back to the preview fallback). KindSessionRenamed.
	SessionName string `json:"session_name,omitempty"` // KindSessionRenamed
	Queue       int    `json:"queue,omitempty"`        // KindMessage (user)
}

// SessionSummary is one row of the session list (GET /api/sessions). ID and
// CreatedAt are always present; Preview is the first user message truncated to
// a single line, or empty when the session has no messages yet.
type SessionSummary struct {
	ID        string `json:"id"`
	CreatedAt int64  `json:"created_at"`
	// Name is the session's custom display name ("" when none is set; the
	// sidebar falls back to Preview). Omitted from the JSON when empty.
	Name    string `json:"name,omitempty"`
	Preview string `json:"preview,omitempty"`
	// ArchivedAt is the epoch-ms time the session was archived, or 0 when it
	// is active. Omitted from the JSON for active sessions so the sidebar
	// response stays terse.
	ArchivedAt int64 `json:"archived_at,omitempty"`
}

// SessionsResponse is returned by GET /api/sessions, newest first. The server
// is the source of truth for which sessions exist; a client (e.g. the web
// sidebar) renders these instead of keeping its own registry.
type SessionsResponse struct {
	Sessions []SessionSummary `json:"sessions"`
}

// Skill is the metadata for one discovered skill, as reported by an execution
// provider. Name and Description are what the model sees in the load_skill
// tool; Path is where the provider reads the full SKILL.md body from when the
// model asks to load it. It is never shown to the model.
type Skill struct {
	Name        string `json:"name"`
	Description string `json:"description"`
	Path        string `json:"path"`
}

// CLI is one command-line tool a provider declares available in its
// environment, loaded from the provider's ~/.porter/clis.json manifest. Name
// is the binary the model should invoke via the shell tool; Description is a
// one-line hint ("Graphite stack management (submit, split, restack, up/down)").
// CLIs are shown in the model's environment message as shell-usable tools;
// they are not tools in their own right.
type CLI struct {
	Name        string `json:"name"`
	Description string `json:"description"`
}

// MCPTool is one tool an MCPServer exposes, as discovered at connect time.
type MCPTool struct {
	Name        string         `json:"name"`
	Description string         `json:"description"`
	InputSchema map[string]any `json:"inputSchema,omitempty"`
}

// MCPServer is the metadata a provider reports for an MCP server it hosts —
// e.g. a laptop-only server reachable only from the provider's machine (a
// corporate VPN). The server-side hub uses it to list the server in FindMCP
// and to route CallMCP down the provider's exec channel, so the server's
// credentials never leave the provider. Tools are the server's tool list as
// discovered when the provider connected.
type MCPServer struct {
	Name        string `json:"name"`
	Description string `json:"description"`
	// Host is the execution host that serves this server (the provider's id),
	// shown in FindMCP so the model knows where the call will run.
	Host   string    `json:"host,omitempty"`
	Status string    `json:"status,omitempty"` // "ok" | "error" | "pending"
	Error  string    `json:"error,omitempty"`
	Tools  []MCPTool `json:"tools,omitempty"`
}

// ExecContext is the environment an execution provider reports when it
// connects: what system it runs on, the working directory, files there, the
// skills it can load, the CLI tools it declares available, and the MCP
// servers it hosts. The server injects it into the model's context and
// exposes a load_skill tool backed by the reported skills, so the model knows
// where commands will run and what skills exist without guessing. ID/Name
// identify the reporting client so the server can attach the context to the
// right registry entry; the REPL posts its context before opening the exec
// connection, so these fields are how the context finds its client.
type ExecContext struct {
	ID     string `json:"id,omitempty"`
	Name   string `json:"name,omitempty"`
	System string `json:"system"`
	CWD    string `json:"cwd"`
	// Repos lists the local git repositories the provider can sandbox
	// sessions in (a git worktree per session). Execution hosts discover the
	// repos under the user's home directory and report them so the web UI's
	// "new chat on" picker can offer them; the list is not part of the model's
	// environment message.
	Repos  []string `json:"repos,omitempty"`
	Files  []string `json:"files"`
	Skills []Skill  `json:"skills"`
	// CLIs lists the command-line tools the provider declares available (from
	// ~/.porter/clis.json), rendered in the model's environment message as
	// tools to run via shell. Not part of the load_skill surface.
	CLIs []CLI `json:"clis,omitempty"`
	// MCPServers lists the MCP servers this provider hosts (executed on the
	// provider's machine), so the server-side hub can expose them through
	// FindMCP and route CallMCP down this provider's exec channel. Empty when
	// the provider hosts none.
	MCPServers []MCPServer `json:"mcp_servers,omitempty"`
}

// ExecClient describes one execution provider a session can run tools with:
// the local server process (always present) or a connected remote client.
// Name is a human-readable label (e.g. the client's hostname), shown in the
// picker; ID is the stable identity the selector addresses.
type ExecClient struct {
	ID        string       `json:"id"`
	Name      string       `json:"name,omitempty"`
	Kind      string       `json:"kind"` // "local" | "remote" (future: "cloud")
	Connected bool         `json:"connected"`
	Context   *ExecContext `json:"context,omitempty"`
}

// ExecStatus describes a session's execution provider state. Connected
// reports whether the active provider is a remote execution client; Kind is
// "remote" when one is active, else "local". Context is the environment the
// active provider reported (the local server's own context when local is
// active). ActiveID is the id of the active provider ("local" for the server
// process), and Clients is the full registry of providers the selector
// renders — local plus every connected remote client.
type ExecStatus struct {
	Connected bool         `json:"connected"`
	Kind      string       `json:"kind"`
	Context   *ExecContext `json:"context,omitempty"`
	ActiveID  string       `json:"active_id,omitempty"`
	Clients   []ExecClient `json:"clients,omitempty"`
}

// ExecSelectRequest is the body of POST /api/sessions/{id}/exec/select. ID is
// the execution client to make active; "local" selects the server process.
type ExecSelectRequest struct {
	ID string `json:"id"`
}
