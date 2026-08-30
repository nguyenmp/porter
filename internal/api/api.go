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
	// SessionExecContextPath registers the environment context of the connected
	// execution provider (system, working directory, files, skills): POST.
	SessionExecContextPath = "/api/sessions/{id}/exec/context"
	// SessionExecStatusPath returns the session's current execution provider
	// status (connected/local + its reported context): GET.
	SessionExecStatusPath = "/api/sessions/{id}/exec/status"
)

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
}

// AppendRequest is the body of POST /api/sessions/{id}/messages.
type AppendRequest struct {
	Content string `json:"content"`
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
)

// Envelope is a single NDJSON line on a session's event bus. Kind selects which
// fields are meaningful.
type Envelope struct {
	Kind    string           `json:"kind"`
	Seq     uint64           `json:"seq,omitempty"`     // KindMessage, KindTurnDone
	Event   *codec.Event     `json:"event,omitempty"`   // KindLLM
	Message *llm.ChatMessage `json:"message,omitempty"` // KindMessage
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
	Queue      int         `json:"queue,omitempty"`       // KindMessage (user)
}

// SessionSummary is one row of the session list (GET /api/sessions). ID and
// CreatedAt are always present; Preview is the first user message truncated to
// a single line, or empty when the session has no messages yet.
type SessionSummary struct {
	ID        string `json:"id"`
	CreatedAt int64  `json:"created_at"`
	Preview   string `json:"preview,omitempty"`
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

// ExecContext is the environment an execution provider reports when it
// connects: what system it runs on, the working directory, files there, and
// the skills it can load. The server injects it into the model's context and
// exposes a load_skill tool backed by the reported skills, so the model knows
// where commands will run and what skills exist without guessing.
type ExecContext struct {
	System string   `json:"system"`
	CWD    string   `json:"cwd"`
	Files  []string `json:"files"`
	Skills []Skill  `json:"skills"`
}

// ExecStatus describes a session's current execution provider. Connected
// reports whether a remote execution client is registered; Kind is "remote"
// when one is, else "local". Context is the environment the connected provider
// reported, set only while it is connected.
type ExecStatus struct {
	Connected bool         `json:"connected"`
	Kind      string       `json:"kind"`
	Context   *ExecContext `json:"context,omitempty"`
}
