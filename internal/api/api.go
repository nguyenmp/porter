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
	// SessionsPath creates a session: POST.
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
	// SessionExecPath registers a client as the session's execution provider
	// and holds the connection open for exec requests: GET (NDJSON requests).
	SessionExecPath = "/api/sessions/{id}/exec"
	// SessionExecResultPath streams a tool call's output back to the server,
	// after the client runs it: POST (streaming body).
	SessionExecResultPath = "/api/sessions/{id}/exec/{call_id}"
)

// ExecRequest is one tool call the server pushes to a session's execution
// provider.
type ExecRequest struct {
	CallID    string `json:"call_id"`
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
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
	// KindMessage marks a message the server just committed to history, stamped
	// with its seq. This is what reconciles a subscriber with history.
	KindMessage = "message_committed"
	// KindTurnDone marks that a turn finished, carrying its usage.
	KindTurnDone = "turn_completed"
	// KindResync tells a subscriber its `since` is too old to bridge to live;
	// it must refetch history and resubscribe. No further lines follow.
	KindResync = "resync"
)

// Envelope is a single NDJSON line on a session's event bus. Kind selects which
// fields are meaningful.
type Envelope struct {
	Kind    string           `json:"kind"`
	Seq     uint64           `json:"seq,omitempty"`     // KindMessage
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
	Delta       string `json:"delta,omitempty"`        // KindToolResultDelta
	StartedAt   int64  `json:"started_at,omitempty"`   // KindToolStarted, KindToolResult
	FinishedAt  int64  `json:"finished_at,omitempty"`  // KindToolResult
	TurnID      int64  `json:"turn_id,omitempty"`      // KindTurnDone
	Input       int    `json:"input,omitempty"`        // KindTurnDone
	Output      int    `json:"output,omitempty"`       // KindTurnDone
	Error       string `json:"error,omitempty"`        // KindTurnDone
}
