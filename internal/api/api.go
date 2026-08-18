// Package api defines how a thin porter client talks to the stateless server
// that runs the agent loop. The server owns the LLM connection and tool
// execution. The client owns no conversation state, so it sends the full
// history with every turn.
package api

import (
	"porter/internal/llm"
)

// StreamPath is the single endpoint the server exposes.
const StreamPath = "/api/stream"

// StreamRequest is the body of a stream call. It holds the full conversation
// history, including the latest user message.
type StreamRequest struct {
	History []llm.ChatMessage `json:"history"`
}

// Completion is the final line of a stream. It carries the finished turn (final
// text, usage, and the full history) so the client can pick up where it left
// off on the next call. Completed marks this line as the trailer rather than a
// codec.Event line in the same NDJSON body.
type Completion struct {
	Completed bool               `json:"completed"`
	Text      string             `json:"text,omitempty"`
	Input     int                `json:"input,omitempty"`
	Output    int                `json:"output,omitempty"`
	History   []llm.ChatMessage  `json:"history,omitempty"`
}