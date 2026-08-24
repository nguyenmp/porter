package codec

import (
	"encoding/json"
	"io"
)

// EventType identifies the kind of an Event.
type EventType string

// Event types emitted on stdout as JSONL. An Event is purely what the LLM
// produced: streamed content, reasoning, a completed message, usage, and the
// tool calls the model requested. System-side facts (like tool results) are not
// Events; they travel as their own api.Envelope kind.
const (
	TypeMessageDelta   EventType = "message_delta"
	TypeReasoningDelta EventType = "reasoning_delta"
	TypeMessage        EventType = "message"
	TypeUsage          EventType = "usage"
	TypeToolCall       EventType = "tool_call"
	// TypeToolCallDelta is one streamed fragment of a tool call (an id/name
	// sighting or a slice of arguments), emitted live as it arrives so
	// consumers can render the call while it forms. The assembled
	// TypeToolCall is still emitted at stream end for the commit path.
	TypeToolCallDelta EventType = "tool_call_delta"
)

// Event is a single JSONL line written to stdout. Reasoning is kept separate
// from content so consumers can surface it without ever folding it into
// resubmitted assistant context. Tool calls are emitted separately so what the
// agent ran is visible and auditable.
type Event struct {
	Type       EventType `json:"type"`
	Role       string    `json:"role,omitempty"`
	Delta      string    `json:"delta,omitempty"`
	Content    string    `json:"content,omitempty"`
	Reasoning  string    `json:"reasoning,omitempty"`
	ToolCallID string    `json:"tool_call_id,omitempty"`
	Name       string    `json:"name,omitempty"`
	Arguments  string    `json:"arguments,omitempty"`
	// Index identifies which tool call in a stream this event belongs to
	// (tool_call_delta and the assembled tool_call). Zero is the first call.
	Index        int `json:"index,omitempty"`
	InputTokens  int `json:"input_tokens,omitempty"`
	OutputTokens int `json:"output_tokens,omitempty"`
}

// Encoder writes Events as JSONL to an underlying writer, one per line.
type Encoder struct {
	w io.Writer
}

// NewEncoder returns an Encoder writing to w.
func NewEncoder(w io.Writer) *Encoder {
	return &Encoder{w: w}
}

// Write serializes and writes one event followed by a newline.
func (e *Encoder) Write(ev Event) error {
	data, err := json.Marshal(ev)
	if err != nil {
		return err
	}
	data = append(data, '\n')
	_, err = e.w.Write(data)
	return err
}
