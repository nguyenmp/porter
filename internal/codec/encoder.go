package codec

import (
	"encoding/json"
	"io"
)

// Event types emitted on stdout as JSONL.
const (
	TypeMessageDelta   = "message_delta"
	TypeReasoningDelta = "reasoning_delta"
	TypeMessage        = "message"
	TypeUsage          = "usage"
	TypeToolCall       = "tool_call"
	TypeToolResult     = "tool_result"
	TypeError          = "error"
)

// Event is a single JSONL line written to stdout. Reasoning is kept separate
// from content so consumers can surface it without ever folding it into
// resubmitted assistant context. Tool calls and results are emitted separately
// so what the agent ran is visible and auditable.
type Event struct {
	Type         string `json:"type"`
	Role         string `json:"role,omitempty"`
	Delta        string `json:"delta,omitempty"`
	Content      string `json:"content,omitempty"`
	Reasoning    string `json:"reasoning,omitempty"`
	ToolCallID   string `json:"tool_call_id,omitempty"`
	Name         string `json:"name,omitempty"`
	Arguments    string `json:"arguments,omitempty"`
	Result       string `json:"result,omitempty"`
	InputTokens  int    `json:"input_tokens,omitempty"`
	OutputTokens int    `json:"output_tokens,omitempty"`
	Message      string `json:"message,omitempty"`
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
