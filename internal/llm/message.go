package llm

// ChatMessage is a single conversation turn sent to the model.
//
// A message carries text, a tool call request (role "assistant"), or a tool
// result (role "tool"). Empty fields are omitted so only the relevant payload
// is serialized.
type ChatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content,omitempty"`
	// Reasoning is the model's private chain-of-thought, streamed separately
	// from Content by reasoning-capable providers. It is persisted on the
	// committed assistant message so it survives a reload (server-side /view
	// render and SSE replay), and is never folded back into Content when the
	// message is resubmitted to the model.
	Reasoning  string     `json:"reasoning,omitempty"`
	ToolCalls  []ToolCall `json:"tool_calls,omitempty"`
	ToolCallID string     `json:"tool_call_id,omitempty"`
	// StartedAt/FinishedAt are server-clock epoch milliseconds for a tool run,
	// stamped by the agent when the tool runs. They are excluded from every
	// JSON encoding (json:"-") so timing never leaks into the LLM request or
	// the history API; the /view endpoint reads them directly from the
	// in-memory history to render reload timing.
	StartedAt  int64 `json:"-"`
	FinishedAt int64 `json:"-"`
	// Cancelled reports that a tool run was aborted by the user before it
	// completed. It is set on the committed role-"tool" message so history (and
	// /view) can render the result as cancelled rather than a normal exit.
	// Unlike StartedAt/FinishedAt it IS serialized (json:"cancelled,omitempty")
	// so the SSE replay of the committed message carries it to a reconnecting
	// client, and so the model knows on the next turn that the previous run was
	// aborted. It only appears when true.
	Cancelled bool `json:"cancelled,omitempty"`
}

// ToolCall is a request the assistant made to run a named tool. Arguments is
// the raw, possibly incomplete JSON string as streamed by the model.
type ToolCall struct {
	ID       string       `json:"id"`
	Type     string       `json:"type"`
	Function ToolFunction `json:"function"`
}

// ToolFunction names a tool and its JSON-encoded arguments.
type ToolFunction struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

// Tool describes a callable tool to the model via a JSON schema of its input.
// This matches the Chat Completions format: `type` sits at the top level while
// name/description/parameters live under a nested `function` object.
type Tool struct {
	Type     string   `json:"type"`
	Function Function `json:"function"`
}

// Function carries a tool's model-facing name, description, and input schema.
type Function struct {
	Name        string         `json:"name"`
	Description string         `json:"description,omitempty"`
	Parameters  map[string]any `json:"parameters"`
}

// AssistantMessage builds an assistant message that may carry the content the
// model produced, its (separate) reasoning, and any tool calls it requested.
func AssistantMessage(content, reasoning string, calls []ToolCall) ChatMessage {
	return ChatMessage{Role: "assistant", Content: content, Reasoning: reasoning, ToolCalls: calls}
}

// ToolResult builds a role-"tool" message reporting the outcome of a tool run,
// keyed to the assistant's calling tool call by id.
func ToolResult(id, result string) ChatMessage {
	return ChatMessage{Role: "tool", ToolCallID: id, Content: result}
}

// UserMessage builds a plain user message.
func UserMessage(content string) ChatMessage {
	return ChatMessage{Role: "user", Content: content}
}
