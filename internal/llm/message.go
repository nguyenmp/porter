package llm

// ChatMessage is a single conversation turn sent to the model.
//
// A message carries text, a tool call request (role "assistant"), or a tool
// result (role "tool"). Empty fields are omitted so only the relevant payload
// is serialized.
type ChatMessage struct {
	Role       string     `json:"role"`
	Content    string     `json:"content,omitempty"`
	ToolCalls  []ToolCall `json:"tool_calls,omitempty"`
	ToolCallID string     `json:"tool_call_id,omitempty"`
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
// model produced alongside any tool calls it requested.
func AssistantMessage(content string, calls []ToolCall) ChatMessage {
	return ChatMessage{Role: "assistant", Content: content, ToolCalls: calls}
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
