package llm

// ChatMessage is a single conversation turn sent to the model.
//
// A message carries text, a tool call request (role "assistant"), or a tool
// result (role "tool"). Empty fields are omitted so only the relevant payload
// is serialized — except Content, which is ALWAYS present on the wire: an
// assistant turn that is pure tool calls has no prose, and some providers
// (notably DeepSeek) reject a message object that lacks the `content` key
// entirely, while every OpenAI-compatible backend accepts an explicit empty
// string. So Content is tagged `json:"content"` (no omitempty), and an empty
// value serializes as "content":"".
type ChatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
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
	// ToolOutput is structured metadata about a tool result's size and model-view
	// presentation (see ToolOutputMeta): total/shown bytes, whether the model
	// view truncated the result, and recall details for recall_tool_output results. It
	// is json:"-" so it is never serialized into the LLM request payload; the
	// DB persists it explicitly and the UI reads it from the committed message
	// or the bus envelope.
	ToolOutput *ToolOutputMeta `json:"-"`
	Cancelled  bool            `json:"cancelled,omitempty"`
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

// SystemMessage builds a role-"system" message carrying environment or
// instruction context (e.g. the execution provider's system, working
// directory, and available skills).
func SystemMessage(content string) ChatMessage {
	return ChatMessage{Role: "system", Content: content}
}

// ToolOutputMeta is structured metadata about a tool result's size and how the
// model view presented it. It is persisted in the database and rendered in the
// UI as a size/truncation badge, and is deliberately json:"-" on ChatMessage so
// it is never serialized into the LLM request payload: a role-"tool" message
// must carry only standard fields (some providers reject unknown ones), and
// sending it would waste tokens on every request.
type ToolOutputMeta struct {
	// Truncated reports that the model view showed less than the full output
	// (a head + tail slice). The full output is still stored in the DB and
	// broadcast to the UI; only the model's view is trimmed.
	Truncated bool `json:"truncated"`
	// TotalBytes is the full output size in bytes.
	TotalBytes int `json:"total_bytes"`
	// ShownBytes is how many bytes the model view showed: HeadBytes+TailBytes
	// when Truncated, else TotalBytes. For a recall (recall_tool_output) result it is
	// the window size served to the model's context.
	ShownBytes int `json:"shown_bytes"`
	// Recall marks a recall_tool_output result: the window bytes were served to the
	// model's context for the current turn only, and the persisted/broadcast
	// copy is a short placeholder rather than the window (the full output
	// lives once, under the source tool result). The model-view projection
	// keeps a Recall message's content intact — it is the one exception to
	// truncation.
	Recall bool `json:"recall,omitempty"`
	// SourceCallID is the tool_call_id the recall_tool_output read from (Recall only).
	SourceCallID string `json:"source_call_id,omitempty"`
	// Offset is the byte offset the recall_tool_output served from (Recall only).
	Offset int `json:"offset,omitempty"`
	// MaxBytes is the size of the window served to the model (Recall only).
	MaxBytes int `json:"max_bytes,omitempty"`
}
