package llm

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestMessageSerialization(t *testing.T) {
	msg := AssistantMessage("thinking out loud", "", []ToolCall{
		{ID: "call_1", Type: "function", Function: ToolFunction{Name: "shell", Arguments: `{"command":"ls"}`}},
	})
	data, err := json.Marshal(msg)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	for _, want := range []string{
		`"role":"assistant"`, `"content":"thinking out loud"`,
		`"tool_calls"`, `"id":"call_1"`, `"name":"shell"`,
		`"arguments":"{\"command\":\"ls\"}"`,
	} {
		if !strings.Contains(string(data), want) {
			t.Errorf("missing %q; got: %s", want, data)
		}
	}
}

// TestReasoningSerializationSeparate ensures reasoning round-trips as its own
// field on the committed assistant message (so it survives a reload) and is
// never folded into content.
func TestReasoningSerializationSeparate(t *testing.T) {
	msg := AssistantMessage("visible answer", "hidden chain of thought", nil)
	data, err := json.Marshal(msg)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	got := string(data)
	for _, want := range []string{
		`"content":"visible answer"`,
		`"reasoning":"hidden chain of thought"`,
	} {
		if !strings.Contains(got, want) {
			t.Errorf("missing %q; got: %s", want, got)
		}
	}
	// Reasoning must stay out of content, not be concatenated into it.
	if strings.Contains(got, `"content":"visible answerhidden chain of thought"`) {
		t.Errorf("reasoning leaked into content: %s", got)
	}

	// An empty reasoning value is omitted entirely from the wire.
	plain, err := json.Marshal(AssistantMessage("plain", "", nil))
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if strings.Contains(string(plain), "reasoning") {
		t.Errorf("empty reasoning should be omitted; got: %s", plain)
	}
}

func TestToolSerializationNestsFunction(t *testing.T) {
	tool := Tool{
		Type: "function",
		Function: Function{
			Name:        "shell",
			Description: "run",
			Parameters: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"command": map[string]any{"type": "string"},
				},
				"required": []string{"command"},
			},
		},
	}
	data, err := json.Marshal(tool)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	got := string(data)
	// name/description/parameters must nest under "function", not the root.
	for _, want := range []string{
		`"type":"function"`,
		`"function":{`,
		`"function":{"name":"shell"`,
		`"name":"shell"`,
		`"parameters":{`,
		`"type":"object"`,
		`"required":["command"]`,
	} {
		if !strings.Contains(got, want) {
			t.Errorf("missing %q; got: %s", want, got)
		}
	}
}

func TestToolResultSerialization(t *testing.T) {
	data, err := json.Marshal(ToolResult("call_1", "exit code: 0"))
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	for _, want := range []string{
		`"role":"tool"`, `"tool_call_id":"call_1"`, `"content":"exit code: 0"`,
	} {
		if !strings.Contains(string(data), want) {
			t.Errorf("missing %q; got: %s", want, data)
		}
	}
}

// TestContentAlwaysSerialized is the regression test for a production 400 from
// DeepSeek: an assistant turn that is pure tool calls has no prose, and with
// `json:"content,omitempty"` the wire payload dropped the `content` key
// entirely ({"role":"assistant","tool_calls":[...]}). DeepSeek's strict
// deserializer requires `content` on every message object and rejected the
// whole request deep into a conversation. Every message shape must therefore
// serialize with a `content` key — an explicit empty string satisfies DeepSeek
// and is accepted by all OpenAI-compatible backends.
func TestContentAlwaysSerialized(t *testing.T) {
	cases := []struct {
		name string
		msg  ChatMessage
	}{
		{
			name: "assistant with tool calls and no prose",
			msg: AssistantMessage("", "", []ToolCall{
				{ID: "call_1", Type: "function", Function: ToolFunction{Name: "shell", Arguments: `{"command":"ls"}`}},
			}),
		},
		{
			name: "assistant with reasoning only",
			msg:  AssistantMessage("", "hidden chain of thought", nil),
		},
		{
			name: "empty tool result",
			msg:  ToolResult("call_1", ""),
		},
		{
			name: "empty user message",
			msg:  UserMessage(""),
		},
		{
			name: "empty system message",
			msg:  SystemMessage(""),
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			data, err := json.Marshal(tc.msg)
			if err != nil {
				t.Fatalf("marshal: %v", err)
			}
			got := string(data)
			if !strings.Contains(got, `"content":""`) {
				t.Errorf("message serialized without a content key; DeepSeek rejects this. got: %s", got)
			}
			// The key must be present as a string, never null.
			if strings.Contains(got, `"content":null`) {
				t.Errorf("content must be an explicit empty string, not null. got: %s", got)
			}
		})
	}
}

// TestAssistantToolCallNoProse is a focused version of the production failure:
// the exact wire shape that 400'd, {"role":"assistant","tool_calls":[...]} with
// no content key, must never be produced.
func TestAssistantToolCallNoProse(t *testing.T) {
	msg := AssistantMessage("", "", []ToolCall{
		{ID: "call_1", Type: "function", Function: ToolFunction{Name: "shell", Arguments: `{}`}},
	})
	data, err := json.Marshal(msg)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	got := string(data)
	if !strings.Contains(got, `"role":"assistant"`) {
		t.Fatalf("expected assistant role; got: %s", got)
	}
	if !strings.Contains(got, `"content":""`) {
		t.Errorf("assistant tool-call message must carry an explicit empty content; got: %s", got)
	}
	if !strings.Contains(got, `"tool_calls"`) {
		t.Errorf("tool_calls missing; got: %s", got)
	}
}
