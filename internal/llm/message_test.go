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
