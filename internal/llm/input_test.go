package llm

import (
	"encoding/json"
	"strings"
	"testing"
)

// decodeItems unmarshals wireInput's output into general maps so a test can
// assert the exact shape porter sends, key by key.
func decodeItems(t *testing.T, msgs []ChatMessage) []map[string]any {
	t.Helper()
	raw, err := wireInput(msgs)
	if err != nil {
		t.Fatalf("wireInput: %v", err)
	}
	out := make([]map[string]any, 0, len(raw))
	for _, r := range raw {
		var m map[string]any
		if err := json.Unmarshal(r, &m); err != nil {
			t.Fatalf("unmarshal item %s: %v", r, err)
		}
		out = append(out, m)
	}
	return out
}

// TestWireInputMessageCarriesTextThenImages pins the input message shape: text
// first as an input_text part, then one input_image part per attached image,
// each holding its bytes inline as a data URL.
func TestWireInputMessageCarriesTextThenImages(t *testing.T) {
	msg := UserMessage("look at these")
	msg.Images = []ImagePart{
		{Name: "a.png", URL: "data:image/png;base64,AAAA"},
		{Name: "b.jpg", URL: "data:image/jpeg;base64,BBBB"},
	}
	items := decodeItems(t, []ChatMessage{msg})
	if len(items) != 1 {
		t.Fatalf("items = %d, want 1", len(items))
	}
	if items[0]["type"] != "message" || items[0]["role"] != "user" {
		t.Errorf("item = %+v, want a user message item", items[0])
	}
	parts, _ := items[0]["content"].([]any)
	if len(parts) != 3 {
		t.Fatalf("parts = %d, want 3 (text + two images)", len(parts))
	}
	first, _ := parts[0].(map[string]any)
	if first["type"] != "input_text" || first["text"] != "look at these" {
		t.Errorf("first part = %+v, want the text as input_text", first)
	}
	for i, want := range []string{"data:image/png;base64,AAAA", "data:image/jpeg;base64,BBBB"} {
		part, _ := parts[i+1].(map[string]any)
		if part["type"] != "input_image" || part["image_url"] != want {
			t.Errorf("part %d = %+v, want input_image %q", i+1, part, want)
		}
	}
}

// TestWireInputSplitsAssistantTurn pins the one-to-many mapping: an assistant
// turn that wrote prose and asked for tools becomes a message item first, then
// one function_call item per call, so each call precedes its result. The
// chain of thought is never sent back.
func TestWireInputSplitsAssistantTurn(t *testing.T) {
	msgs := []ChatMessage{
		AssistantMessage("let me check", "hidden reasoning", []ToolCall{
			{ID: "call_1", Type: "function", Function: ToolFunction{Name: "shell", Arguments: `{"command":"ls"}`}},
			{ID: "call_2", Type: "function", Function: ToolFunction{Name: "shell", Arguments: `{"command":"pwd"}`}},
		}),
		ToolResult("call_1", "exit code: 0"),
	}
	items := decodeItems(t, msgs)
	if len(items) != 4 {
		t.Fatalf("items = %d, want 4 (message + two calls + one result)", len(items))
	}
	if items[0]["type"] != "message" || items[0]["role"] != "assistant" {
		t.Errorf("item 0 = %+v, want an assistant message item", items[0])
	}
	parts, _ := items[0]["content"].([]any)
	part, _ := parts[0].(map[string]any)
	if part["type"] != "output_text" || part["text"] != "let me check" {
		t.Errorf("assistant part = %+v, want the prose as output_text", part)
	}
	for i, want := range []struct{ callID, args string }{
		{"call_1", `{"command":"ls"}`},
		{"call_2", `{"command":"pwd"}`},
	} {
		call := items[i+1]
		if call["type"] != "function_call" || call["call_id"] != want.callID || call["name"] != "shell" || call["arguments"] != want.args {
			t.Errorf("item %d = %+v, want function_call %s", i+1, call, want.callID)
		}
	}
	result := items[3]
	if result["type"] != "function_call_output" || result["call_id"] != "call_1" || result["output"] != "exit code: 0" {
		t.Errorf("item 3 = %+v, want the tool result item", result)
	}
	for _, item := range items {
		if s, _ := json.Marshal(item); strings.Contains(string(s), "reasoning") {
			t.Errorf("reasoning leaked into the request: %s", s)
		}
	}
}

// TestWireInputEmptyMessageKeepsOnePart keeps the shape valid: a message with
// no text and no image still carries one (empty) input_text part, because an
// empty content array is not a shape providers accept.
func TestWireInputEmptyMessageKeepsOnePart(t *testing.T) {
	items := decodeItems(t, []ChatMessage{UserMessage("")})
	if len(items) != 1 {
		t.Fatalf("items = %d, want 1", len(items))
	}
	parts, _ := items[0]["content"].([]any)
	if len(parts) != 1 {
		t.Fatalf("parts = %d, want 1", len(parts))
	}
	part, _ := parts[0].(map[string]any)
	if part["type"] != "input_text" || part["text"] != "" {
		t.Errorf("part = %+v, want an empty input_text part", part)
	}
}

// TestWireInputSystemMessageStaysASystemRole pins that the standing style and
// environment turns keep their system role.
func TestWireInputSystemMessageStaysASystemRole(t *testing.T) {
	items := decodeItems(t, []ChatMessage{SystemMessage("be brief"), UserMessage("hi")})
	if len(items) != 2 {
		t.Fatalf("items = %d, want 2", len(items))
	}
	if items[0]["role"] != "system" || items[1]["role"] != "user" {
		t.Errorf("roles = %v, %v; want system then user", items[0]["role"], items[1]["role"])
	}
}
