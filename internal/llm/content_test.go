package llm

import (
	"encoding/json"
	"strings"
	"testing"
)

// TestImagesBecomeContentParts checks that an attached image is merged into
// the message's content as an image_url part, which is the only shape the Chat
// Completions API accepts for image input, and that the internal `images`
// field is dropped so the provider never sees a field it does not know.
func TestImagesBecomeContentParts(t *testing.T) {
	msg := UserMessage("what is this?")
	msg.Images = []ImagePart{{Name: "shot.png", URL: "data:image/png;base64,AAAA"}}

	wire, err := wireMessages([]ChatMessage{msg})
	if err != nil {
		t.Fatalf("wireMessages: %v", err)
	}
	got := string(wire[0])
	for _, want := range []string{
		`"role":"user"`,
		`"type":"text","text":"what is this?"`,
		`"type":"image_url"`,
		`"url":"data:image/png;base64,AAAA"`,
	} {
		if !strings.Contains(got, want) {
			t.Errorf("missing %q; got: %s", want, got)
		}
	}
	if strings.Contains(got, `"images"`) {
		t.Errorf("images field should not be sent to the provider; got: %s", got)
	}
	// Content must be an array of parts, not a string.
	var obj map[string]json.RawMessage
	if err := json.Unmarshal(wire[0], &obj); err != nil {
		t.Fatalf("unmarshal wire message: %v", err)
	}
	var parts []contentPart
	if err := json.Unmarshal(obj["content"], &parts); err != nil {
		t.Fatalf("content is not a parts array: %v (%s)", err, obj["content"])
	}
	if len(parts) != 2 || parts[0].Type != "text" || parts[1].Type != "image_url" {
		t.Errorf("parts = %+v, want a text part then an image part", parts)
	}
}

// TestImageOnlyMessageHasNoTextPart checks that a message carrying only an
// image (a pasted screenshot with no comment) sends no empty text part.
func TestImageOnlyMessageHasNoTextPart(t *testing.T) {
	msg := UserMessage("")
	msg.Images = []ImagePart{{Name: "shot.png", URL: "data:image/png;base64,AAAA"}}

	wire, err := wireMessages([]ChatMessage{msg})
	if err != nil {
		t.Fatalf("wireMessages: %v", err)
	}
	var obj map[string]json.RawMessage
	if err := json.Unmarshal(wire[0], &obj); err != nil {
		t.Fatalf("unmarshal wire message: %v", err)
	}
	var parts []contentPart
	if err := json.Unmarshal(obj["content"], &parts); err != nil {
		t.Fatalf("content is not a parts array: %v", err)
	}
	if len(parts) != 1 || parts[0].Type != "image_url" {
		t.Errorf("parts = %+v, want a single image part", parts)
	}
}

// TestTextOnlyMessageWireUnchanged checks that a message without images is
// sent exactly as the message marshals on its own: the image path must not
// disturb the shape every provider already accepts.
func TestTextOnlyMessageWireUnchanged(t *testing.T) {
	msgs := []ChatMessage{
		SystemMessage("env"),
		UserMessage("hello"),
		AssistantMessage("hi", "", nil),
		ToolResult("call_1", "exit code: 0"),
	}
	wire, err := wireMessages(msgs)
	if err != nil {
		t.Fatalf("wireMessages: %v", err)
	}
	for i, m := range msgs {
		want, err := json.Marshal(m)
		if err != nil {
			t.Fatalf("marshal: %v", err)
		}
		if string(wire[i]) != string(want) {
			t.Errorf("message %d changed\n got: %s\nwant: %s", i, wire[i], want)
		}
	}
	// A text-only message must not carry an images key at all.
	if strings.Contains(string(wire[1]), "images") {
		t.Errorf("text-only message carries an images key: %s", wire[1])
	}
}
