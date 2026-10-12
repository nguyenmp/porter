package db

import (
	"testing"

	"porter/internal/llm"
)

// TestImagesRoundTrip checks that images attached to a user message survive a
// save and reload: the bytes live in their own column and come back on the
// message, while content stays plain text.
func TestImagesRoundTrip(t *testing.T) {
	d := openTemp(t)
	id, err := d.CreateSession(100)
	if err != nil {
		t.Fatalf("CreateSession: %v", err)
	}
	msg := llm.UserMessage("what is this?")
	msg.Images = []llm.ImagePart{
		{Name: "shot.png", URL: "data:image/png;base64,AAAA"},
		{Name: "second.jpg", URL: "data:image/jpeg;base64,BBBB"},
	}
	if err := d.AppendMessage(id, Message{Seq: 1, ChatMessage: msg}); err != nil {
		t.Fatalf("AppendMessage: %v", err)
	}

	got, err := d.LoadSession(id)
	if err != nil {
		t.Fatalf("LoadSession: %v", err)
	}
	if len(got.Messages) != 1 {
		t.Fatalf("messages = %d, want 1", len(got.Messages))
	}
	m := got.Messages[0]
	if m.Content != "what is this?" {
		t.Errorf("content = %q, want %q", m.Content, "what is this?")
	}
	if len(m.Images) != 2 {
		t.Fatalf("images = %d, want 2", len(m.Images))
	}
	if m.Images[0].Name != "shot.png" || m.Images[0].URL != "data:image/png;base64,AAAA" {
		t.Errorf("images[0] = %+v, want shot.png with its data URL", m.Images[0])
	}
	if m.Images[1].Name != "second.jpg" || m.Images[1].URL != "data:image/jpeg;base64,BBBB" {
		t.Errorf("images[1] = %+v, want second.jpg with its data URL", m.Images[1])
	}
}

// TestMessagesWithoutImagesLoadWithoutImages checks that a text-only message
// comes back with no images, so the new column's default does not invent any.
func TestMessagesWithoutImagesLoadWithoutImages(t *testing.T) {
	d := openTemp(t)
	id, err := d.CreateSession(100)
	if err != nil {
		t.Fatalf("CreateSession: %v", err)
	}
	if err := d.AppendMessage(id, Message{Seq: 1, ChatMessage: llm.UserMessage("plain")}); err != nil {
		t.Fatalf("AppendMessage: %v", err)
	}
	got, err := d.LoadSession(id)
	if err != nil {
		t.Fatalf("LoadSession: %v", err)
	}
	if len(got.Messages[0].Images) != 0 {
		t.Errorf("images = %+v, want none", got.Messages[0].Images)
	}
}
