package db

import (
	"errors"
	"path/filepath"
	"testing"

	"porter/internal/llm"
)

// openTemp opens a fresh database in a temp dir and registers its cleanup.
func openTemp(t *testing.T) *DB {
	t.Helper()
	d, err := Open(filepath.Join(t.TempDir(), "porter.db"))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	t.Cleanup(func() { d.Close() })
	return d
}

func TestCreateSessionReturnsUniqueIDs(t *testing.T) {
	d := openTemp(t)
	a, err := d.CreateSession(1)
	if err != nil {
		t.Fatalf("CreateSession(1): %v", err)
	}
	b, err := d.CreateSession(2)
	if err != nil {
		t.Fatalf("CreateSession(2): %v", err)
	}
	if a == b {
		t.Errorf("session ids not unique: %d == %d", a, b)
	}
	if a < 1 || b < 1 {
		t.Errorf("session ids should be positive, got %d and %d", a, b)
	}
}

func TestMessageRoundTrip(t *testing.T) {
	d := openTemp(t)
	id, err := d.CreateSession(100)
	if err != nil {
		t.Fatalf("CreateSession: %v", err)
	}

	want := []Message{
		{Seq: 1, ChatMessage: llm.UserMessage("hello")},
		{
			Seq: 2,
			ChatMessage: llm.ChatMessage{
				Role:      "assistant",
				Content:   "reply with a tool",
				Reasoning: "private chain of thought",
				ToolCalls: []llm.ToolCall{{
					ID:   "call-1",
					Type: "function",
					Function: llm.ToolFunction{
						Name:      "shell",
						Arguments: `{"command":"echo hi"}`,
					},
				}},
			},
		},
		{
			Seq:         3,
			ChatMessage: llm.ChatMessage{Role: "tool", ToolCallID: "call-1", Content: "hi\nexit code: 0", StartedAt: 50, FinishedAt: 60},
		},
	}
	for _, m := range want {
		if err := d.AppendMessage(id, m); err != nil {
			t.Fatalf("AppendMessage(seq %d): %v", m.Seq, err)
		}
	}

	got, err := d.LoadSession(id)
	if err != nil {
		t.Fatalf("LoadSession: %v", err)
	}
	if got.CreatedAt != 100 {
		t.Errorf("created_at = %d, want 100", got.CreatedAt)
	}
	if len(got.Messages) != len(want) {
		t.Fatalf("got %d messages, want %d", len(got.Messages), len(want))
	}
	for i, w := range want {
		g := got.Messages[i]
		if g.Seq != w.Seq || g.Role != w.Role || g.Content != w.Content || g.Reasoning != w.Reasoning {
			t.Errorf("message %d = %+v, want %+v", i, g, w)
		}
		if g.ToolCallID != w.ToolCallID || g.StartedAt != w.StartedAt || g.FinishedAt != w.FinishedAt {
			t.Errorf("message %d metadata = %+v, want %+v", i, g, w)
		}
		// The tool call must survive the JSON round-trip, including arguments.
		if len(g.ToolCalls) != len(w.ToolCalls) {
			t.Errorf("message %d tool calls = %+v, want %+v", i, g.ToolCalls, w.ToolCalls)
			continue
		}
		for j := range w.ToolCalls {
			if g.ToolCalls[j] != w.ToolCalls[j] {
				t.Errorf("message %d tool call %d = %+v, want %+v", i, j, g.ToolCalls[j], w.ToolCalls[j])
			}
		}
	}
	if got.MaxSeq != 3 {
		t.Errorf("max seq = %d, want 3", got.MaxSeq)
	}
}

func TestMessagesLoadInSeqOrder(t *testing.T) {
	d := openTemp(t)
	id, err := d.CreateSession(1)
	if err != nil {
		t.Fatalf("CreateSession: %v", err)
	}
	// Append deliberately out of order; the load must sort by seq.
	for _, m := range []Message{
		{Seq: 3, ChatMessage: llm.UserMessage("third")},
		{Seq: 1, ChatMessage: llm.UserMessage("first")},
		{Seq: 2, ChatMessage: llm.UserMessage("second")},
	} {
		if err := d.AppendMessage(id, m); err != nil {
			t.Fatalf("AppendMessage(seq %d): %v", m.Seq, err)
		}
	}
	got, err := d.LoadSession(id)
	if err != nil {
		t.Fatalf("LoadSession: %v", err)
	}
	var seqs []uint64
	for _, m := range got.Messages {
		seqs = append(seqs, m.Seq)
	}
	if len(seqs) != 3 || seqs[0] != 1 || seqs[1] != 2 || seqs[2] != 3 {
		t.Errorf("message seqs = %v, want [1 2 3]", seqs)
	}
}

func TestListSessionsNewestFirstWithPreview(t *testing.T) {
	d := openTemp(t)
	// Older session first; it gets a user message so its preview is populated.
	oldID, err := d.CreateSession(10)
	if err != nil {
		t.Fatalf("CreateSession: %v", err)
	}
	if err := d.AppendMessage(oldID, Message{Seq: 1, ChatMessage: llm.UserMessage("first message here")}); err != nil {
		t.Fatalf("AppendMessage: %v", err)
	}
	if _, err := d.CreateSession(20); err != nil {
		t.Fatalf("CreateSession: %v", err)
	}

	list, err := d.ListSessions()
	if err != nil {
		t.Fatalf("ListSessions: %v", err)
	}
	if len(list) != 2 {
		t.Fatalf("got %d sessions, want 2: %+v", len(list), list)
	}
	if list[0].CreatedAt != 20 || list[1].CreatedAt != 10 {
		t.Errorf("order = %+v, want newest (20) first", list)
	}
	// Preview source: the raw first user message on the older session, none on
	// the empty one.
	if list[1].FirstUser != "first message here" {
		t.Errorf("older preview = %q, want %q", list[1].FirstUser, "first message here")
	}
	if list[0].FirstUser != "" {
		t.Errorf("empty session preview = %q, want empty", list[0].FirstUser)
	}
}

func TestLoadUnknownSession(t *testing.T) {
	d := openTemp(t)
	_, err := d.LoadSession(42)
	if !errors.Is(err, ErrNotFound) {
		t.Errorf("LoadSession(42) error = %v, want ErrNotFound", err)
	}
}

func TestPersistsAcrossReopen(t *testing.T) {
	path := filepath.Join(t.TempDir(), "porter.db")

	d, err := Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	id, err := d.CreateSession(5)
	if err != nil {
		t.Fatalf("CreateSession: %v", err)
	}
	if err := d.AppendMessage(id, Message{Seq: 1, ChatMessage: llm.UserMessage("survive me")}); err != nil {
		t.Fatalf("AppendMessage: %v", err)
	}
	if err := d.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	// Reopen the same file: schema re-migrate must be a no-op and the data
	// must still be there.
	d2, err := Open(path)
	if err != nil {
		t.Fatalf("reopen: %v", err)
	}
	defer d2.Close()
	got, err := d2.LoadSession(id)
	if err != nil {
		t.Fatalf("LoadSession after reopen: %v", err)
	}
	if len(got.Messages) != 1 || got.Messages[0].Content != "survive me" {
		t.Errorf("messages after reopen = %+v, want the committed message", got.Messages)
	}
}
