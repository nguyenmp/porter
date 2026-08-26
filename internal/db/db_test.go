package db

import (
	"database/sql"
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
		{
			// A tool run the user aborted: the cancelled flag must survive the
			// round-trip so a reload renders it as cancelled.
			Seq:         4,
			ChatMessage: llm.ChatMessage{Role: "tool", ToolCallID: "call-2", Content: "partial", StartedAt: 70, FinishedAt: 80, Cancelled: true},
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
		if g.ToolCallID != w.ToolCallID || g.StartedAt != w.StartedAt || g.FinishedAt != w.FinishedAt || g.Cancelled != w.Cancelled {
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
	if got.MaxSeq != 4 {
		t.Errorf("max seq = %d, want 4", got.MaxSeq)
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

// TestMigratesFromV1 verifies an existing v1 database (schema without the
// `cancelled` column) is migrated to v2 on open: the column is added, existing
// rows survive, and their cancelled flag reads false.
func TestMigratesFromV1(t *testing.T) {
	path := filepath.Join(t.TempDir(), "porter.db")

	// Build a v1 database by hand, exactly as schema v1 defined it.
	raw, err := sql.Open("sqlite", path)
	if err != nil {
		t.Fatalf("open raw v1 db: %v", err)
	}
	_, err = raw.Exec(`
		CREATE TABLE sessions (
			id         INTEGER PRIMARY KEY AUTOINCREMENT,
			created_at INTEGER NOT NULL
		);
		CREATE TABLE messages (
			session_id   INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
			seq          INTEGER NOT NULL,
			role         TEXT    NOT NULL,
			content      TEXT    NOT NULL DEFAULT '',
			reasoning    TEXT    NOT NULL DEFAULT '',
			tool_call_id TEXT    NOT NULL DEFAULT '',
			tool_calls   TEXT    NOT NULL DEFAULT '[]',
			started_at   INTEGER NOT NULL DEFAULT 0,
			finished_at  INTEGER NOT NULL DEFAULT 0,
			PRIMARY KEY (session_id, seq)
		);
		PRAGMA user_version=1;
	`)
	if err != nil {
		t.Fatalf("create v1 schema: %v", err)
	}
	res, err := raw.Exec(`INSERT INTO sessions (created_at) VALUES (7)`)
	if err != nil {
		t.Fatalf("insert v1 session: %v", err)
	}
	id, _ := res.LastInsertId()
	if _, err := raw.Exec(
		`INSERT INTO messages (session_id, seq, role, content) VALUES (?, 1, 'user', 'old row')`, id); err != nil {
		t.Fatalf("insert v1 message: %v", err)
	}
	if err := raw.Close(); err != nil {
		t.Fatalf("close raw db: %v", err)
	}

	// Opening runs the v1->v2 migration.
	d, err := Open(path)
	if err != nil {
		t.Fatalf("Open after v1: %v", err)
	}
	defer d.Close()

	got, err := d.LoadSession(id)
	if err != nil {
		t.Fatalf("LoadSession after migration: %v", err)
	}
	if len(got.Messages) != 1 || got.Messages[0].Content != "old row" {
		t.Fatalf("messages after migration = %+v, want the v1 row intact", got.Messages)
	}
	if got.Messages[0].Cancelled {
		t.Errorf("v1 row should load with cancelled=false")
	}
	// New writes use the column too.
	if err := d.AppendMessage(id, Message{Seq: 2, ChatMessage: llm.ChatMessage{Role: "tool", Content: "x", Cancelled: true}}); err != nil {
		t.Fatalf("AppendMessage after migration: %v", err)
	}
	got, err = d.LoadSession(id)
	if err != nil {
		t.Fatalf("LoadSession after write: %v", err)
	}
	if !got.Messages[1].Cancelled {
		t.Errorf("cancelled flag did not round-trip after migration")
	}
}
