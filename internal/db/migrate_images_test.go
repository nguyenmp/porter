package db

import (
	"database/sql"
	"path/filepath"
	"testing"

	"porter/internal/llm"
)

// TestMigratesFromV12ToV13 checks that a database written before images
// existed picks up the new column, that old rows still load, and that a message
// with images then round-trips. A v12 database has every other table already,
// so this migration runs on its own.
func TestMigratesFromV12ToV13(t *testing.T) {
	path := filepath.Join(t.TempDir(), "porter.db")

	raw, err := sql.Open("sqlite", path)
	if err != nil {
		t.Fatalf("open raw v12 db: %v", err)
	}
	_, err = raw.Exec(`
		CREATE TABLE sessions (
			id         INTEGER PRIMARY KEY AUTOINCREMENT,
			created_at INTEGER NOT NULL,
			archived_at_epoch_ms INTEGER NOT NULL DEFAULT 0,
			name       TEXT    NOT NULL DEFAULT '',
			local_sandbox INTEGER NOT NULL DEFAULT 0
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
			cancelled    INTEGER NOT NULL DEFAULT 0,
			tool_output  TEXT    NOT NULL DEFAULT '',
			output_tokens INTEGER NOT NULL DEFAULT 0,
			PRIMARY KEY (session_id, seq)
		);
		CREATE TABLE queries (
			session_id   INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
			turn_seq     INTEGER NOT NULL,
			idx          INTEGER NOT NULL,
			cached_input INTEGER NOT NULL DEFAULT 0,
			uncached_input INTEGER NOT NULL DEFAULT 0,
			output       INTEGER NOT NULL DEFAULT 0,
			error        TEXT    NOT NULL DEFAULT '',
			stopped      INTEGER NOT NULL DEFAULT 0,
			PRIMARY KEY (session_id, turn_seq, idx)
		);
		CREATE TABLE variants (
			session_id     INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
			message_seq    INTEGER NOT NULL,
			idx            INTEGER NOT NULL,
			source         INTEGER NOT NULL DEFAULT -1,
			content        TEXT    NOT NULL DEFAULT '',
			error          TEXT    NOT NULL DEFAULT '',
			prompt_version TEXT    NOT NULL DEFAULT '',
			created_at     INTEGER NOT NULL DEFAULT 0,
			PRIMARY KEY (session_id, message_seq, idx)
		);
		PRAGMA user_version=12;
	`)
	if err != nil {
		t.Fatalf("create v12 schema: %v", err)
	}
	res, err := raw.Exec(`INSERT INTO sessions (created_at) VALUES (12)`)
	if err != nil {
		t.Fatalf("insert v12 session: %v", err)
	}
	id, _ := res.LastInsertId()
	if _, err := raw.Exec(
		`INSERT INTO messages (session_id, seq, role, content) VALUES (?, 1, 'user', 'old message')`, id,
	); err != nil {
		t.Fatalf("insert v12 message: %v", err)
	}
	if err := raw.Close(); err != nil {
		t.Fatalf("close raw db: %v", err)
	}

	d, err := Open(path)
	if err != nil {
		t.Fatalf("Open (migrate): %v", err)
	}
	defer d.Close()

	// The pre-migration row loads, with no images invented for it.
	got, err := d.LoadSession(id)
	if err != nil {
		t.Fatalf("LoadSession after migrate: %v", err)
	}
	if len(got.Messages) != 1 || got.Messages[0].Content != "old message" {
		t.Fatalf("old row = %+v, want the original message", got.Messages)
	}
	if len(got.Messages[0].Images) != 0 {
		t.Errorf("old row images = %+v, want none", got.Messages[0].Images)
	}

	// And the new column works on the migrated database.
	msg := llm.UserMessage("with a picture")
	msg.Images = []llm.ImagePart{{Name: "shot.png", URL: "data:image/png;base64,AAAA"}}
	if err := d.AppendMessage(id, Message{Seq: 2, ChatMessage: msg}); err != nil {
		t.Fatalf("AppendMessage after migrate: %v", err)
	}
	got, err = d.LoadSession(id)
	if err != nil {
		t.Fatalf("LoadSession: %v", err)
	}
	last := got.Messages[len(got.Messages)-1]
	if len(last.Images) != 1 || last.Images[0].URL != "data:image/png;base64,AAAA" {
		t.Errorf("images after migrate = %+v, want the attached image", last.Images)
	}
}
