package db

import (
	"database/sql"
	"path/filepath"
	"testing"

	"porter/internal/llm"
)

// searchIDs runs a full-text MATCH query and returns the seq of every hit, so
// a test can assert which stored messages the index finds.
func searchIDs(t *testing.T, d *DB, query string) []int {
	t.Helper()
	rows, err := d.db.Query(`
		SELECT m.seq FROM messages_fts
		JOIN messages m ON m.rowid = messages_fts.rowid
		WHERE messages_fts MATCH ?
		ORDER BY bm25(messages_fts)`, query)
	if err != nil {
		t.Fatalf("search %q: %v", query, err)
	}
	defer rows.Close()
	var out []int
	for rows.Next() {
		var seq uint64
		if err := rows.Scan(&seq); err != nil {
			t.Fatalf("scan hit: %v", err)
		}
		out = append(out, int(seq))
	}
	if err := rows.Err(); err != nil {
		t.Fatalf("iterate hits: %v", err)
	}
	return out
}

func appendText(t *testing.T, d *DB, sessionID int64, seq uint64, role, content string) {
	t.Helper()
	m := Message{Seq: seq, ChatMessage: llm.ChatMessage{Role: role, Content: content}}
	if err := d.AppendMessage(sessionID, m); err != nil {
		t.Fatalf("AppendMessage(seq %d): %v", seq, err)
	}
}

// TestMessagesFTSFollowsAppends covers the trigger half of the index: a message
// written through AppendMessage is searchable straight away, a removed one
// drops out, and a role filter still narrows the hits.
func TestMessagesFTSFollowsAppends(t *testing.T) {
	d := openTemp(t)
	id, err := d.CreateSession(1)
	if err != nil {
		t.Fatalf("CreateSession: %v", err)
	}
	appendText(t, d, id, 1, "user", "how do I reload Caddy?")
	appendText(t, d, id, 2, "assistant", "Run systemctl reload caddy.")
	appendText(t, d, id, 3, "tool", "caddy validate --config /etc/caddy/Caddyfile")

	if got := searchIDs(t, d, `"caddy"`); len(got) != 3 {
		t.Errorf("hits for caddy = %v, want all 3 messages", got)
	}
	// A phrase query keeps word order, so only the tool output matches.
	if got := searchIDs(t, d, `"caddy validate"`); len(got) != 1 || got[0] != 3 {
		t.Errorf("phrase hits = %v, want [3]", got)
	}
	// Prefix queries are what make a half-remembered word findable.
	if got := searchIDs(t, d, `"cadd"*`); len(got) != 3 {
		t.Errorf("prefix hits = %v, want all 3 messages", got)
	}

	var userHits int
	if err := d.db.QueryRow(`
		SELECT count(*) FROM messages_fts
		JOIN messages m ON m.rowid = messages_fts.rowid
		WHERE messages_fts MATCH '"caddy"' AND m.role = 'user'`).Scan(&userHits); err != nil {
		t.Fatalf("role-filtered search: %v", err)
	}
	if userHits != 1 {
		t.Errorf("role-filtered hits = %d, want 1", userHits)
	}

	if _, err := d.db.Exec("DELETE FROM messages WHERE session_id = ? AND seq = 3", id); err != nil {
		t.Fatalf("delete message: %v", err)
	}
	if got := searchIDs(t, d, `"validate"`); len(got) != 0 {
		t.Errorf("hits after delete = %v, want none", got)
	}
}

// TestMigrationV14IndexesExistingMessages covers the backfill: messages stored
// before the index existed are found once the migration runs, which is what a
// live database upgrading relies on.
func TestMigrationV14IndexesExistingMessages(t *testing.T) {
	path := filepath.Join(t.TempDir(), "porter.db")

	raw, err := sql.Open("sqlite", path)
	if err != nil {
		t.Fatalf("open raw v13 db: %v", err)
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
			cancelled    INTEGER NOT NULL DEFAULT 0,
			tool_output  TEXT    NOT NULL DEFAULT '',
			output_tokens INTEGER NOT NULL DEFAULT 0,
			images       TEXT    NOT NULL DEFAULT '',
			PRIMARY KEY (session_id, seq)
		);
		INSERT INTO sessions (created_at) VALUES (7);
		INSERT INTO messages (session_id, seq, role, content) VALUES (1, 1, 'user', 'what did we decide about the FTS index?');
		PRAGMA user_version=13;
	`)
	if err != nil {
		t.Fatalf("create v13 schema: %v", err)
	}
	if err := raw.Close(); err != nil {
		t.Fatalf("close raw db: %v", err)
	}

	d, err := Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	t.Cleanup(func() { d.Close() })

	var v int
	if err := d.db.QueryRow("PRAGMA user_version").Scan(&v); err != nil {
		t.Fatalf("read user_version: %v", err)
	}
	if v != schemaVersion {
		t.Errorf("user_version = %d, want %d", v, schemaVersion)
	}
	if got := searchIDs(t, d, `"index"`); len(got) != 1 || got[0] != 1 {
		t.Errorf("hits for existing message = %v, want [1]", got)
	}
}
