// Package db is porter's SQLite persistence layer. It owns the schema and the
// row-level CRUD for sessions and their committed message history; it knows
// nothing about the event bus, rendering, or the agent loop. The session store
// treats it as the source of truth: committed history is written here first
// (and fail-fast), and in-memory state is limited to transient or rebuildable
// caches.
package db

import (
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"

	"porter/internal/llm"

	_ "modernc.org/sqlite"
)

// ErrNotFound is returned when a session does not exist in the database.
var ErrNotFound = errors.New("db: session not found")

// schemaVersion is the current schema revision, tracked in PRAGMA user_version.
// Bump it and extend migrate when the schema changes.
const schemaVersion = 1

// DB wraps the SQLite database handle. It deliberately uses a single
// connection: pragmas like foreign_keys are per-connection, and a single
// writer serializes commits without extra locking.
type DB struct {
	db *sql.DB
}

// Open opens (creating if needed) the SQLite database at path, applies the
// connection pragmas and the current schema, and returns a ready handle.
func Open(path string) (*DB, error) {
	// The modernc.org/sqlite driver registers the "sqlite" name.
	d, err := sql.Open("sqlite", path)
	if err != nil {
		return nil, fmt.Errorf("open sqlite: %w", err)
	}
	d.SetMaxOpenConns(1)

	db := &DB{db: d}
	if err := db.configure(); err != nil {
		d.Close()
		return nil, err
	}
	if err := db.migrate(); err != nil {
		d.Close()
		return nil, err
	}
	return db, nil
}

// Close closes the database.
func (d *DB) Close() error { return d.db.Close() }

// configure applies the per-connection pragmas. journal_mode is queried (it
// returns the active mode as a row) while the other two are statements.
func (d *DB) configure() error {
	var mode string
	if err := d.db.QueryRow("PRAGMA journal_mode=WAL").Scan(&mode); err != nil {
		return fmt.Errorf("set journal_mode=WAL: %w", err)
	}
	if _, err := d.db.Exec("PRAGMA busy_timeout=5000"); err != nil {
		return fmt.Errorf("set busy_timeout: %w", err)
	}
	if _, err := d.db.Exec("PRAGMA foreign_keys=ON"); err != nil {
		return fmt.Errorf("set foreign_keys: %w", err)
	}
	return nil
}

// migrate brings an existing database up to the current schema version,
// applying each missing migration in order.
func (d *DB) migrate() error {
	var v int
	if err := d.db.QueryRow("PRAGMA user_version").Scan(&v); err != nil {
		return fmt.Errorf("read user_version: %w", err)
	}
	if v < 1 {
		const ddl = `
CREATE TABLE IF NOT EXISTS sessions (
	id         INTEGER PRIMARY KEY AUTOINCREMENT,
	created_at INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
	session_id   INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
	seq          INTEGER NOT NULL,
	role         TEXT    NOT NULL,
	content      TEXT    NOT NULL DEFAULT '',
	reasoning    TEXT    NOT NULL DEFAULT '',
	tool_call_id TEXT    NOT NULL DEFAULT '',
	tool_calls   TEXT    NOT NULL DEFAULT '[]',  -- JSON array of llm.ToolCall
	started_at   INTEGER NOT NULL DEFAULT 0,
	finished_at  INTEGER NOT NULL DEFAULT 0,
	PRIMARY KEY (session_id, seq)
);
CREATE INDEX IF NOT EXISTS idx_sessions_created ON sessions(created_at DESC, id DESC);
`
		if _, err := d.db.Exec(ddl); err != nil {
			return fmt.Errorf("apply schema v1: %w", err)
		}
		if _, err := d.db.Exec(fmt.Sprintf("PRAGMA user_version=%d", schemaVersion)); err != nil {
			return fmt.Errorf("set user_version: %w", err)
		}
	}
	return nil
}

// Message is one committed message as stored: its bus position (seq) plus the
// message payload. tool_calls is stored as JSON and round-tripped back into the
// embedded llm.ChatMessage on load.
type Message struct {
	Seq uint64
	llm.ChatMessage
}

// Session is the full persisted state of one session: its creation time and all
// committed messages in seq order. MaxSeq is the highest seq written, which is
// where a restarted session resumes its bus position.
type Session struct {
	ID        int64
	CreatedAt int64
	Messages  []Message
	MaxSeq    uint64
}

// Summary is one row of the session list, newest first. FirstUser is the raw
// content of the session's first user message (empty when the session has no
// messages yet); truncation to a single-line preview is the caller's job.
type Summary struct {
	ID        int64
	CreatedAt int64
	FirstUser string
}

// CreateSession inserts a new session and returns its database id (the numeric
// part of the "session_<id>" identifier the store exposes).
func (d *DB) CreateSession(createdAt int64) (int64, error) {
	res, err := d.db.Exec("INSERT INTO sessions (created_at) VALUES (?)", createdAt)
	if err != nil {
		return 0, fmt.Errorf("insert session: %w", err)
	}
	id, err := res.LastInsertId()
	if err != nil {
		return 0, fmt.Errorf("session id: %w", err)
	}
	return id, nil
}

// AppendMessage writes one committed message. seq must be the message's bus
// position, unique within the session.
func (d *DB) AppendMessage(sessionID int64, m Message) error {
	calls, err := json.Marshal(m.ToolCalls)
	if err != nil {
		return fmt.Errorf("marshal tool calls: %w", err)
	}
	_, err = d.db.Exec(
		`INSERT INTO messages (session_id, seq, role, content, reasoning, tool_call_id, tool_calls, started_at, finished_at)
		 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)`,
		sessionID, m.Seq, m.Role, m.Content, m.Reasoning, m.ToolCallID, string(calls), m.StartedAt, m.FinishedAt,
	)
	if err != nil {
		return fmt.Errorf("insert message: %w", err)
	}
	return nil
}

// LoadSession returns a session's full persisted state, or ErrNotFound if the
// session does not exist.
func (d *DB) LoadSession(id int64) (Session, error) {
	var s Session
	s.ID = id
	if err := d.db.QueryRow("SELECT created_at FROM sessions WHERE id = ?", id).Scan(&s.CreatedAt); err != nil {
		if errors.Is(err, sql.ErrNoRows) {
			return Session{}, ErrNotFound
		}
		return Session{}, fmt.Errorf("load session %d: %w", id, err)
	}
	rows, err := d.db.Query(
		`SELECT seq, role, content, reasoning, tool_call_id, tool_calls, started_at, finished_at
		 FROM messages WHERE session_id = ? ORDER BY seq ASC`, id)
	if err != nil {
		return Session{}, fmt.Errorf("load messages for %d: %w", id, err)
	}
	defer rows.Close()
	for rows.Next() {
		var m Message
		var calls string
		if err := rows.Scan(&m.Seq, &m.Role, &m.Content, &m.Reasoning, &m.ToolCallID, &calls, &m.StartedAt, &m.FinishedAt); err != nil {
			return Session{}, fmt.Errorf("scan message: %w", err)
		}
		if calls != "" {
			if err := json.Unmarshal([]byte(calls), &m.ToolCalls); err != nil {
				return Session{}, fmt.Errorf("decode tool calls for seq %d: %w", m.Seq, err)
			}
		}
		s.Messages = append(s.Messages, m)
		if m.Seq > s.MaxSeq {
			s.MaxSeq = m.Seq
		}
	}
	if err := rows.Err(); err != nil {
		return Session{}, fmt.Errorf("iterate messages: %w", err)
	}
	return s, nil
}

// ListSessions returns every session, newest first, with the raw content of each
// session's first user message (the sidebar preview source).
func (d *DB) ListSessions() ([]Summary, error) {
	rows, err := d.db.Query(`
		SELECT s.id, s.created_at, COALESCE((
			SELECT m.content FROM messages m
			WHERE m.session_id = s.id AND m.role = 'user'
			ORDER BY m.seq ASC LIMIT 1
		), '')
		FROM sessions s
		ORDER BY s.created_at DESC, s.id DESC`)
	if err != nil {
		return nil, fmt.Errorf("list sessions: %w", err)
	}
	defer rows.Close()
	var out []Summary
	for rows.Next() {
		var s Summary
		if err := rows.Scan(&s.ID, &s.CreatedAt, &s.FirstUser); err != nil {
			return nil, fmt.Errorf("scan session: %w", err)
		}
		out = append(out, s)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("iterate sessions: %w", err)
	}
	return out, nil
}
