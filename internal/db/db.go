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
const schemaVersion = 5

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
	if v < 2 {
		// v2: the `cancelled` flag on committed tool messages, so a run aborted
		// by the user survives a reload as "cancelled" rather than a plain exit.
		const ddl = "ALTER TABLE messages ADD COLUMN cancelled INTEGER NOT NULL DEFAULT 0"
		if _, err := d.db.Exec(ddl); err != nil {
			return fmt.Errorf("apply schema v2: %w", err)
		}
		if _, err := d.db.Exec("PRAGMA user_version=2"); err != nil {
			return fmt.Errorf("set user_version: %w", err)
		}
	}
	if v < 3 {
		// v3: per-request usage/error records (queries). One row per model
		// request — the origin of token usage and request failures. Turns are
		// derived (not stored) by grouping queries under the user message that
		// started them; a successful query carries its tokens, a failed one its
		// error. This is what lets a reload render the REPL-style "(N in, M out
		// tokens)" line and failed-turn errors from persisted data instead of
		// only from the live bus.
		const ddl = `
CREATE TABLE IF NOT EXISTS queries (
	session_id INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
	turn_seq   INTEGER NOT NULL,
	idx        INTEGER NOT NULL,
	input      INTEGER NOT NULL DEFAULT 0,
	output     INTEGER NOT NULL DEFAULT 0,
	error      TEXT    NOT NULL DEFAULT '',
	PRIMARY KEY (session_id, turn_seq, idx)
);
CREATE INDEX IF NOT EXISTS idx_queries_turn ON queries(session_id, turn_seq);
`
		if _, err := d.db.Exec(ddl); err != nil {
			return fmt.Errorf("apply schema v3: %w", err)
		}
		if _, err := d.db.Exec("PRAGMA user_version=3"); err != nil {
			return fmt.Errorf("set user_version: %w", err)
		}
	}
	if v < 4 {
		// v4: split input tokens into cached vs uncached. The provider reports
		// how many prompt tokens were served from cache
		// (prompt_tokens_details.cached_tokens); storing the split explicitly
		// (instead of a single input total) is what lets cost display price
		// cache hits differently from misses. Old rows are backfilled as
		// all-uncached — the split was not recorded before v4, so treating
		// their input as fresh is the honest default. The input column is
		// dropped so the split is the one source of truth (the total is
		// derived as cached + uncached, never stored redundantly).
		const ddl = `
ALTER TABLE queries ADD COLUMN cached_input INTEGER NOT NULL DEFAULT 0;
ALTER TABLE queries ADD COLUMN uncached_input INTEGER NOT NULL DEFAULT 0;
UPDATE queries SET uncached_input = input;
ALTER TABLE queries DROP COLUMN input;
`
		if _, err := d.db.Exec(ddl); err != nil {
			return fmt.Errorf("apply schema v4: %w", err)
		}
		if _, err := d.db.Exec("PRAGMA user_version=4"); err != nil {
			return fmt.Errorf("set user_version: %w", err)
		}
	}
	if v < 5 {
		// v5: the `stopped` flag on per-request query records, so a turn the
		// user aborted with the Stop button survives a reload as "stopped"
		// (rather than a failed turn or a normal completion). Like the v2
		// `cancelled` flag on tool messages, it is set on the query that marks
		// the stop so the derived turn can render its footer from persisted
		// data instead of only from the live bus.
		const ddl = "ALTER TABLE queries ADD COLUMN stopped INTEGER NOT NULL DEFAULT 0"
		if _, err := d.db.Exec(ddl); err != nil {
			return fmt.Errorf("apply schema v5: %w", err)
		}
		if _, err := d.db.Exec("PRAGMA user_version=5"); err != nil {
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

// Query is one model request as stored: its token usage and, when the request
// failed, the error. TurnSeq is the bus position of the user message that
// started the turn the request belongs to; Idx is the request's zero-based
// position within that turn (the agent runs requests sequentially). A failed
// request carries no message of its own, so queries are keyed by
// (turn, idx) rather than by a message.
type Query struct {
	TurnSeq       uint64
	Idx           int
	CachedInput   int
	UncachedInput int
	Output        int
	Error         string
	// Stopped reports that the turn was aborted by the user (the Stop button)
	// rather than completing or failing. It is set on the query that marks the
	// stop (the streaming request, with its partial usage, or a request that
	// never ran), so a reload can render the turn's stopped footer.
	Stopped bool
}

// Session is the full persisted state of one session: its creation time, all
// committed messages in seq order, and every per-request usage/error record.
// MaxSeq is the highest seq written, which is where a restarted session resumes
// its bus position.
type Session struct {
	ID        int64
	CreatedAt int64
	Messages  []Message
	Queries   []Query
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

// boolInt converts a Go bool to the 0/1 integer SQLite stores booleans as.
func boolInt(b bool) int {
	if b {
		return 1
	}
	return 0
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
		`INSERT INTO messages (session_id, seq, role, content, reasoning, tool_call_id, tool_calls, started_at, finished_at, cancelled)
		 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
		sessionID, m.Seq, m.Role, m.Content, m.Reasoning, m.ToolCallID, string(calls), m.StartedAt, m.FinishedAt, boolInt(m.Cancelled),
	)
	if err != nil {
		return fmt.Errorf("insert message: %w", err)
	}
	return nil
}

// AppendQuery writes one per-request usage/error record. It is independent of
// the message stream: a successful request also produced an assistant message
// (written separately), while a failed request produced none.
func (d *DB) AppendQuery(sessionID int64, q Query) error {
	_, err := d.db.Exec(
		`INSERT INTO queries (session_id, turn_seq, idx, cached_input, uncached_input, output, error, stopped)
		 VALUES (?, ?, ?, ?, ?, ?, ?, ?)`,
		sessionID, q.TurnSeq, q.Idx, q.CachedInput, q.UncachedInput, q.Output, q.Error, q.Stopped,
	)
	if err != nil {
		return fmt.Errorf("insert query: %w", err)
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
		`SELECT seq, role, content, reasoning, tool_call_id, tool_calls, started_at, finished_at, cancelled
		 FROM messages WHERE session_id = ? ORDER BY seq ASC`, id)
	if err != nil {
		return Session{}, fmt.Errorf("load messages for %d: %w", id, err)
	}
	for rows.Next() {
		var m Message
		var calls string
		var cancelled int
		if err := rows.Scan(&m.Seq, &m.Role, &m.Content, &m.Reasoning, &m.ToolCallID, &calls, &m.StartedAt, &m.FinishedAt, &cancelled); err != nil {
			rows.Close()
			return Session{}, fmt.Errorf("scan message: %w", err)
		}
		m.Cancelled = cancelled != 0
		if calls != "" {
			if err := json.Unmarshal([]byte(calls), &m.ToolCalls); err != nil {
				rows.Close()
				return Session{}, fmt.Errorf("decode tool calls for seq %d: %w", m.Seq, err)
			}
		}
		s.Messages = append(s.Messages, m)
		if m.Seq > s.MaxSeq {
			s.MaxSeq = m.Seq
		}
	}
	if err := rows.Err(); err != nil {
		rows.Close()
		return Session{}, fmt.Errorf("iterate messages: %w", err)
	}
	// The database uses a single connection, so close the messages result
	// before issuing the queries query (a second query cannot start while a
	// result set is still open on the one connection).
	rows.Close()
	qrows, err := d.db.Query(
		`SELECT turn_seq, idx, cached_input, uncached_input, output, error, stopped
		 FROM queries WHERE session_id = ? ORDER BY turn_seq ASC, idx ASC`, id)
	if err != nil {
		return Session{}, fmt.Errorf("load queries for %d: %w", id, err)
	}
	defer qrows.Close()
	for qrows.Next() {
		var q Query
		var stopped int
		if err := qrows.Scan(&q.TurnSeq, &q.Idx, &q.CachedInput, &q.UncachedInput, &q.Output, &q.Error, &stopped); err != nil {
			return Session{}, fmt.Errorf("scan query: %w", err)
		}
		q.Stopped = stopped != 0
		s.Queries = append(s.Queries, q)
	}
	if err := qrows.Err(); err != nil {
		return Session{}, fmt.Errorf("iterate queries: %w", err)
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
