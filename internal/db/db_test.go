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
func TestQueryRoundTrip(t *testing.T) {
	d := openTemp(t)
	id, err := d.CreateSession(100)
	if err != nil {
		t.Fatalf("CreateSession: %v", err)
	}
	want := []Query{
		{TurnSeq: 1, Idx: 0, UncachedInput: 2, Output: 3},
		// A provider-reported cache split round-trips: 1 cached + 3 miss in.
		{TurnSeq: 1, Idx: 1, CachedInput: 1, UncachedInput: 3, Output: 5},
		{TurnSeq: 8, Idx: 0, CachedInput: 0, UncachedInput: 0, Output: 0, Error: "boom"},
		// A turn the user stopped round-trips its stopped flag (with the
		// partial usage recorded at the stop).
		{TurnSeq: 9, Idx: 0, CachedInput: 1, UncachedInput: 2, Output: 3, Stopped: true},
	}
	for _, q := range want {
		if err := d.AppendQuery(id, q); err != nil {
			t.Fatalf("AppendQuery(%+v): %v", q, err)
		}
	}
	got, err := d.LoadSession(id)
	if err != nil {
		t.Fatalf("LoadSession: %v", err)
	}
	if len(got.Queries) != len(want) {
		t.Fatalf("queries = %+v, want %d rows", got.Queries, len(want))
	}
	for i, w := range want {
		if got.Queries[i] != w {
			t.Errorf("queries[%d] = %+v, want %+v", i, got.Queries[i], w)
		}
	}
}

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
	// The v3 migration added the queries table; it must exist and be writable.
	if err := d.AppendQuery(id, Query{TurnSeq: 1, Idx: 0, UncachedInput: 1, Output: 1}); err != nil {
		t.Fatalf("AppendQuery after v1->v3 migration: %v", err)
	}
	got, err = d.LoadSession(id)
	if err != nil {
		t.Fatalf("LoadSession after query write: %v", err)
	}
	if len(got.Queries) != 1 || got.Queries[0].TurnSeq != 1 {
		t.Errorf("queries after migration = %+v, want one row for turn 1", got.Queries)
	}
}

// TestMigratesFromV3ToV4 verifies the v3->v4 upgrade path, which is the real
// one for existing databases: the queries table gains cached_input/uncached_input,
// existing `input` totals are backfilled as all-uncached (the split was not
// recorded before v4), and the now-redundant `input` column is dropped.
func TestMigratesFromV3ToV4(t *testing.T) {
	path := filepath.Join(t.TempDir(), "porter.db")

	// Build a v3 database by hand, exactly as schema v3 defined it.
	raw, err := sql.Open("sqlite", path)
	if err != nil {
		t.Fatalf("open raw v3 db: %v", err)
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
			PRIMARY KEY (session_id, seq)
		);
		CREATE TABLE queries (
			session_id INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
			turn_seq   INTEGER NOT NULL,
			idx        INTEGER NOT NULL,
			input      INTEGER NOT NULL DEFAULT 0,
			output     INTEGER NOT NULL DEFAULT 0,
			error      TEXT    NOT NULL DEFAULT '',
			PRIMARY KEY (session_id, turn_seq, idx)
		);
		PRAGMA user_version=3;
	`)
	if err != nil {
		t.Fatalf("create v3 schema: %v", err)
	}
	res, err := raw.Exec(`INSERT INTO sessions (created_at) VALUES (7)`)
	if err != nil {
		t.Fatalf("insert v3 session: %v", err)
	}
	id, _ := res.LastInsertId()
	if _, err := raw.Exec(
		`INSERT INTO queries (session_id, turn_seq, idx, input, output) VALUES (?, 1, 0, 7, 8)`, id); err != nil {
		t.Fatalf("insert v3 query: %v", err)
	}
	if err := raw.Close(); err != nil {
		t.Fatalf("close raw db: %v", err)
	}

	// Opening runs the v3->v4 migration.
	d, err := Open(path)
	if err != nil {
		t.Fatalf("Open after v3: %v", err)
	}
	defer d.Close()

	// The old total is backfilled as all-uncached: cached_input=0,
	// uncached_input=7, output=8.
	got, err := d.LoadSession(id)
	if err != nil {
		t.Fatalf("LoadSession after migration: %v", err)
	}
	if len(got.Queries) != 1 {
		t.Fatalf("queries after migration = %+v, want 1 row", got.Queries)
	}
	if q := got.Queries[0]; q.CachedInput != 0 || q.UncachedInput != 7 || q.Output != 8 || q.Error != "" {
		t.Errorf("query after migration = %+v, want cached 0, uncached 7, output 8", q)
	}

	// The input column is gone: a write must use the split, and the split
	// round-trips. (Probing the dropped column directly also confirms DROP
	// COLUMN ran, not just that the backfill is invisible to LoadSession.)
	var cols []string
	rows, err := d.db.Query(`SELECT name FROM pragma_table_info('queries')`)
	if err != nil {
		t.Fatalf("list query columns: %v", err)
	}
	for rows.Next() {
		var c string
		if err := rows.Scan(&c); err != nil {
			rows.Close()
			t.Fatalf("scan column: %v", err)
		}
		cols = append(cols, c)
	}
	rows.Close()
	for _, banned := range []string{"input"} {
		for _, c := range cols {
			if c == banned {
				t.Errorf("queries still has column %q after v4 migration: %v", banned, cols)
			}
		}
	}
	if err := d.AppendQuery(id, Query{TurnSeq: 1, Idx: 1, CachedInput: 2, UncachedInput: 3, Output: 4}); err != nil {
		t.Fatalf("AppendQuery after migration: %v", err)
	}
	got, err = d.LoadSession(id)
	if err != nil {
		t.Fatalf("LoadSession after write: %v", err)
	}
	if len(got.Queries) != 2 {
		t.Fatalf("queries after write = %+v, want 2 rows", got.Queries)
	}
}

// TestMigratesFromV4ToV5 verifies the v4->v5 migration: opening a v4 database
// adds the `stopped` column to queries (defaulting to 0, so a pre-existing
// query reads as not stopped), and a stopped query written after the migration
// round-trips.
func TestMigratesFromV4ToV5(t *testing.T) {
	path := filepath.Join(t.TempDir(), "porter.db")

	// Build a v4 database by hand, exactly as schema v4 defined it (v1
	// sessions/messages plus v3 queries with the v4 cached/uncached split).
	raw, err := sql.Open("sqlite", path)
	if err != nil {
		t.Fatalf("open raw v4 db: %v", err)
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
			PRIMARY KEY (session_id, turn_seq, idx)
		);
		PRAGMA user_version=4;
	`)
	if err != nil {
		t.Fatalf("create v4 schema: %v", err)
	}
	res, err := raw.Exec(`INSERT INTO sessions (created_at) VALUES (7)`)
	if err != nil {
		t.Fatalf("insert v4 session: %v", err)
	}
	id, _ := res.LastInsertId()
	if _, err := raw.Exec(
		`INSERT INTO queries (session_id, turn_seq, idx, cached_input, uncached_input, output) VALUES (?, 1, 0, 1, 2, 3)`, id); err != nil {
		t.Fatalf("insert v4 query: %v", err)
	}
	if err := raw.Close(); err != nil {
		t.Fatalf("close raw db: %v", err)
	}

	// Opening runs the v4->v5 migration.
	d, err := Open(path)
	if err != nil {
		t.Fatalf("Open after v4: %v", err)
	}
	defer d.Close()

	// The pre-existing query reads back as not stopped (the column defaults 0).
	got, err := d.LoadSession(id)
	if err != nil {
		t.Fatalf("LoadSession after migration: %v", err)
	}
	if len(got.Queries) != 1 {
		t.Fatalf("queries after migration = %+v, want 1 row", got.Queries)
	}
	if q := got.Queries[0]; q.Stopped {
		t.Errorf("pre-existing query marked stopped after migration: %+v", q)
	}

	// A stopped query written after the migration round-trips.
	if err := d.AppendQuery(id, Query{TurnSeq: 1, Idx: 1, CachedInput: 4, UncachedInput: 5, Output: 6, Stopped: true}); err != nil {
		t.Fatalf("AppendQuery stopped after migration: %v", err)
	}
	got, err = d.LoadSession(id)
	if err != nil {
		t.Fatalf("LoadSession after write: %v", err)
	}
	if len(got.Queries) != 2 {
		t.Fatalf("queries after write = %+v, want 2 rows", got.Queries)
	}
	if !got.Queries[1].Stopped {
		t.Errorf("stopped query did not round-trip: %+v", got.Queries[1])
	}
}

// TestToolOutputRoundTrip verifies the structured tool-output metadata survives
// a write/load round-trip (and that a message without it comes back nil).
func TestToolOutputRoundTrip(t *testing.T) {
	d := openTemp(t)
	id, err := d.CreateSession(100)
	if err != nil {
		t.Fatalf("CreateSession: %v", err)
	}

	meta := &llm.ToolOutputMeta{Truncated: true, TotalBytes: 2550, ShownBytes: 1536}
	want := []Message{
		{
			Seq:         1,
			ChatMessage: llm.ChatMessage{Role: "tool", ToolCallID: "call-1", Content: "big output", ToolOutput: meta},
		},
		{
			Seq:         2,
			ChatMessage: llm.ChatMessage{Role: "tool", ToolCallID: "call-2", Content: "small output"},
		},
		{
			Seq: 3,
			ChatMessage: llm.ChatMessage{Role: "tool", ToolCallID: "call-3", Content: "recall",
				ToolOutput: &llm.ToolOutputMeta{Recall: true, SourceCallID: "call-1", Offset: 1024, MaxBytes: 1000, TotalBytes: 2550, ShownBytes: 1000}},
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
	for i, w := range want {
		g := got.Messages[i]
		if g.ToolOutput == nil && w.ToolOutput != nil {
			t.Errorf("message %d: expected ToolOutput metadata, got nil", i)
			continue
		}
		if g.ToolOutput != nil && w.ToolOutput == nil {
			t.Errorf("message %d: got ToolOutput %+v, want nil", i, g.ToolOutput)
			continue
		}
		if g.ToolOutput != nil && *g.ToolOutput != *w.ToolOutput {
			t.Errorf("message %d: ToolOutput = %+v, want %+v", i, g.ToolOutput, w.ToolOutput)
		}
	}
}

// TestMigratesFromV5ToV6 verifies the v5->v6 upgrade path: the messages table
// gains the tool_output column, pre-existing rows load with nil metadata, and
// new writes round-trip the metadata.
func TestMigratesFromV5ToV6(t *testing.T) {
	path := filepath.Join(t.TempDir(), "porter.db")

	raw, err := sql.Open("sqlite", path)
	if err != nil {
		t.Fatalf("open raw v5 db: %v", err)
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
		PRAGMA user_version=5;
	`)
	if err != nil {
		t.Fatalf("create v5 schema: %v", err)
	}
	res, err := raw.Exec(`INSERT INTO sessions (created_at) VALUES (7)`)
	if err != nil {
		t.Fatalf("insert v5 session: %v", err)
	}
	id, _ := res.LastInsertId()
	if _, err := raw.Exec(
		`INSERT INTO messages (session_id, seq, role, content, tool_call_id) VALUES (?, 1, 'tool', 'old row', 'call-0')`, id); err != nil {
		t.Fatalf("insert v5 message: %v", err)
	}
	if err := raw.Close(); err != nil {
		t.Fatalf("close raw db: %v", err)
	}

	d, err := Open(path)
	if err != nil {
		t.Fatalf("Open after v5: %v", err)
	}
	defer d.Close()

	// Pre-existing rows load with nil metadata (the column defaults empty).
	got, err := d.LoadSession(id)
	if err != nil {
		t.Fatalf("LoadSession after migration: %v", err)
	}
	if len(got.Messages) != 1 || got.Messages[0].ToolOutput != nil {
		t.Fatalf("v5 row after migration = %+v, want intact with nil metadata", got.Messages)
	}

	// New writes round-trip the metadata.
	meta := &llm.ToolOutputMeta{Truncated: true, TotalBytes: 2550, ShownBytes: 1536}
	if err := d.AppendMessage(id, Message{Seq: 2, ChatMessage: llm.ChatMessage{Role: "tool", ToolCallID: "call-1", Content: "x", ToolOutput: meta}}); err != nil {
		t.Fatalf("AppendMessage after migration: %v", err)
	}
	got, err = d.LoadSession(id)
	if err != nil {
		t.Fatalf("LoadSession after write: %v", err)
	}
	if len(got.Messages) != 2 || got.Messages[1].ToolOutput == nil || *got.Messages[1].ToolOutput != *meta {
		t.Errorf("tool_output metadata did not round-trip after migration: %+v", got.Messages[1].ToolOutput)
	}
}

func TestArchiveUnarchiveSession(t *testing.T) {
	d := openTemp(t)

	// Three sessions at increasing created_at, so active ordering is defined.
	a, err := d.CreateSession(10)
	if err != nil {
		t.Fatalf("CreateSession: %v", err)
	}
	b, err := d.CreateSession(20)
	if err != nil {
		t.Fatalf("CreateSession: %v", err)
	}
	c, err := d.CreateSession(30)
	if err != nil {
		t.Fatalf("CreateSession: %v", err)
	}

	// New sessions start active.
	list, err := d.ListSessions()
	if err != nil {
		t.Fatalf("ListSessions: %v", err)
	}
	for _, s := range list {
		if s.ArchivedAt != 0 {
			t.Errorf("fresh session %d archived_at = %d, want 0", s.ID, s.ArchivedAt)
		}
	}

	// Archive b at 50 and c at 40 (so b was archived more recently). The list
	// must fold them below the active session, ordered by archived_at DESC.
	if err := d.ArchiveSession(b, 50); err != nil {
		t.Fatalf("ArchiveSession b: %v", err)
	}
	if err := d.ArchiveSession(c, 40); err != nil {
		t.Fatalf("ArchiveSession c: %v", err)
	}
	list, err = d.ListSessions()
	if err != nil {
		t.Fatalf("ListSessions: %v", err)
	}
	var got []int64
	for _, s := range list {
		got = append(got, s.ID)
	}
	// Active first (a, the only unarchived one), then archived by stamp DESC.
	want := []int64{a, b, c}
	if len(got) != len(want) {
		t.Fatalf("order = %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("order = %v, want %v", got, want)
			break
		}
	}
	if list[1].ArchivedAt != 50 || list[2].ArchivedAt != 40 {
		t.Errorf("archived stamps = %d, %d; want 50, 40", list[1].ArchivedAt, list[2].ArchivedAt)
	}

	// LoadSession carries the flag.
	ps, err := d.LoadSession(b)
	if err != nil {
		t.Fatalf("LoadSession: %v", err)
	}
	if ps.ArchivedAt != 50 {
		t.Errorf("LoadSession archived_at = %d, want 50", ps.ArchivedAt)
	}

	// Unarchive c: it returns to the active list in created_at order.
	if err := d.UnarchiveSession(c); err != nil {
		t.Fatalf("UnarchiveSession: %v", err)
	}
	list, err = d.ListSessions()
	if err != nil {
		t.Fatalf("ListSessions: %v", err)
	}
	got = got[:0]
	for _, s := range list {
		got = append(got, s.ID)
	}
	if want := []int64{c, a, b}; len(got) != len(want) || got[0] != want[0] || got[1] != want[1] || got[2] != want[2] {
		t.Errorf("after unarchive order = %v, want %v", got, want)
	}

	// Idempotent: re-archiving an archived session just re-stamps it.
	if err := d.ArchiveSession(b, 60); err != nil {
		t.Fatalf("re-ArchiveSession: %v", err)
	}
	ps, err = d.LoadSession(b)
	if err != nil {
		t.Fatalf("LoadSession: %v", err)
	}
	if ps.ArchivedAt != 60 {
		t.Errorf("re-archive stamp = %d, want 60", ps.ArchivedAt)
	}
	// Unarchiving an active session is a no-op success.
	if err := d.UnarchiveSession(a); err != nil {
		t.Fatalf("UnarchiveSession active: %v", err)
	}
}

func TestMigratesFromV6ToV7(t *testing.T) {
	path := filepath.Join(t.TempDir(), "porter.db")

	raw, err := sql.Open("sqlite", path)
	if err != nil {
		t.Fatalf("open raw v6 db: %v", err)
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
		PRAGMA user_version=6;
	`)
	if err != nil {
		t.Fatalf("create v6 schema: %v", err)
	}
	res, err := raw.Exec(`INSERT INTO sessions (created_at) VALUES (7)`)
	if err != nil {
		t.Fatalf("insert v6 session: %v", err)
	}
	id, _ := res.LastInsertId()
	if err := raw.Close(); err != nil {
		t.Fatalf("close raw db: %v", err)
	}

	d, err := Open(path)
	if err != nil {
		t.Fatalf("Open after v6: %v", err)
	}
	defer d.Close()

	// Pre-existing rows migrate as active.
	got, err := d.LoadSession(id)
	if err != nil {
		t.Fatalf("LoadSession after migration: %v", err)
	}
	if got.ArchivedAt != 0 {
		t.Errorf("v6 row after migration archived_at = %d, want 0", got.ArchivedAt)
	}

	// Archive/unarchive work on the migrated schema.
	if err := d.ArchiveSession(id, 99); err != nil {
		t.Fatalf("ArchiveSession after migration: %v", err)
	}
	got, err = d.LoadSession(id)
	if err != nil {
		t.Fatalf("LoadSession after archive: %v", err)
	}
	if got.ArchivedAt != 99 {
		t.Errorf("archived_at after archive = %d, want 99", got.ArchivedAt)
	}
	list, err := d.ListSessions()
	if err != nil {
		t.Fatalf("ListSessions: %v", err)
	}
	if len(list) != 1 || list[0].ArchivedAt != 99 {
		t.Errorf("list after archive = %+v, want one archived row", list)
	}
}
