package session

import (
	"context"
	"io"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"

	"porter/internal/db"
)

// newSandboxedStore returns a store whose server-local sandboxes live in a
// fresh temp dir, backed by a file database in the same temp dir (so a
// restart can reopen it).
func newSandboxedStore(t *testing.T) (*Store, string) {
	t.Helper()
	dir := t.TempDir()
	d, err := db.Open(filepath.Join(dir, "porter.db"))
	if err != nil {
		t.Fatalf("open db: %v", err)
	}
	st := NewStore(d, nil)
	st.SetSandboxRoot(filepath.Join(dir, ".porter", "sandboxes"))
	t.Cleanup(func() { st.Close() })
	return st, dir
}

// runLocalShell runs a shell command through the session's active provider
// and returns its combined output.
func runLocalShell(t *testing.T, s *Session, command string) string {
	t.Helper()
	out, err := s.provider().Run(context.Background(), "shell", []byte(`{"command":"`+command+`"}`))
	if err != nil {
		t.Fatalf("run shell %q: %v", command, err)
	}
	b, err := io.ReadAll(out)
	if err != nil {
		t.Fatalf("read shell output: %v", err)
	}
	return string(b)
}

// TestCreateLocalSandbox makes a session and checks its local provider works
// in a fresh per-chat folder, never the server's working directory.
func TestCreateLocalSandbox(t *testing.T) {
	st, _ := newSandboxedStore(t)
	s, err := st.Create(nil)
	if err != nil {
		t.Fatalf("Create: %v", err)
	}
	dir := filepath.Join(st.sandboxRoot, s.ID())
	if s.localDir != dir {
		t.Errorf("localDir = %q, want %q", s.localDir, dir)
	}
	if fi, err := os.Stat(dir); err != nil || !fi.IsDir() {
		t.Fatalf("sandbox folder %s missing: %v", dir, err)
	}
	if reason := s.PausedReason(); reason != "" {
		t.Errorf("fresh sandboxed chat paused: %s", reason)
	}
	out := runLocalShell(t, s, "pwd")
	if !strings.Contains(out, dir) {
		t.Errorf("shell pwd = %q, want it to contain the sandbox folder %q", out, dir)
	}
	// The flag is persisted, so a restart can tell this chat from a
	// pre-sandboxing chat.
	sums, err := st.persist.ListSessions()
	if err != nil {
		t.Fatalf("ListSessions: %v", err)
	}
	if len(sums) != 1 || !sums[0].LocalSandbox {
		t.Errorf("persisted summaries = %+v, want one with LocalSandbox set", sums)
	}
}

// TestLocalSandboxMissingFolderPauses checks host-aligned behavior: a
// sandboxed chat whose folder is gone refuses to run (paused) rather than
// silently executing in the server's working directory, and resumes once the
// folder is recreated.
func TestLocalSandboxMissingFolderPauses(t *testing.T) {
	st, _ := newSandboxedStore(t)
	s, err := st.Create(nil)
	if err != nil {
		t.Fatalf("Create: %v", err)
	}
	if err := os.RemoveAll(s.localDir); err != nil {
		t.Fatalf("remove sandbox folder: %v", err)
	}
	if reason := s.PausedReason(); reason == "" || !strings.Contains(reason, s.localDir) {
		t.Errorf("PausedReason after folder removal = %q, want a reason naming %q", s.PausedReason(), s.localDir)
	}
	_, err = s.provider().Run(context.Background(), "shell", []byte(`{"command":"pwd"}`))
	if err == nil || !strings.Contains(err.Error(), "paused") {
		t.Errorf("provider run with missing folder err = %v, want a paused error", err)
	}
	// Recreate the folder: the chat resumes without a restart.
	if err := os.MkdirAll(s.localDir, 0o755); err != nil {
		t.Fatalf("recreate sandbox folder: %v", err)
	}
	if reason := s.PausedReason(); reason != "" {
		t.Errorf("PausedReason after folder recreate = %q, want none", reason)
	}
	out := runLocalShell(t, s, "pwd")
	if !strings.Contains(out, s.localDir) {
		t.Errorf("shell pwd after recreate = %q, want %q", out, s.localDir)
	}
}

// TestLocalSandboxArchiveReleasesAndUnarchiveRestores checks the lifecycle:
// archiving removes the folder, unarchiving recreates it, and the chat never
// falls back to the server's working directory.
func TestLocalSandboxArchiveReleasesAndUnarchiveRestores(t *testing.T) {
	st, _ := newSandboxedStore(t)
	s, err := st.Create(nil)
	if err != nil {
		t.Fatalf("Create: %v", err)
	}
	if err := st.Archive(s.ID()); err != nil {
		t.Fatalf("Archive: %v", err)
	}
	st.ReleaseSession(s.ID())
	if _, err := os.Stat(s.localDir); !os.IsNotExist(err) {
		t.Errorf("sandbox folder after archive = exists (err %v), want removed", err)
	}
	if err := st.Unarchive(s.ID()); err != nil {
		t.Fatalf("Unarchive: %v", err)
	}
	if fi, err := os.Stat(s.localDir); err != nil || !fi.IsDir() {
		t.Errorf("sandbox folder after unarchive missing: %v", err)
	}
	if reason := s.PausedReason(); reason != "" {
		t.Errorf("unarchived chat paused: %s", reason)
	}
}

// TestLocalSandboxRestartRestoresFlag checks a restart: a sandboxed chat is
// reconnected to its folder from the persisted flag, while a chat that
// predates local sandboxing keeps the server working directory (localDir "").
func TestLocalSandboxRestartRestoresFlag(t *testing.T) {
	dir := t.TempDir()
	dbPath := filepath.Join(dir, "porter.db")
	root := filepath.Join(dir, ".porter", "sandboxes")

	open := func(t *testing.T) (*Store, *db.DB) {
		t.Helper()
		d, err := db.Open(dbPath)
		if err != nil {
			t.Fatalf("open db: %v", err)
		}
		st := NewStore(d, nil)
		st.SetSandboxRoot(root)
		return st, d
	}

	st1, d1 := open(t)
	s, err := st1.Create(nil)
	if err != nil {
		t.Fatalf("Create: %v", err)
	}
	// A pre-sandboxing chat: a row created directly, never flagged.
	legacyID, err := d1.CreateSession(time.Now().UnixMilli())
	if err != nil {
		t.Fatalf("create legacy row: %v", err)
	}
	st1.Close()

	st2, _ := open(t)
	if err := st2.Load(nil); err != nil {
		t.Fatalf("Load: %v", err)
	}
	restored, ok := st2.Get(s.ID())
	if !ok {
		t.Fatalf("session %s not loaded", s.ID())
	}
	if restored.localDir != filepath.Join(root, s.ID()) {
		t.Errorf("restored localDir = %q, want %q", restored.localDir, filepath.Join(root, s.ID()))
	}
	out := runLocalShell(t, restored, "pwd")
	if !strings.Contains(out, restored.localDir) {
		t.Errorf("restored chat shell pwd = %q, want %q", out, restored.localDir)
	}
	legacy, ok := st2.Get("session_" + itoa(legacyID))
	if !ok {
		t.Fatalf("legacy session not loaded")
	}
	if legacy.localDir != "" {
		t.Errorf("legacy chat localDir = %q, want \"\" (keeps the server working directory)", legacy.localDir)
	}
	if reason := legacy.PausedReason(); reason != "" {
		t.Errorf("legacy chat paused: %s", reason)
	}
	st2.Close()
}

// TestLocalSandboxGCRemovesOrphans checks startup GC: folders no live chat
// owns — a crash between archiving and releasing, or a stray — are removed,
// while live chats' folders are kept.
func TestLocalSandboxGCRemovesOrphans(t *testing.T) {
	dir := t.TempDir()
	dbPath := filepath.Join(dir, "porter.db")
	root := filepath.Join(dir, ".porter", "sandboxes")

	d1, err := db.Open(dbPath)
	if err != nil {
		t.Fatalf("open db: %v", err)
	}
	st1 := NewStore(d1, nil)
	st1.SetSandboxRoot(root)
	live, err := st1.Create(nil)
	if err != nil {
		t.Fatalf("Create live: %v", err)
	}
	archived, err := st1.Create(nil)
	if err != nil {
		t.Fatalf("Create archived: %v", err)
	}
	// Simulate a crash between archiving and releasing: the archive is
	// persisted but ReleaseSession never ran, so the folder is orphaned.
	if err := st1.Archive(archived.ID()); err != nil {
		t.Fatalf("Archive: %v", err)
	}
	// A stray folder with no row at all.
	stray := filepath.Join(root, "session_999")
	if err := os.MkdirAll(stray, 0o755); err != nil {
		t.Fatalf("make stray: %v", err)
	}
	st1.Close()

	d2, err := db.Open(dbPath)
	if err != nil {
		t.Fatalf("reopen db: %v", err)
	}
	st2 := NewStore(d2, nil)
	st2.SetSandboxRoot(root)
	if err := st2.Load(nil); err != nil {
		t.Fatalf("Load: %v", err)
	}
	// Live folder survives; the archived chat's and the stray are gone.
	if _, err := os.Stat(live.localDir); err != nil {
		t.Errorf("live sandbox folder removed by GC: %v", err)
	}
	for _, gone := range []string{archived.localDir, stray} {
		if _, err := os.Stat(gone); !os.IsNotExist(err) {
			t.Errorf("orphan folder %s survived GC (err %v)", gone, err)
		}
	}
	st2.Close()
}

func itoa(n int64) string {
	return strconv.FormatInt(n, 10)
}
