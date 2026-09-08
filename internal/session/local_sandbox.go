package session

import (
	"fmt"
	"log"
	"os"
	"path/filepath"
)

// SetSandboxRoot configures the server's own per-chat sandbox root: every
// session the store creates from then on gets a folder <root>/session_<id>
// that its local provider works in. The server calls this once at startup,
// before Load; tests and embedders that opt out never call it, and their
// chats keep running in the process working directory (localDir stays "").
func (st *Store) SetSandboxRoot(root string) {
	st.sandboxRoot = root
}

// sandboxDirFor returns the server-local sandbox folder for a session id,
// or "" when local sandboxing is not configured. The folder name is the
// session's public id ("session_<n>"), which is unique forever and derivable
// at restart with no lookup — unlike host sandboxes, whose names must be
// minted by the server and recorded, the server owns this filesystem and can
// name the folder deterministically.
func (st *Store) sandboxDirFor(id string) string {
	if st.sandboxRoot == "" {
		return ""
	}
	return filepath.Join(st.sandboxRoot, id)
}

// createLocalSandbox makes the per-chat sandbox folder for a brand-new
// session and records the local_sandbox flag in the same breath, so a restart
// can tell this chat (whose folder must exist) from a pre-sandboxing chat
// (which keeps the server cwd as-is). Order matters: the folder is created
// first, then the flag, so a crash between them leaves an orphan folder that
// startup GC removes — never a flagged chat without its folder. On failure the
// folder is removed and the error returned, so Create fails rather than
// letting the chat run unsandboxed.
func (st *Store) createLocalSandbox(dbID int64, s *Session) error {
	dir := st.sandboxDirFor(s.id)
	if dir == "" {
		return nil // local sandboxing not configured (tests, embedders)
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return fmt.Errorf("create local sandbox %s: %w", dir, err)
	}
	if err := st.persist.SetLocalSandbox(dbID, true); err != nil {
		_ = os.RemoveAll(dir)
		return fmt.Errorf("record local sandbox for %s: %w", s.id, err)
	}
	s.setLocalDir(dir)
	return nil
}

// restoreLocalSandbox points a session loaded from the persister at its
// sandbox folder when the local_sandbox flag is set. The folder may be
// missing (deleted by hand); PausedReason then pauses the chat instead of
// silently running it in the server's working directory.
func (st *Store) restoreLocalSandbox(s *Session) {
	dir := st.sandboxDirFor(s.id)
	if dir == "" {
		return
	}
	s.setLocalDir(dir)
}

// removeLocalSandbox deletes a chat's sandbox folder (called when the chat is
// archived). The persisted flag is kept: unarchiving recreates the folder, so
// the chat never silently falls back to the server's working directory. The
// folder holds nothing but the chat's own working files, so removal is plain
// directory deletion — host worktrees have branches to preserve, an empty or
// scratch server-local folder has nothing worth keeping.
func (st *Store) removeLocalSandbox(s *Session) {
	if s == nil || s.localDir == "" {
		return
	}
	if err := os.RemoveAll(s.localDir); err != nil {
		log.Printf("session: remove local sandbox %s: %v", s.localDir, err)
	}
}

// recreateLocalSandbox restores an archived chat's sandbox folder on
// unarchive. Best-effort: a failure leaves the chat paused with a clear
// reason (folder missing) rather than running in the server cwd, and the user
// can archive and unarchive again to retry.
func (st *Store) recreateLocalSandbox(s *Session) {
	if s == nil || s.localDir == "" {
		return
	}
	if err := os.MkdirAll(s.localDir, 0o755); err != nil {
		log.Printf("session: recreate local sandbox %s: %v", s.localDir, err)
	}
}

// gcLocalSandboxes removes sandbox folders that no live chat owns: sessions
// whose rows or folders outlived a crash between archiving and release. It
// runs at startup after Load, and only ever touches the server's own root:
// a folder whose base name is not a live (loaded, unarchived) session's id is
// stale — the archived chat's ReleaseSession should have removed it, and a
// crash before that is exactly what this cleans up. Folders belonging to live
// chats are never touched, even when their files look empty.
func (st *Store) gcLocalSandboxes() {
	if st.sandboxRoot == "" {
		return
	}
	entries, err := os.ReadDir(st.sandboxRoot)
	if err != nil {
		return // no sandbox root yet (first run): nothing to clean
	}
	st.mu.Lock()
	keep := make(map[string]bool, len(st.sessions))
	for _, s := range st.sessions {
		if s.localDir != "" && !s.Archived() {
			keep[filepath.Base(s.localDir)] = true
		}
	}
	st.mu.Unlock()
	for _, e := range entries {
		if !e.IsDir() {
			continue
		}
		if keep[e.Name()] {
			continue
		}
		dir := filepath.Join(st.sandboxRoot, e.Name())
		if err := os.RemoveAll(dir); err != nil {
			log.Printf("session: gc local sandbox %s: %v", dir, err)
			continue
		}
		log.Printf("session: removed stale local sandbox %s", dir)
	}
}
