package hostagent

import (
	"context"
	"fmt"
	"io/fs"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
)

// worktreeRoot returns the directory worktree sandboxes live in. It is
// deliberately not configurable: ~/.porter/sandboxes, next to the other
// porter state (~/.porter for the MCP config).
func worktreeRoot() string {
	if home, err := os.UserHomeDir(); err == nil && home != "" {
		return filepath.Join(home, ".porter", "sandboxes")
	}
	return filepath.Join(os.TempDir(), "porter-sandboxes")
}

// discoverRepos finds the git repositories under home (defaulting to the
// user's home directory) to offer as worktree-sandbox targets in the web
// UI's "new chat" picker: it walks home, bounded in depth and count, and
// collects every directory containing a .git entry. Hidden directories and
// common heavy trees (Library, node_modules, ...) are skipped so the scan
// stays fast and predictable; a repo the scan misses can still be typed by
// path in the UI.
func discoverRepos(home string) []string {
	if home == "" {
		home, _ = os.UserHomeDir()
	}
	var repos []string
	seen := map[string]bool{}
	const maxDepth = 4
	const maxRepos = 100

	walkFn := func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return nil // unreadable dir: skip
		}
		if path == home {
			return nil
		}
		if d.IsDir() {
			base := d.Name()
			if base == ".git" || strings.HasPrefix(base, ".") || skipDir(base) {
				return filepath.SkipDir
			}
			if depth := strings.Count(path[len(home):], string(os.PathSeparator)); depth > maxDepth {
				return filepath.SkipDir
			}
			if _, err := os.Stat(filepath.Join(path, ".git")); err == nil {
				if !seen[path] {
					repos = append(repos, path)
					seen[path] = true
				}
				return filepath.SkipDir // a repo's worktrees are found via git, not the walk
			}
			return nil
		}
		return nil
	}
	_ = filepath.WalkDir(home, walkFn)
	if len(repos) > maxRepos {
		repos = repos[:maxRepos]
	}
	return repos
}

// skipDir reports whether a directory tree should be skipped during repo
// discovery: hidden dirs are handled by the caller; this is for common heavy
// or irrelevant trees that are not user projects.
func skipDir(base string) bool {
	switch base {
	case "Library", "node_modules", "Applications", "Caches", "vendor", "Pods", "DerivedData", ".venv", "venv":
		return true
	}
	return false
}

// provisionWorktree creates a git worktree sandbox for one session: a fresh
// branch (porter/<id>) checked out at base ("" = the repo's HEAD) in
// <root>/<id>. Returns the worktree path and branch name.
func provisionWorktree(ctx context.Context, repo, base, root, id string) (string, string, error) {
	if err := os.MkdirAll(root, 0o755); err != nil {
		return "", "", fmt.Errorf("create sandbox root: %w", err)
	}
	path := filepath.Join(root, id)
	branch := "porter/" + sanitizeBranchID(id)
	args := []string{"-C", repo, "worktree", "add", path, "-b", branch}
	if base != "" {
		args = append(args, base)
	}
	cmd := exec.CommandContext(ctx, "git", args...)
	if out, err := cmd.CombinedOutput(); err != nil {
		return "", "", fmt.Errorf("git worktree add: %v: %s", err, strings.TrimSpace(string(out)))
	}
	return path, branch, nil
}

// removeWorktree removes a worktree sandbox and deletes its branch. Callers
// treat a missing worktree/branch as best-effort cleanup, so errors name the
// exact git failure.
func removeWorktree(repo, path, branch string) error {
	cmd := exec.Command("git", "-C", repo, "worktree", "remove", "--force", path)
	if out, err := cmd.CombinedOutput(); err != nil {
		return fmt.Errorf("git worktree remove: %v: %s", err, strings.TrimSpace(string(out)))
	}
	if branch != "" {
		cmd := exec.Command("git", "-C", repo, "branch", "-D", branch)
		if out, err := cmd.CombinedOutput(); err != nil {
			return fmt.Errorf("git branch -D %s: %v: %s", branch, err, strings.TrimSpace(string(out)))
		}
	}
	return nil
}

// cleanupStaleWorktrees removes worktree sandboxes left behind by a previous
// host run (the host died without releasing them). Each sandbox is a linked
// worktree whose .git file records the repo it came from ("gitdir: <repo>/.git/
// worktrees/<name>"), so cleanup can derive the repo and branch to remove —
// no sidecar needed. The repos are pruned of stale admin entries afterwards.
// Best effort: failures are logged, never fatal.
func cleanupStaleWorktrees(root string) {
	entries, err := os.ReadDir(root)
	if err != nil {
		return // no sandbox root yet: nothing to clean
	}
	prune := map[string]bool{}
	for _, e := range entries {
		if !e.IsDir() {
			continue
		}
		id := e.Name()
		path := filepath.Join(root, id)
		repo := repoOfWorktree(path)
		if repo == "" {
			continue // not a worktree we created (no gitdir file): leave it
		}
		if err := removeWorktree(repo, path, "porter/"+sanitizeBranchID(id)); err != nil {
			log.Printf("execution host: cleanup stale worktree %s: %v", path, err)
		}
		_ = os.RemoveAll(path)
		prune[repo] = true
	}
	for repo := range prune {
		_ = exec.Command("git", "-C", repo, "worktree", "prune").Run()
	}
}

// repoOfWorktree parses a linked worktree's .git file ("gitdir: <path>") and
// returns the main repo's path, or "" when path is not a linked worktree we
// created.
func repoOfWorktree(path string) string {
	b, err := os.ReadFile(filepath.Join(path, ".git"))
	if err != nil {
		return ""
	}
	line := strings.TrimSpace(string(b))
	if !strings.HasPrefix(line, "gitdir:") {
		return ""
	}
	gitdir := strings.TrimSpace(strings.TrimPrefix(line, "gitdir:"))
	// gitdir is <repo>/.git/worktrees/<name>: the repo is everything before
	// the /.git/ segment.
	marker := string(os.PathSeparator) + ".git" + string(os.PathSeparator)
	idx := strings.Index(gitdir, marker)
	if idx <= 0 {
		return ""
	}
	return gitdir[:idx]
}

// sanitizeBranchID makes a provider id safe to use as the tail of a git
// branch name (porter/<id>).
func sanitizeBranchID(id string) string {
	var b strings.Builder
	for _, r := range id {
		switch {
		case r >= 'a' && r <= 'z', r >= 'A' && r <= 'Z', r >= '0' && r <= '9', r == '.', r == '_', r == '-':
			b.WriteRune(r)
		default:
			b.WriteByte('-')
		}
	}
	s := strings.Trim(b.String(), ".")
	if s == "" {
		return "sandbox"
	}
	if strings.HasPrefix(s, "-") || strings.Contains(s, "..") {
		s = "sandbox-" + s
	}
	return s
}
