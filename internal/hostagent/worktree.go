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

	"porter/internal/api"
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

// resolveRepo turns a repo path from the web UI into an absolute path on the
// host. Repos are named relative to the user's home directory — the same root
// discoverRepos scans — so a bare name like "porter" resolves to ~/porter, and
// a leading "~/" (or "~") is expanded to the home directory. Absolute paths
// pass through unchanged; the result is cleaned. A path that doesn't exist or
// isn't a repo is not an error here: provisionWorktree reports it, so the
// user sees the real git failure instead of a guessed path.
func resolveRepo(repo string) string {
	if repo == "" {
		return repo
	}
	if repo == "~" || strings.HasPrefix(repo, "~/") {
		if home, err := os.UserHomeDir(); err == nil && home != "" {
			repo = filepath.Join(home, strings.TrimPrefix(repo, "~"))
		}
	}
	if !filepath.IsAbs(repo) {
		if home, err := os.UserHomeDir(); err == nil && home != "" {
			repo = filepath.Join(home, repo)
		}
	}
	return filepath.Clean(repo)
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

// worktreeRef records one provisioned worktree sandbox so release and
// rollback can tear it down: the main repo it came from, the worktree path,
// and the branch created for it.
type worktreeRef struct {
	repo   string
	path   string
	branch string
}

// provisionWorktrees creates one git worktree per requested repo inside the
// sandbox container dir, following the naming rules that keep a chat's
// sandboxes legible and conflict-free:
//
//   - Branch: the first worktree of a repo keeps porter/<providerID>; a
//     second worktree of the same repo gets porter/<providerID>-2, and so
//     on, so comparing two branches of one repo works. Different repos share
//     the same branch name freely (branches live in each repo's own refs).
//   - Directory: the repo's basename ("porter", "data-kernel"). When the
//     same repo is requested more than once, every one of its directories
//     gets the base branch appended ("porter-main", "porter-feature") so the
//     two branches are distinguishable at a glance. Name collisions (two
//     repos with the same basename) get a numeric suffix: "porter", "porter-2".
//
// On error it returns the worktrees created so far plus the error — the
// caller rolls them back; provisionWorktrees never tears down after itself.
func provisionWorktrees(ctx context.Context, providerID string, repos []api.RepoRef, sandboxDir string) ([]worktreeRef, error) {
	counts := map[string]int{} // resolved repo path -> how many times requested
	for _, ref := range repos {
		counts[resolveRepo(ref.Path)]++
	}
	used := map[string]bool{}       // directory names already taken in this sandbox
	occurrences := map[string]int{} // resolved repo path -> worktrees so far
	baseBranch := "porter/" + sanitizeBranchID(providerID)
	var out []worktreeRef
	for _, ref := range repos {
		repo := resolveRepo(ref.Path)
		occurrences[repo]++
		branch := baseBranch
		if occurrences[repo] > 1 {
			branch = fmt.Sprintf("%s-%d", baseBranch, occurrences[repo])
		}
		name := sanitizeDirName(filepath.Base(repo))
		if counts[repo] > 1 {
			slug := sanitizeBranchID(branchSlug(ref.Branch))
			name = uniqueSandboxName(name+"-"+slug, used)
		} else {
			name = uniqueSandboxName(name, used)
		}
		path, err := provisionWorktree(ctx, repo, ref.Branch, branch, sandboxDir, name)
		if err != nil {
			return out, err
		}
		out = append(out, worktreeRef{repo: repo, path: path, branch: branch})
	}
	return out, nil
}

// branchSlug returns the base branch used to name a duplicated repo's
// sandbox directories, defaulting to "head" when no base branch was given.
func branchSlug(branch string) string {
	if branch == "" {
		return "head"
	}
	return branch
}

// uniqueSandboxName returns base if unused, else base-2, base-3, ... marking
// each name as taken. used must outlive all calls for one sandbox.
func uniqueSandboxName(base string, used map[string]bool) string {
	if !used[base] {
		used[base] = true
		return base
	}
	for n := 2; ; n++ {
		cand := fmt.Sprintf("%s-%d", base, n)
		if !used[cand] {
			used[cand] = true
			return cand
		}
	}
}

// provisionWorktree creates a git worktree sandbox for one session: branch
// checked out at base ("" = the repo's HEAD) in <root>/<name>. Returns the
// worktree path.
func provisionWorktree(ctx context.Context, repo, base, branch, root, name string) (string, error) {
	if err := os.MkdirAll(root, 0o755); err != nil {
		return "", fmt.Errorf("create sandbox root: %w", err)
	}
	path := filepath.Join(root, name)
	args := []string{"-C", repo, "worktree", "add", path, "-b", branch}
	if base != "" {
		args = append(args, base)
	}
	cmd := exec.CommandContext(ctx, "git", args...)
	if out, err := cmd.CombinedOutput(); err != nil {
		return "", fmt.Errorf("git worktree add: %v: %s", err, strings.TrimSpace(string(out)))
	}
	return path, nil
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
// worktrees/<name>"), so cleanup can derive the repo to remove — no sidecar
// needed. Two layouts are handled: a legacy flat sandbox where the entry
// itself is the worktree, and a multi-repo container whose entry holds one
// worktree per repo in subdirectories. The branch to delete is read from each
// worktree's HEAD (the branch the sandbox was created on). The repos are
// pruned of stale admin entries afterwards. Best effort: failures are logged,
// never fatal.
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
		path := filepath.Join(root, e.Name())
		if repo := repoOfWorktree(path); repo != "" {
			// Legacy flat sandbox: the entry is the worktree itself.
			if err := removeWorktree(repo, path, worktreeBranch(path)); err != nil {
				log.Printf("execution host: cleanup stale worktree %s: %v", path, err)
			}
			_ = os.RemoveAll(path)
			prune[repo] = true
			continue
		}
		// Multi-repo container: the entry holds one worktree per repo.
		subs, err := os.ReadDir(path)
		if err != nil {
			continue
		}
		removed := false
		for _, sub := range subs {
			if !sub.IsDir() {
				continue
			}
			sp := filepath.Join(path, sub.Name())
			repo := repoOfWorktree(sp)
			if repo == "" {
				continue // not a worktree we created: leave it
			}
			if err := removeWorktree(repo, sp, worktreeBranch(sp)); err != nil {
				log.Printf("execution host: cleanup stale worktree %s: %v", sp, err)
			}
			prune[repo] = true
			removed = true
		}
		if removed {
			_ = os.RemoveAll(path)
		}
	}
	for repo := range prune {
		pruneRepoWorktrees(repo)
	}
}

// worktreeBranch returns the branch a linked worktree currently has checked
// out — the branch the sandbox was created on — used by cleanup to delete it.
// Empty when it cannot be determined (git error) or the worktree is detached;
// removeWorktree then skips branch deletion, which only leaks the branch
// until git gc, never breaks cleanup.
func worktreeBranch(path string) string {
	cmd := exec.Command("git", "-C", path, "rev-parse", "--abbrev-ref", "HEAD")
	out, err := cmd.Output()
	if err != nil {
		return ""
	}
	branch := strings.TrimSpace(string(out))
	if branch == "HEAD" {
		return "" // detached: nothing to delete
	}
	return branch
}

// pruneRepoWorktrees runs git worktree prune on a repo, clearing admin
// entries for worktrees that were force-removed. Best effort.
func pruneRepoWorktrees(repo string) {
	_ = exec.Command("git", "-C", repo, "worktree", "prune").Run()
}

// sanitizeDirName makes a repo basename safe to use as a sandbox directory
// name: strips leading dots (a hidden directory is invisible to the file
// listing and easy to miss), replaces characters git or shells dislike, and
// caps the length so a pathological basename cannot create a giant path.
func sanitizeDirName(name string) string {
	name = strings.TrimSpace(name)
	name = strings.TrimLeft(name, ".")
	var b strings.Builder
	for _, r := range name {
		switch {
		case r >= 'a' && r <= 'z', r >= 'A' && r <= 'Z', r >= '0' && r <= '9', r == '.', r == '_', r == '-':
			b.WriteRune(r)
		default:
			b.WriteByte('-')
		}
	}
	s := strings.TrimRight(b.String(), ".-_")
	if s == "" {
		return "repo"
	}
	if len(s) > 80 {
		s = s[:80]
	}
	return s
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
