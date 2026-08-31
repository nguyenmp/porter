package hostagent

import (
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// initGitRepo creates a temp git repo with one commit on branch "main" and
// returns its path.
func initGitRepo(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	runGit(t, dir, "init", "-b", "main")
	runGit(t, dir, "config", "user.email", "test@example.com")
	runGit(t, dir, "config", "user.name", "Test")
	if err := os.WriteFile(filepath.Join(dir, "hello.txt"), []byte("main\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	runGit(t, dir, "add", ".")
	runGit(t, dir, "commit", "-m", "initial")
	return dir
}

// runGit runs git -C dir with args, failing the test on error, and returns
// its stdout.
func runGit(t *testing.T, dir string, args ...string) string {
	t.Helper()
	cmd := exec.Command("git", append([]string{"-C", dir}, args...)...)
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("git %v: %v: %s", args, err, out)
	}
	return string(out)
}

func TestProvisionWorktreeCreatesIsolatedSandbox(t *testing.T) {
	repo := initGitRepo(t)
	root := t.TempDir()
	path, branch, err := provisionWorktree(context.Background(), repo, "", root, "mac-provider-1")
	if err != nil {
		t.Fatalf("provisionWorktree: %v", err)
	}
	defer removeWorktree(repo, path, branch)

	if branch != "porter/mac-provider-1" {
		t.Errorf("branch = %q, want porter/mac-provider-1", branch)
	}
	// The sandbox has the repo's files and is a linked worktree (.git is a
	// file pointing at the main repo's admin dir, not a directory).
	if b, err := os.ReadFile(filepath.Join(path, "hello.txt")); err != nil || string(b) != "main\n" {
		t.Fatalf("sandbox file: %q, %v", b, err)
	}
	if fi, err := os.Stat(filepath.Join(path, ".git")); err != nil || fi.IsDir() {
		t.Fatalf("sandbox is not a linked worktree: %v", err)
	}
	// Editing the sandbox does not touch the main worktree: the two are
	// isolated working trees on the same repo.
	if err := os.WriteFile(filepath.Join(path, "hello.txt"), []byte("sandbox\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if b, _ := os.ReadFile(filepath.Join(repo, "hello.txt")); string(b) != "main\n" {
		t.Errorf("main worktree changed: %q", b)
	}
	// The worktree is self-describing: its .git file records the repo, so
	// crash cleanup can derive where it came from without a sidecar.
	b, err := os.ReadFile(filepath.Join(path, ".git"))
	if err != nil || !strings.HasPrefix(string(b), "gitdir:") || !strings.Contains(string(b), repo) {
		t.Errorf("worktree .git = %q, %v; want gitdir pointing at %q", b, err, repo)
	}
}

func TestProvisionWorktreeFromBaseBranch(t *testing.T) {
	repo := initGitRepo(t)
	runGit(t, repo, "checkout", "-b", "feature")
	if err := os.WriteFile(filepath.Join(repo, "feature.txt"), []byte("feat\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	runGit(t, repo, "add", ".")
	runGit(t, repo, "commit", "-m", "feature")
	runGit(t, repo, "checkout", "main")

	path, branch, err := provisionWorktree(context.Background(), repo, "feature", t.TempDir(), "mac-provider-2")
	if err != nil {
		t.Fatalf("provisionWorktree: %v", err)
	}
	defer removeWorktree(repo, path, branch)
	if _, err := os.Stat(filepath.Join(path, "feature.txt")); err != nil {
		t.Fatalf("sandbox should be based on the feature branch: %v", err)
	}
}

func TestProvisionWorktreeNotARepo(t *testing.T) {
	_, _, err := provisionWorktree(context.Background(), t.TempDir(), "", t.TempDir(), "mac-provider-3")
	if err == nil {
		t.Fatal("provisioning a non-repo should fail")
	}
}

func TestRemoveWorktreeDeletesBranch(t *testing.T) {
	repo := initGitRepo(t)
	root := t.TempDir()
	path, branch, err := provisionWorktree(context.Background(), repo, "", root, "mac-provider-4")
	if err != nil {
		t.Fatal(err)
	}
	if err := removeWorktree(repo, path, branch); err != nil {
		t.Fatalf("removeWorktree: %v", err)
	}
	if _, err := os.Stat(path); !os.IsNotExist(err) {
		t.Errorf("worktree dir still exists after remove")
	}
	if b := runGit(t, repo, "branch", "--list", branch); strings.TrimSpace(b) != "" {
		t.Errorf("branch %q still exists: %q", branch, b)
	}
}

func TestCleanupStaleWorktrees(t *testing.T) {
	repo := initGitRepo(t)
	root := t.TempDir()
	path, _, err := provisionWorktree(context.Background(), repo, "", root, "mac-provider-5")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(path); err != nil {
		t.Fatalf("worktree missing: %v", err)
	}

	// Simulate a crashed host: no release happened; startup cleanup must
	// remove the sandbox, its branch, and the sidecar.
	cleanupStaleWorktrees(root)

	if _, err := os.Stat(path); !os.IsNotExist(err) {
		t.Errorf("stale worktree not cleaned up")
	}
	if b := runGit(t, repo, "branch", "--list", "porter/mac-provider-5"); strings.TrimSpace(b) != "" {
		t.Errorf("stale branch remains: %q", b)
	}
}

func TestSanitizeBranchID(t *testing.T) {
	cases := map[string]string{
		"mac-provider-1":   "mac-provider-1",
		"host with spaces": "host-with-spaces",
		"-leading":         "sandbox--leading",
		"..":               "sandbox",
		"dot.trailing.":    "dot.trailing",
		"a..b":             "sandbox-a..b",
		"weird/name:colon": "weird-name-colon",
	}
	for in, want := range cases {
		if got := sanitizeBranchID(in); got != want {
			t.Errorf("sanitizeBranchID(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestDiscoverRepos(t *testing.T) {
	home := t.TempDir()
	mk := func(rel string) {
		t.Helper()
		if err := os.MkdirAll(filepath.Join(home, rel), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.Mkdir(filepath.Join(home, rel, ".git"), 0o755); err != nil {
			t.Fatal(err)
		}
	}
	mk("repo1")
	mk("code/porter")
	mk("code/work/repo3")             // depth 3
	mk("code/work/deep/repo4")        // depth 4: at the bound, included
	mk("code/work/deep/nested/repo5") // depth 5: beyond the bound, skipped
	mk("hidden/.secretrepo")          // under a hidden dir: skipped
	// Non-repo dirs
	if err := os.MkdirAll(filepath.Join(home, "plain"), 0o755); err != nil {
		t.Fatal(err)
	}
	// Skipped heavy trees even if they contain a repo
	mk("Library/SomeRepo")
	mk("node_modules/pkg")

	got := discoverRepos(home)
	want := map[string]bool{
		filepath.Join(home, "code", "porter"):                true,
		filepath.Join(home, "code", "work", "repo3"):         true,
		filepath.Join(home, "code", "work", "deep", "repo4"): true,
		filepath.Join(home, "repo1"):                         true,
	}
	if len(got) != len(want) {
		t.Fatalf("discoverRepos = %v, want %v", got, want)
	}
	for _, g := range got {
		if !want[g] {
			t.Errorf("discoverRepos found unexpected %q (all: %v)", g, got)
		}
	}
}

func TestDiscoverReposEmptyHome(t *testing.T) {
	if got := discoverRepos(t.TempDir()); len(got) != 0 {
		t.Errorf("discoverRepos on empty home = %v, want none", got)
	}
}

func TestRepoOfWorktree(t *testing.T) {
	repo := initGitRepo(t)
	root := t.TempDir()
	path, _, err := provisionWorktree(context.Background(), repo, "", root, "mac-provider-9")
	if err != nil {
		t.Fatal(err)
	}
	defer removeWorktree(repo, path, "porter/mac-provider-9")

	// git resolves the repo path through the /var -> /private/var symlink on
	// macOS, so compare against the resolved path.
	resolved, err := filepath.EvalSymlinks(repo)
	if err != nil {
		t.Fatal(err)
	}
	if got := repoOfWorktree(path); got != resolved {
		t.Errorf("repoOfWorktree = %q, want %q", got, resolved)
	}
	// A plain directory (not a linked worktree) has no gitdir file.
	if got := repoOfWorktree(t.TempDir()); got != "" {
		t.Errorf("repoOfWorktree on plain dir = %q, want empty", got)
	}
}
