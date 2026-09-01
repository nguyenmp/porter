package hostagent

import (
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"porter/internal/api"
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

// initGitRepoWithBranch creates a temp git repo whose "main" branch has
// hello.txt and an extra branch (created from main) that adds extra.txt, so a
// worktree based on it is distinguishable from one based on main. Returns the
// repo path.
func initGitRepoWithBranch(t *testing.T) string {
	t.Helper()
	repo := initGitRepo(t)
	runGit(t, repo, "checkout", "-b", "feature")
	if err := os.WriteFile(filepath.Join(repo, "extra.txt"), []byte("feature\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	runGit(t, repo, "add", ".")
	runGit(t, repo, "commit", "-m", "feature")
	runGit(t, repo, "checkout", "main")
	return repo
}

// branchOf returns the branch a worktree path currently has checked out.
func branchOf(t *testing.T, path string) string {
	t.Helper()
	return strings.TrimSpace(runGit(t, path, "rev-parse", "--abbrev-ref", "HEAD"))
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
	path, err := provisionWorktree(context.Background(), repo, "", "porter/mac-provider-1", root, "mac-provider-1")
	if err != nil {
		t.Fatalf("provisionWorktree: %v", err)
	}
	defer removeWorktree(repo, path, "porter/mac-provider-1")

	if got := branchOf(t, path); got != "porter/mac-provider-1" {
		t.Errorf("worktree branch = %q, want porter/mac-provider-1", got)
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

	path, err := provisionWorktree(context.Background(), repo, "feature", "porter/mac-provider-2", t.TempDir(), "mac-provider-2")
	if err != nil {
		t.Fatalf("provisionWorktree: %v", err)
	}
	defer removeWorktree(repo, path, "porter/mac-provider-2")
	if _, err := os.Stat(filepath.Join(path, "feature.txt")); err != nil {
		t.Fatalf("sandbox should be based on the feature branch: %v", err)
	}
}

func TestProvisionWorktreeNotARepo(t *testing.T) {
	_, err := provisionWorktree(context.Background(), t.TempDir(), "", "porter/mac-provider-3", t.TempDir(), "mac-provider-3")
	if err == nil {
		t.Fatal("provisioning a non-repo should fail")
	}
}

func TestRemoveWorktreeDeletesBranch(t *testing.T) {
	repo := initGitRepo(t)
	root := t.TempDir()
	path, err := provisionWorktree(context.Background(), repo, "", "porter/mac-provider-4", root, "mac-provider-4")
	if err != nil {
		t.Fatal(err)
	}
	if err := removeWorktree(repo, path, "porter/mac-provider-4"); err != nil {
		t.Fatalf("removeWorktree: %v", err)
	}
	if _, err := os.Stat(path); !os.IsNotExist(err) {
		t.Errorf("worktree dir still exists after remove")
	}
	if b := runGit(t, repo, "branch", "--list", "porter/mac-provider-4"); strings.TrimSpace(b) != "" {
		t.Errorf("branch %q still exists: %q", "porter/mac-provider-4", b)
	}
}

func TestCleanupStaleWorktrees(t *testing.T) {
	repo := initGitRepo(t)
	root := t.TempDir()
	path, err := provisionWorktree(context.Background(), repo, "", "porter/mac-provider-5", root, "mac-provider-5")
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
	path, err := provisionWorktree(context.Background(), repo, "", "porter/mac-provider-9", root, "mac-provider-9")
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

func TestResolveRepo(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	abs := filepath.Join(home, "porter")

	cases := []struct {
		name string
		in   string
		want string
	}{
		{"bare name resolves to home", "porter", abs},
		{"tilde-slash expands", "~/porter", abs},
		{"tilde alone expands", "~", home},
		{"absolute passes through", abs, abs},
		{"dot-slash resolves to home", "./porter", abs},
		{"empty stays empty", "", ""},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := resolveRepo(c.in); got != c.want {
				t.Errorf("resolveRepo(%q) = %q, want %q", c.in, got, c.want)
			}
		})
	}
}

// TestProvisionWorktreeResolvesRepoFromHome covers the bug where typing a bare
// repo name (e.g. "porter") in the web UI failed because it was resolved
// against the host process's cwd instead of the user's home directory: a bare
// name must mean ~/porter, matching how discoverRepos finds repos.
func TestProvisionWorktreeResolvesRepoFromHome(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	repo := filepath.Join(home, "porter")
	if err := os.MkdirAll(repo, 0o755); err != nil {
		t.Fatal(err)
	}
	runGit(t, repo, "init", "-b", "main")
	runGit(t, repo, "config", "user.email", "test@example.com")
	runGit(t, repo, "config", "user.name", "Test")
	if err := os.WriteFile(filepath.Join(repo, "hello.txt"), []byte("main\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	runGit(t, repo, "add", ".")
	runGit(t, repo, "commit", "-m", "initial")

	path, err := provisionWorktree(context.Background(), resolveRepo("porter"), "", "porter/mac-provider-5", t.TempDir(), "mac-provider-5")
	if err != nil {
		t.Fatalf("provisionWorktree with resolved repo: %v", err)
	}
	defer removeWorktree(repo, path, "porter/mac-provider-5")
	if _, err := os.Stat(filepath.Join(path, "hello.txt")); err != nil {
		t.Fatalf("sandbox should have the repo's files: %v", err)
	}
}

func TestProvisionWorktreesMultipleRepos(t *testing.T) {
	repo1 := initGitRepo(t)
	repo2 := initGitRepo(t)
	sandbox := t.TempDir()

	wts, err := provisionWorktrees(context.Background(), "mac-provider-10",
		[]api.RepoRef{{Path: repo1}, {Path: repo2}}, sandbox)
	if err != nil {
		t.Fatalf("provisionWorktrees: %v", err)
	}
	if len(wts) != 2 {
		t.Fatalf("got %d worktrees, want 2", len(wts))
	}
	// Each worktree sits in the sandbox container under its repo's basename
	// and has the repo's files.
	for i, w := range wts {
		wantDir := filepath.Join(sandbox, filepath.Base([]string{repo1, repo2}[i]))
		if w.path != wantDir {
			t.Errorf("worktree %d path = %q, want %q", i, w.path, wantDir)
		}
		if _, err := os.Stat(filepath.Join(w.path, "hello.txt")); err != nil {
			t.Errorf("worktree %d missing repo files: %v", i, err)
		}
		if got := branchOf(t, w.path); got != w.branch {
			t.Errorf("worktree %d branch = %q, want %q", i, got, w.branch)
		}
	}
	// Distinct repos share the plain branch name.
	if wts[0].branch != "porter/mac-provider-10" || wts[1].branch != "porter/mac-provider-10" {
		t.Errorf("branches = %q, %q; want porter/mac-provider-10 for both", wts[0].branch, wts[1].branch)
	}
	// Clean up via release-style teardown.
	for _, w := range wts {
		if err := removeWorktree(w.repo, w.path, w.branch); err != nil {
			t.Fatal(err)
		}
	}
}

func TestProvisionWorktreesDuplicateRepoTwoBranches(t *testing.T) {
	repo := initGitRepoWithBranch(t)
	sandbox := t.TempDir()

	wts, err := provisionWorktrees(context.Background(), "mac-provider-11",
		[]api.RepoRef{{Path: repo, Branch: "main"}, {Path: repo, Branch: "feature"}}, sandbox)
	if err != nil {
		t.Fatalf("provisionWorktrees: %v", err)
	}
	if len(wts) != 2 {
		t.Fatalf("got %d worktrees, want 2", len(wts))
	}
	// The duplicated repo's directories carry the base branch as a slug, so
	// the two sandboxes are distinguishable at a glance.
	base := filepath.Base(repo)
	wantDirs := map[string]bool{filepath.Join(sandbox, base+"-main"): true, filepath.Join(sandbox, base+"-feature"): true}
	for _, w := range wts {
		if !wantDirs[w.path] {
			t.Errorf("unexpected worktree dir %q (want %v)", w.path, wantDirs)
		}
	}
	// Branches: the first worktree keeps the plain name, the second gets -2.
	branches := map[string]bool{}
	for _, w := range wts {
		branches[w.branch] = true
		if got := branchOf(t, w.path); got != w.branch {
			t.Errorf("worktree %q branch = %q, want %q", w.path, got, w.branch)
		}
	}
	if !branches["porter/mac-provider-11"] || !branches["porter/mac-provider-11-2"] {
		t.Errorf("branches = %v, want porter/mac-provider-11 and porter/mac-provider-11-2", branches)
	}
	// The feature-based worktree has the extra file; the main-based one does
	// not — the two branches are genuinely isolated checkouts.
	for _, w := range wts {
		_, hasExtra := os.Stat(filepath.Join(w.path, "extra.txt"))
		if strings.HasSuffix(w.path, "-feature") && hasExtra != nil {
			t.Errorf("feature worktree missing extra.txt: %v", hasExtra)
		}
		if strings.HasSuffix(w.path, "-main") && hasExtra == nil {
			t.Errorf("main worktree should not have extra.txt")
		}
	}
}

func TestProvisionWorktreesSameRepoSameBranch(t *testing.T) {
	repo := initGitRepo(t)
	sandbox := t.TempDir()

	wts, err := provisionWorktrees(context.Background(), "mac-provider-12",
		[]api.RepoRef{{Path: repo, Branch: "main"}, {Path: repo, Branch: "main"}}, sandbox)
	if err != nil {
		t.Fatalf("provisionWorktrees: %v", err)
	}
	if len(wts) != 2 {
		t.Fatalf("got %d worktrees, want 2", len(wts))
	}
	// Two worktrees at the same base commit: allowed. Dirs are distinct
	// (repo-main, repo-main-2) and both branches exist.
	dirs := map[string]bool{}
	branches := map[string]bool{}
	for _, w := range wts {
		dirs[w.path] = true
		branches[w.branch] = true
	}
	base := filepath.Base(repo)
	if len(dirs) != 2 || !dirs[filepath.Join(sandbox, base+"-main")] || !dirs[filepath.Join(sandbox, base+"-main-2")] {
		t.Errorf("dirs = %v, want %s-main and %s-main-2", dirs, base, base)
	}
	if !branches["porter/mac-provider-12"] || !branches["porter/mac-provider-12-2"] {
		t.Errorf("branches = %v, want porter/mac-provider-12 and porter/mac-provider-12-2", branches)
	}
}

func TestProvisionWorktreesSameBasenameCollision(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	repoA := initGitRepo(t) // basename is a random temp name; force a shared basename below
	_ = repoA
	// Two repos in different parent dirs with the same basename.
	dirA := filepath.Join(home, "a")
	dirB := filepath.Join(home, "b")
	if err := os.MkdirAll(dirA, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(dirB, 0o755); err != nil {
		t.Fatal(err)
	}
	repoA = filepath.Join(dirA, "porter")
	repoB := filepath.Join(dirB, "porter")
	runGit(t, dirA, "init", "-b", "main", "porter")
	runGit(t, dirB, "init", "-b", "main", "porter")
	for _, r := range []string{repoA, repoB} {
		runGit(t, r, "config", "user.email", "test@example.com")
		runGit(t, r, "config", "user.name", "Test")
		if err := os.WriteFile(filepath.Join(r, "hello.txt"), []byte("main\n"), 0o644); err != nil {
			t.Fatal(err)
		}
		runGit(t, r, "add", ".")
		runGit(t, r, "commit", "-m", "initial")
	}

	sandbox := t.TempDir()
	wts, err := provisionWorktrees(context.Background(), "mac-provider-13",
		[]api.RepoRef{{Path: "a/porter"}, {Path: "b/porter"}}, sandbox)
	if err != nil {
		t.Fatalf("provisionWorktrees: %v", err)
	}
	if len(wts) != 2 {
		t.Fatalf("got %d worktrees, want 2", len(wts))
	}
	dirs := map[string]bool{}
	for _, w := range wts {
		dirs[w.path] = true
	}
	if !dirs[filepath.Join(sandbox, "porter")] || !dirs[filepath.Join(sandbox, "porter-2")] {
		t.Errorf("dirs = %v, want porter and porter-2", dirs)
	}
}

func TestProvisionWorktreesErrorKeepsPartial(t *testing.T) {
	repo := initGitRepo(t)
	sandbox := t.TempDir()

	// Second entry is not a repo: provisionWorktrees returns the first
	// worktree plus the error, so the caller can roll it back.
	wts, err := provisionWorktrees(context.Background(), "mac-provider-14",
		[]api.RepoRef{{Path: repo}, {Path: t.TempDir()}}, sandbox)
	if err == nil {
		t.Fatal("provisioning a non-repo should fail")
	}
	if len(wts) != 1 {
		t.Fatalf("got %d worktrees before failure, want 1", len(wts))
	}
	// The rollback (as provision does) removes the partial worktree.
	if err := removeWorktree(wts[0].repo, wts[0].path, wts[0].branch); err != nil {
		t.Fatalf("rollback: %v", err)
	}
	if _, err := os.Stat(wts[0].path); !os.IsNotExist(err) {
		t.Errorf("rolled-back worktree still exists")
	}
}

func TestReleaseRemovesMultiRepoSandbox(t *testing.T) {
	repo1 := initGitRepo(t)
	repo2 := initGitRepo(t)
	sandbox := t.TempDir()

	wts, err := provisionWorktrees(context.Background(), "mac-provider-15",
		[]api.RepoRef{{Path: repo1}, {Path: repo2}}, sandbox)
	if err != nil {
		t.Fatalf("provisionWorktrees: %v", err)
	}

	p := &provisioner{serves: map[string]*serveState{}}
	p.mu.Lock()
	p.serves["mac-provider-15"] = &serveState{dir: sandbox, worktrees: wts, cancel: func() {}}
	p.mu.Unlock()

	p.release(api.HostRequest{ProviderID: "mac-provider-15"})

	// Every worktree, its branch, and the container directory are gone.
	for _, w := range wts {
		if _, err := os.Stat(w.path); !os.IsNotExist(err) {
			t.Errorf("worktree %q still exists after release", w.path)
		}
		if b := runGit(t, w.repo, "branch", "--list", w.branch); strings.TrimSpace(b) != "" {
			t.Errorf("branch %q still exists after release", w.branch)
		}
	}
	if _, err := os.Stat(sandbox); !os.IsNotExist(err) {
		t.Errorf("sandbox container still exists after release")
	}
}

func TestCleanupStaleWorktreesContainerLayout(t *testing.T) {
	repo1 := initGitRepo(t)
	repo2 := initGitRepo(t)
	root := t.TempDir()
	sandbox := filepath.Join(root, "mac-provider-16")
	wts, err := provisionWorktrees(context.Background(), "mac-provider-16",
		[]api.RepoRef{{Path: repo1}, {Path: repo2}}, sandbox)
	if err != nil {
		t.Fatalf("provisionWorktrees: %v", err)
	}

	// Simulate a crashed host: no release happened; startup cleanup must
	// remove both worktrees, their branches, and the container.
	cleanupStaleWorktrees(root)

	for _, w := range wts {
		if _, err := os.Stat(w.path); !os.IsNotExist(err) {
			t.Errorf("stale worktree %q not cleaned up", w.path)
		}
		if b := runGit(t, w.repo, "branch", "--list", w.branch); strings.TrimSpace(b) != "" {
			t.Errorf("stale branch %q remains", w.branch)
		}
	}
	if _, err := os.Stat(sandbox); !os.IsNotExist(err) {
		t.Errorf("stale sandbox container not cleaned up")
	}
}

func TestCleanupStaleWorktreesDuplicateRepo(t *testing.T) {
	repo := initGitRepoWithBranch(t)
	root := t.TempDir()
	sandbox := filepath.Join(root, "mac-provider-17")
	wts, err := provisionWorktrees(context.Background(), "mac-provider-17",
		[]api.RepoRef{{Path: repo, Branch: "main"}, {Path: repo, Branch: "feature"}}, sandbox)
	if err != nil {
		t.Fatalf("provisionWorktrees: %v", err)
	}

	// Both branches of the same repo are cleaned up by HEAD discovery — the
	// cleanup can no longer guess the branch from the provider id alone.
	cleanupStaleWorktrees(root)

	for _, w := range wts {
		if _, err := os.Stat(w.path); !os.IsNotExist(err) {
			t.Errorf("stale worktree %q not cleaned up", w.path)
		}
		if b := runGit(t, w.repo, "branch", "--list", w.branch); strings.TrimSpace(b) != "" {
			t.Errorf("stale branch %q remains", w.branch)
		}
	}
}

func TestSanitizeDirName(t *testing.T) {
	cases := map[string]string{
		"porter":           "porter",
		"data-kernel":      "data-kernel",
		".hidden-repo":     "hidden-repo",
		"..dots":           "dots",
		"repo with spaces": "repo-with-spaces",
		"weird:name":       "weird-name",
		"...":              "repo",
		"":                 "repo",
	}
	for in, want := range cases {
		if got := sanitizeDirName(in); got != want {
			t.Errorf("sanitizeDirName(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestWorktreeBranch(t *testing.T) {
	repo := initGitRepo(t)
	path, err := provisionWorktree(context.Background(), repo, "", "porter/mac-provider-18", t.TempDir(), "mac-provider-18")
	if err != nil {
		t.Fatal(err)
	}
	defer removeWorktree(repo, path, "porter/mac-provider-18")

	if got := worktreeBranch(path); got != "porter/mac-provider-18" {
		t.Errorf("worktreeBranch = %q, want porter/mac-provider-18", got)
	}
	// Not a worktree (no git): empty, and removeWorktree skips branch delete.
	if got := worktreeBranch(t.TempDir()); got != "" {
		t.Errorf("worktreeBranch on plain dir = %q, want empty", got)
	}
}
