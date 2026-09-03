package exec

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"porter/internal/api"
	"porter/internal/humanize"
	"porter/internal/remoteedit"
)

// writeSkill creates a skill at root/<dir>/skills/<name>/SKILL.md and returns
// its path.
func writeSkill(t *testing.T, root, hiddenDir, name, body string) string {
	t.Helper()
	dir := filepath.Join(root, hiddenDir, "skills", name)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatalf("mkdir %s: %v", dir, err)
	}
	p := filepath.Join(dir, "SKILL.md")
	if err := os.WriteFile(p, []byte(body), 0o644); err != nil {
		t.Fatalf("write %s: %v", p, err)
	}
	return p
}

// findSkillsIn runs FindSkills with cwd in repo, overriding the user-home root
// via HOME so global skills resolve to home.
func findSkillsIn(t *testing.T, repo, home string) []api.Skill {
	t.Helper()
	t.Setenv("HOME", home)
	return FindSkills(repo)
}

func TestFindSkillsDiscoversRepoAndGlobal(t *testing.T) {
	repo := t.TempDir()
	home := t.TempDir()

	// Repo skills via the two patterns.
	writeSkill(t, repo, ".agents", "repo-skill", "# repo skill\n\nDo the repo thing.\n")
	writeSkill(t, repo, ".claude", "claude-skill", "# claude skill\n\nDo the claude thing.\n")
	// A global skill in the user root.
	writeSkill(t, home, ".agents", "global-skill", "# global skill\n\nDo the global thing.\n")
	// A duplicate name: repo wins over global.
	writeSkill(t, repo, ".agents", "dup", "---\nname: dup\n---\n# repo dup\n\nrepo version\n")
	writeSkill(t, home, ".agents", "dup", "# global dup\n\nglobal version\n")

	skills := findSkillsIn(t, repo, home)
	names := map[string]string{}
	for _, s := range skills {
		names[s.Name] = s.Path
	}
	for _, want := range []string{"repo-skill", "claude-skill", "global-skill", "dup"} {
		if _, ok := names[want]; !ok {
			t.Errorf("missing skill %q in %+v", want, names)
		}
	}
	// The dedup picks the repo (more specific) copy of "dup".
	dupPath := filepath.Join(repo, ".agents", "skills", "dup", "SKILL.md")
	if names["dup"] != dupPath {
		t.Errorf("dup skill path = %q, want repo copy %q", names["dup"], dupPath)
	}
	// The repo skill's description comes from its first non-heading line.
	for _, s := range skills {
		if s.Name == "repo-skill" && s.Description != "Do the repo thing." {
			t.Errorf("repo-skill description = %q, want first line", s.Description)
		}
	}
}

func TestFindSkillsSkipsDotAndDotDot(t *testing.T) {
	root := t.TempDir()
	home := t.TempDir()
	// A skills dir in the parent of root must NOT be found (the `.*` glob
	// matches `..`).
	parent := filepath.Dir(root)
	writeSkill(t, parent, ".agents", "parent-skill", "# parent skill\n")

	skills := findSkillsIn(t, root, home)
	for _, s := range skills {
		if s.Name == "parent-skill" {
			t.Errorf("found parent-dir skill %q at %q; should be out of scope", s.Name, s.Path)
		}
	}
}

func TestParseFrontmatter(t *testing.T) {
	fm, ok := parseFrontmatter("---\nname: foo\ndescription: \"Does the foo\"\n---\nbody\n")
	if !ok {
		t.Fatal("expected frontmatter")
	}
	if fm["name"] != "foo" || fm["description"] != "Does the foo" {
		t.Errorf("frontmatter = %+v, want name=foo, description=Does the foo", fm)
	}

	if _, ok := parseFrontmatter("no frontmatter\n"); ok {
		t.Error("expected no frontmatter")
	}
}

func TestSystemMessageRendersSections(t *testing.T) {
	c := api.ExecContext{
		System: "linux/amd64",
		CWD:    "/work",
		Files:  []string{"README.md", "main.go"},
		Skills: []api.Skill{{Name: "foo", Description: "does foo"}},
	}
	msg := SystemMessage(c)
	for _, want := range []string{
		"You are running on: linux/amd64",
		"Working directory: /work",
		"README.md",
		"main.go",
		"foo: does foo",
	} {
		if !strings.Contains(msg, want) {
			t.Errorf("system message missing %q:\n%s", want, msg)
		}
	}

	// Empty context yields an empty message.
	if got := SystemMessage(api.ExecContext{}); got != "" {
		t.Errorf("empty context message = %q, want empty", got)
	}
}

func TestDiscoverListsCWD(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "README.md"), []byte("# hi"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.Mkdir(filepath.Join(dir, "sub"), 0o755); err != nil {
		t.Fatal(err)
	}
	ctx, err := Discover(dir)
	if err != nil {
		t.Fatalf("Discover: %v", err)
	}
	if ctx.CWD != dir {
		t.Errorf("CWD = %q, want %q", ctx.CWD, dir)
	}
	if ctx.System == "" {
		t.Error("System empty")
	}
	var sawReadme, sawSub bool
	for _, f := range ctx.Files {
		if f == "README.md" {
			sawReadme = true
		}
		if f == "sub/" {
			sawSub = true
		}
	}
	if !sawReadme || !sawSub {
		t.Errorf("files = %v, want README.md and sub/", ctx.Files)
	}
}

func TestFindCLIsFromManifest(t *testing.T) {
	home := t.TempDir()
	dir := filepath.Join(home, ".porter")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	body := `{"clis":{"gt":"Graphite stack management (submit, split, restack, up/down)","slack":"Slack CLI","gws":"Google Workspace CLI"}}`
	if err := os.WriteFile(filepath.Join(dir, "clis.json"), []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	t.Setenv("HOME", home)
	clis := FindCLIs()
	if len(clis) != 3 {
		t.Fatalf("FindCLIs = %d entries, want 3", len(clis))
	}
	// Sorted by name.
	if clis[0].Name != "gt" || clis[1].Name != "gws" || clis[2].Name != "slack" {
		t.Errorf("FindCLIs order = %+v, want gt, gws, slack", clis)
	}
	desc := ""
	for _, c := range clis {
		if c.Name == "gt" {
			desc = c.Description
		}
	}
	if !strings.Contains(desc, "Graphite") {
		t.Errorf("gt description = %q", desc)
	}
}

func TestFindCLIsMissingOrBroken(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	// No manifest: no CLIs, no error.
	if got := FindCLIs(); len(got) != 0 {
		t.Errorf("FindCLIs with no manifest = %+v, want none", got)
	}
	// Malformed manifest: ignored, not fatal.
	dir := filepath.Join(home, ".porter")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "clis.json"), []byte(`{nope`), 0o644); err != nil {
		t.Fatal(err)
	}
	if got := FindCLIs(); len(got) != 0 {
		t.Errorf("FindCLIs with broken manifest = %+v, want none", got)
	}
}

func TestSystemMessageIncludesCLIs(t *testing.T) {
	msg := SystemMessage(api.ExecContext{
		System: "darwin/arm64",
		CWD:    "/home/me",
		CLIs: []api.CLI{
			{Name: "gt", Description: "Graphite stack management"},
		},
	})
	if !strings.Contains(msg, "Available CLI tools") || !strings.Contains(msg, "gt: Graphite stack management") {
		t.Errorf("system message missing CLI section:\n%s", msg)
	}
}

func TestDiscoverRootsListsFilesPerRoot(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "top.txt"), []byte("top"), 0o644); err != nil {
		t.Fatal(err)
	}
	// Two extra roots (e.g. worktrees in a multi-repo sandbox), each with its
	// own files.
	rootA := filepath.Join(dir, "porter-main")
	rootB := filepath.Join(dir, "data-kernel")
	for _, r := range []string{rootA, rootB} {
		if err := os.MkdirAll(r, 0o755); err != nil {
			t.Fatal(err)
		}
	}
	if err := os.WriteFile(filepath.Join(rootA, "go.mod"), []byte("module x"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(rootA, "internal"), []byte("dir"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.Mkdir(filepath.Join(rootB, "models"), 0o755); err != nil {
		t.Fatal(err)
	}

	ctx, err := DiscoverRoots(dir, []string{rootA, rootB})
	if err != nil {
		t.Fatalf("DiscoverRoots: %v", err)
	}
	// The container's own listing stays bare; each extra root's entries are
	// prefixed with the root's basename so the model can tell which repo a
	// file belongs to.
	want := map[string]bool{
		"top.txt":              true,
		"porter-main/":         true,
		"data-kernel/":         true,
		"porter-main/go.mod":   true,
		"porter-main/internal": true,
		"data-kernel/models/":  true,
	}
	for _, f := range ctx.Files {
		if !want[f] {
			t.Errorf("unexpected file %q (all: %v)", f, ctx.Files)
		}
		delete(want, f)
	}
	if len(want) != 0 {
		t.Errorf("missing files %v (all: %v)", want, ctx.Files)
	}
}

func TestListFilesInPerRootBudget(t *testing.T) {
	dir := t.TempDir()
	// The container root has more than maxFiles entries of its own.
	for i := 0; i < maxFiles+5; i++ {
		if err := os.WriteFile(filepath.Join(dir, fmt.Sprintf("f%d.txt", i)), []byte("x"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	// An extra root also has more than maxFiles entries.
	extra := filepath.Join(dir, "repo")
	if err := os.MkdirAll(extra, 0o755); err != nil {
		t.Fatal(err)
	}
	for i := 0; i < maxFiles+5; i++ {
		if err := os.WriteFile(filepath.Join(extra, fmt.Sprintf("g%d.txt", i)), []byte("x"), 0o644); err != nil {
			t.Fatal(err)
		}
	}

	names, err := ListFilesIn(dir, []string{extra})
	if err != nil {
		t.Fatalf("ListFilesIn: %v", err)
	}
	// Each root gets its own maxFiles budget: the container lists maxFiles of
	// its own entries and the extra root lists maxFiles prefixed entries — not
	// a shared cap. (maxFiles+maxFiles total.)
	if len(names) != 2*maxFiles {
		t.Errorf("ListFilesIn returned %d names, want %d (per-root budget)", len(names), 2*maxFiles)
	}
	prefixed := 0
	for _, n := range names {
		if strings.HasPrefix(n, "repo/") {
			prefixed++
		}
	}
	if prefixed != maxFiles {
		t.Errorf("extra root contributed %d entries, want %d", prefixed, maxFiles)
	}
}

func TestFindSkillsInOrdering(t *testing.T) {
	// Two roots (e.g. two worktrees in one sandbox) both define the same
	// skill name: the first root wins, mirroring FindSkills' repo-over-global
	// preference. The built-in skills are appended after the filesystem scan
	// (no filesystem copy shadows them in this test), so the result is the
	// deduped filesystem skill plus both built-ins.
	root1 := t.TempDir()
	root2 := t.TempDir()
	writeSkill(t, root1, ".agents", "dup", "# first\n\nfirst version\n")
	writeSkill(t, root2, ".agents", "dup", "# second\n\nsecond version\n")

	skills := FindSkillsIn([]string{root1, root2})
	byName := make(map[string]api.Skill, len(skills))
	for _, s := range skills {
		byName[s.Name] = s
	}
	if len(byName) != 3 {
		t.Fatalf("FindSkillsIn = %d unique skills, want 3 (filesystem dup + both built-ins)", len(byName))
	}
	want := filepath.Join(root1, ".agents", "skills", "dup", "SKILL.md")
	if got := byName["dup"].Path; got != want {
		t.Errorf("dup skill path = %q, want first root's %q", got, want)
	}
	// A filesystem skill named like a built-in shadows that one (first-root
	// rule); the other built-in is still appended.
	root3 := t.TempDir()
	writeSkill(t, root3, ".agents", humanize.SkillName, "# mine\n\nmy plain-language\n")
	skills = FindSkillsIn([]string{root3})
	byName = make(map[string]api.Skill, len(skills))
	for _, s := range skills {
		byName[s.Name] = s
	}
	if len(skills) != 2 {
		t.Fatalf("FindSkillsIn with shadowing skill = %d skills, want 2 (filesystem plain-language + built-in editing-remote-files)", len(skills))
	}
	if got := byName[humanize.SkillName].Path; got == api.BuiltinPrefix+humanize.SkillName {
		t.Errorf("built-in shadowed the filesystem skill; path = %q", got)
	}
	if _, ok := byName[remoteedit.SkillName]; !ok {
		t.Errorf("FindSkillsIn must still include the other built-in %q; got %v", remoteedit.SkillName, skills)
	}
}

func TestFindSkillsIncludesBuiltin(t *testing.T) {
	// With no skills anywhere, the binary's built-in skills (plain-language
	// and editing-remote-files) are still discovered: the server is a single
	// binary with no SKILL.md files, so they are hard-coded and served from
	// memory (sentinel path).
	skills := FindSkillsIn(nil)
	if len(skills) != 2 {
		t.Fatalf("FindSkillsIn(nil) = %d skills, want 2 (built-in plain-language + editing-remote-files)", len(skills))
	}
	byName := make(map[string]api.Skill, len(skills))
	for _, s := range skills {
		byName[s.Name] = s
	}
	for _, name := range []string{humanize.SkillName, remoteedit.SkillName} {
		s, ok := byName[name]
		if !ok {
			t.Errorf("missing built-in skill %q in %v", name, skills)
			continue
		}
		if s.Path != api.BuiltinPrefix+name {
			t.Errorf("built-in skill %q path = %q, want sentinel %q", name, s.Path, api.BuiltinPrefix+name)
		}
		if s.Description == "" {
			t.Errorf("built-in skill %q description is empty", name)
		}
	}
}
