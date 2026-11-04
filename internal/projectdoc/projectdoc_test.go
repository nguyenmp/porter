package projectdoc

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// write puts a file in place, creating its directory.
func write(t *testing.T, path, body string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatalf("mkdir %s: %v", filepath.Dir(path), err)
	}
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatalf("write %s: %v", path, err)
	}
}

// TestFindWalksUpToRoot checks the walk covers every directory between the file
// and the root, and that the widest scope comes first.
func TestFindWalksUpToRoot(t *testing.T) {
	root := t.TempDir()
	write(t, filepath.Join(root, "AGENTS.md"), "root agents")
	write(t, filepath.Join(root, "CLAUDE.md"), "root claude")
	write(t, filepath.Join(root, "sub", "AGENTS.md"), "sub agents")
	write(t, filepath.Join(root, "sub", "deep", "file.go"), "package deep")

	docs := Find(filepath.Join(root, "sub", "deep", "file.go"), root)
	var got []string
	for _, d := range docs {
		got = append(got, filepath.ToSlash(strings.TrimPrefix(d.Path, root+string(filepath.Separator)))+"="+d.Body)
	}
	want := []string{"AGENTS.md=root agents", "CLAUDE.md=root claude", "sub/AGENTS.md=sub agents"}
	if strings.Join(got, "|") != strings.Join(want, "|") {
		t.Errorf("Find = %v, want %v", got, want)
	}
}

// TestFindStopsOutsideRoot checks a file outside the root gets no docs: the
// root bounds the walk, so a sandbox never picks up instructions from another
// tree.
func TestFindStopsOutsideRoot(t *testing.T) {
	root := t.TempDir()
	write(t, filepath.Join(root, "AGENTS.md"), "agents")
	other := t.TempDir()
	write(t, filepath.Join(other, "AGENTS.md"), "other agents")
	write(t, filepath.Join(other, "file.go"), "package other")

	if docs := Find(filepath.Join(other, "file.go"), root); len(docs) != 0 {
		t.Errorf("Find outside the root = %v, want none", docs)
	}
}

// TestFindSkipsEmptyAndNonFiles checks a whitespace-only doc and a directory
// named like a doc are both ignored.
func TestFindSkipsEmptyAndNonFiles(t *testing.T) {
	root := t.TempDir()
	write(t, filepath.Join(root, "AGENTS.md"), "  \n\n")
	if err := os.MkdirAll(filepath.Join(root, "CLAUDE.md"), 0o755); err != nil {
		t.Fatalf("mkdir: %v", err)
	}
	write(t, filepath.Join(root, "file.go"), "package root")

	if docs := Find(filepath.Join(root, "file.go"), root); len(docs) != 0 {
		t.Errorf("Find = %v, want none", docs)
	}
}

// TestFindTruncatesLongDoc checks a doc over MaxDoc is cut, keeps its first
// line, and says it was cut.
func TestFindTruncatesLongDoc(t *testing.T) {
	root := t.TempDir()
	body := "first line\n" + strings.Repeat("a longer line of instructions\n", MaxDoc/10)
	write(t, filepath.Join(root, "AGENTS.md"), body)
	write(t, filepath.Join(root, "file.go"), "package root")

	docs := Find(filepath.Join(root, "file.go"), root)
	if len(docs) != 1 {
		t.Fatalf("Find = %d docs, want 1", len(docs))
	}
	got := docs[0].Body
	if !strings.HasPrefix(got, "first line\n") {
		t.Errorf("truncated doc does not keep the opening: %q", got[:40])
	}
	if !strings.Contains(got, "truncated at") {
		t.Errorf("truncated doc does not say it was cut: %q", got[len(got)-60:])
	}
	if len(got) > MaxDoc+100 {
		t.Errorf("truncated doc is %d bytes, want about %d", len(got), MaxDoc)
	}
}

// TestMessageNamesEachDoc checks the block names every file, and that no docs
// render as the empty string so a caller can skip appending anything.
func TestMessageNamesEachDoc(t *testing.T) {
	if got := Message(nil); got != "" {
		t.Errorf("Message(nil) = %q, want \"\"", got)
	}
	got := Message([]Doc{
		{Path: "/repo/AGENTS.md", Body: "build with make"},
		{Path: "/repo/sub/CLAUDE.md", Body: "tests use gotestsum"},
	})
	for _, want := range []string{"/repo/AGENTS.md", "build with make", "/repo/sub/CLAUDE.md", "tests use gotestsum", "End of project instructions."} {
		if !strings.Contains(got, want) {
			t.Errorf("Message missing %q:\n%s", want, got)
		}
	}
	if !strings.Contains(got, "from the repository rather than from the user") {
		t.Errorf("Message does not say where the docs come from:\n%s", got)
	}
}
