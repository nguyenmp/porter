package tools

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// writeFile puts a file in place, creating its directory.
func writeFile(t *testing.T, path, body string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatalf("mkdir %s: %v", filepath.Dir(path), err)
	}
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatalf("write %s: %v", path, err)
	}
}

// readTool runs a read_with_line_numbers call against root and returns its
// output.
func readTool(t *testing.T, d *Dispatcher, root, path string) string {
	t.Helper()
	res, err := runDir(context.Background(), d, ReadLinesTool, []byte(`{"path":"`+path+`"}`), root)
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}
	return res
}

// TestReadAppendsProjectDocs checks a read carries the AGENTS.md and CLAUDE.md
// files from the working directory down to the file, widest scope first.
func TestReadAppendsProjectDocs(t *testing.T) {
	root := t.TempDir()
	writeFile(t, filepath.Join(root, "AGENTS.md"), "root rules")
	writeFile(t, filepath.Join(root, "sub", "AGENTS.md"), "sub rules")
	writeFile(t, filepath.Join(root, "sub", "file.go"), "package sub")

	res := readTool(t, NewDispatcher(), root, "sub/file.go")
	if !strings.Contains(res, "package sub") {
		t.Errorf("output lost the file contents:\n%s", res)
	}
	for _, want := range []string{"root rules", "sub rules", "Project instructions"} {
		if !strings.Contains(res, want) {
			t.Errorf("output missing %q:\n%s", want, res)
		}
	}
	if strings.Index(res, "root rules") > strings.Index(res, "sub rules") {
		t.Errorf("the nearest directory's rules should come last:\n%s", res)
	}
	if !strings.Contains(res, "End of project instructions.") {
		t.Errorf("output does not close the doc block:\n%s", res)
	}
}

// TestProjectDocsAppendOnce checks a second read in the same tree adds no docs:
// the session already has them.
func TestProjectDocsAppendOnce(t *testing.T) {
	root := t.TempDir()
	writeFile(t, filepath.Join(root, "AGENTS.md"), "root rules")
	writeFile(t, filepath.Join(root, "a.go"), "package a")
	writeFile(t, filepath.Join(root, "b.go"), "package b")

	d := NewDispatcher()
	if res := readTool(t, d, root, "a.go"); !strings.Contains(res, "root rules") {
		t.Fatalf("first read is missing the docs:\n%s", res)
	}
	if res := readTool(t, d, root, "b.go"); strings.Contains(res, "root rules") {
		t.Errorf("second read repeated the docs:\n%s", res)
	}
}

// TestReadOfDocAppendsNothing checks reading an instruction file itself shows
// its text once — as the file — and not again as appended instructions.
func TestReadOfDocAppendsNothing(t *testing.T) {
	root := t.TempDir()
	writeFile(t, filepath.Join(root, "AGENTS.md"), "root rules")

	res := readTool(t, NewDispatcher(), root, "AGENTS.md")
	if strings.Count(res, "root rules") != 1 {
		t.Errorf("the doc appears %d times, want once:\n%s", strings.Count(res, "root rules"), res)
	}
	if strings.Contains(res, "Project instructions") {
		t.Errorf("reading the doc appended itself:\n%s", res)
	}
}

// TestNonFileToolPassesThrough checks a tool that is not one of the file tools
// gets no doc block, even when it works in the same directory.
func TestNonFileToolPassesThrough(t *testing.T) {
	root := t.TempDir()
	writeFile(t, filepath.Join(root, "AGENTS.md"), "root rules")

	res, err := run(context.Background(), NewDispatcher(), "shell", []byte(`{"command":"echo hi"}`))
	if err != nil {
		t.Fatalf("shell: %v", err)
	}
	if strings.Contains(res, "root rules") {
		t.Errorf("shell output carried project docs:\n%s", res)
	}
}

// TestReadOutsideWorkingDirGetsNoDocs checks a read outside the working
// directory picks up nothing: the working directory bounds the search.
func TestReadOutsideWorkingDirGetsNoDocs(t *testing.T) {
	root := t.TempDir()
	outside := t.TempDir()
	writeFile(t, filepath.Join(outside, "AGENTS.md"), "outside rules")
	writeFile(t, filepath.Join(outside, "file.go"), "package outside")

	res := readTool(t, NewDispatcher(), root, filepath.Join(outside, "file.go"))
	if strings.Contains(res, "outside rules") {
		t.Errorf("read outside the working directory carried docs:\n%s", res)
	}
}
