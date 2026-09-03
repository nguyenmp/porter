package tools

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// writeTemp writes content to a fresh file in a temp dir and returns its path.
func writeTemp(t *testing.T, content string, mode os.FileMode) string {
	t.Helper()
	dir := t.TempDir()
	path := filepath.Join(dir, "f.txt")
	if err := os.WriteFile(path, []byte(content), mode); err != nil {
		t.Fatal(err)
	}
	return path
}

func readTemp(t *testing.T, path string) string {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	return string(b)
}

func TestReadFullFile(t *testing.T) {
	path := writeTemp(t, "a\nb\nc\n", 0o644)
	res, err := run(context.Background(), NewDispatcher(), ReadLinesTool, []byte(`{"path":`+quote(path)+`}`))
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	wantHeader := "[file: 3 lines, 6 bytes, ends with newline: yes, showing lines 1-3]\n"
	if !strings.HasPrefix(res, wantHeader) {
		t.Errorf("read header mismatch:\n got %q\nwant prefix %q", res, wantHeader)
	}
	for _, want := range []string{"\n     1\ta\n", "\n     2\tb\n", "\n     3\tc\n"} {
		if !strings.Contains(res, want) {
			t.Errorf("read result missing %q:\n%s", want, res)
		}
	}
}

func TestReadWindow(t *testing.T) {
	path := writeTemp(t, "1\n2\n3\n4\n5\n", 0o644)
	res, err := run(context.Background(), NewDispatcher(), ReadLinesTool, []byte(`{"path":`+quote(path)+`,"start":2,"limit":2}`))
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if !strings.HasPrefix(res, "[file: 5 lines, 10 bytes, ends with newline: yes, showing lines 2-3]\n") {
		t.Errorf("window header mismatch:\n%s", res)
	}
	if strings.Contains(res, "     1\t") || strings.Contains(res, "     4\t") {
		t.Errorf("window leaked lines outside 2-3:\n%s", res)
	}
	if !strings.Contains(res, "\n     2\t2\n") || !strings.Contains(res, "\n     3\t3\n") {
		t.Errorf("window missing lines 2-3:\n%s", res)
	}
}

func TestReadNoTrailingNewline(t *testing.T) {
	path := writeTemp(t, "a\nb", 0o644)
	res, err := run(context.Background(), NewDispatcher(), ReadLinesTool, []byte(`{"path":`+quote(path)+`}`))
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if !strings.Contains(res, "ends with newline: no, showing lines 1-2") {
		t.Errorf("header must report the missing trailing newline:\n%s", res)
	}
	// Both lines must be numbered; the last line lacks a newline in the file.
	if !strings.Contains(res, "\n     2\tb\n") {
		t.Errorf("last line missing from read:\n%s", res)
	}
}

func TestReadEmptyFile(t *testing.T) {
	path := writeTemp(t, "", 0o644)
	res, err := run(context.Background(), NewDispatcher(), ReadLinesTool, []byte(`{"path":`+quote(path)+`}`))
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if !strings.HasPrefix(res, "[file: 0 lines, 0 bytes, ends with newline: no, showing no lines]") {
		t.Errorf("empty-file header mismatch:\n%q", res)
	}
}

func TestReadErrors(t *testing.T) {
	dir := t.TempDir()
	if _, err := run(context.Background(), NewDispatcher(), ReadLinesTool, []byte(`{"path":`+quote(filepath.Join(dir, "nope.txt"))+`}`)); err == nil || !strings.Contains(err.Error(), "no such file") {
		t.Errorf("missing file: err = %v, want 'no such file'", err)
	}
	if _, err := run(context.Background(), NewDispatcher(), ReadLinesTool, []byte(`{"path":`+quote(dir)+`}`)); err == nil || !strings.Contains(err.Error(), "directory") {
		t.Errorf("directory: err = %v, want directory error", err)
	}
	bin := filepath.Join(dir, "bin.dat")
	if err := os.WriteFile(bin, []byte{'a', 0, 'b'}, 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := run(context.Background(), NewDispatcher(), ReadLinesTool, []byte(`{"path":`+quote(bin)+`}`)); err == nil || !strings.Contains(err.Error(), "binary") {
		t.Errorf("binary: err = %v, want binary error", err)
	}
	path := writeTemp(t, "a\n", 0o644)
	if _, err := run(context.Background(), NewDispatcher(), ReadLinesTool, []byte(`{"path":`+quote(path)+`,"start":3}`)); err == nil || !strings.Contains(err.Error(), "past the end") {
		t.Errorf("start past end: err = %v", err)
	}
	if _, err := run(context.Background(), NewDispatcher(), ReadLinesTool, []byte(`{"path":`+quote(path)+`,"limit":0}`)); err == nil {
		t.Errorf("limit 0 should be rejected")
	}
}

func TestLineReplaceRange(t *testing.T) {
	path := writeTemp(t, "a\nb\nc\nd\n", 0o644)
	res, err := run(context.Background(), NewDispatcher(), LineReplaceTool, []byte(`{"path":`+quote(path)+`,"start":2,"end":3,"new_text":"X\nY\n"}`))
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if got, want := readTemp(t, path), "a\nX\nY\nd\n"; got != want {
		t.Errorf("content = %q, want %q", got, want)
	}
	if !strings.Contains(res, "replaced lines 2-3 with new text") {
		t.Errorf("echo missing action: %s", res)
	}
	if !strings.Contains(res, "removed (old line numbers):") || !strings.Contains(res, "now (new line numbers):") {
		t.Errorf("echo missing before/after blocks:\n%s", res)
	}
}

func TestLineReplaceSingleLine(t *testing.T) {
	path := writeTemp(t, "a\nb\nc\n", 0o644)
	res, err := run(context.Background(), NewDispatcher(), LineReplaceTool, []byte(`{"path":`+quote(path)+`,"start":2,"end":2,"new_text":"X\n"}`))
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if got, want := readTemp(t, path), "a\nX\nc\n"; got != want {
		t.Errorf("content = %q, want %q", got, want)
	}
	if !strings.Contains(res, "replaced line 2 with new text") {
		t.Errorf("echo missing action: %s", res)
	}
}

func TestLineReplaceAutoTrailingNewline(t *testing.T) {
	// Replacement text that does not end in a newline must not glue onto the
	// following line: the tool treats new text as whole lines and adds the
	// missing newline itself.
	path := writeTemp(t, "a\nb\nc\nd\n", 0o644)
	if _, err := run(context.Background(), NewDispatcher(), LineReplaceTool, []byte(`{"path":`+quote(path)+`,"start":2,"end":3,"new_text":"X\nY"}`)); err != nil {
		t.Fatalf("Run: %v", err)
	}
	if got, want := readTemp(t, path), "a\nX\nY\nd\n"; got != want {
		t.Errorf("content = %q, want %q (trailing newline added, no glue)", got, want)
	}
}

func TestLineReplaceDeleteRange(t *testing.T) {
	path := writeTemp(t, "a\nb\nc\nd\n", 0o644)
	res, err := run(context.Background(), NewDispatcher(), LineReplaceTool, []byte(`{"path":`+quote(path)+`,"start":2,"end":3,"new_text":""}`))
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if got, want := readTemp(t, path), "a\nd\n"; got != want {
		t.Errorf("content = %q, want %q", got, want)
	}
	if !strings.Contains(res, "deleted lines 2-3") {
		t.Errorf("echo missing action: %s", res)
	}
}

func TestLineReplaceDeleteSingleLine(t *testing.T) {
	path := writeTemp(t, "a\nb\nc\nd\n", 0o644)
	if _, err := run(context.Background(), NewDispatcher(), LineReplaceTool, []byte(`{"path":`+quote(path)+`,"start":2,"end":2,"new_text":""}`)); err != nil {
		t.Fatalf("Run: %v", err)
	}
	if got, want := readTemp(t, path), "a\nc\nd\n"; got != want {
		t.Errorf("content = %q, want %q", got, want)
	}
}

func TestLineReplaceWholeFile(t *testing.T) {
	path := writeTemp(t, "a\nb\nc\n", 0o644)
	res, err := run(context.Background(), NewDispatcher(), LineReplaceTool, []byte(`{"path":`+quote(path)+`,"start":1,"end":3,"new_text":"X\nY\n"}`))
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if got, want := readTemp(t, path), "X\nY\n"; got != want {
		t.Errorf("content = %q, want %q", got, want)
	}
	if !strings.Contains(res, "replaced the whole file with new text") {
		t.Errorf("echo missing action: %s", res)
	}
}

func TestLineReplaceFinalUnterminatedLine(t *testing.T) {
	// Replacing a range that includes the file's final unterminated line
	// leaves the file ending in a newline (replacement text is whole lines).
	path := writeTemp(t, "a\nb", 0o644)
	if _, err := run(context.Background(), NewDispatcher(), LineReplaceTool, []byte(`{"path":`+quote(path)+`,"start":2,"end":2,"new_text":"c"}`)); err != nil {
		t.Fatalf("Run: %v", err)
	}
	if got, want := readTemp(t, path), "a\nc\n"; got != want {
		t.Errorf("content = %q, want %q", got, want)
	}
}

func TestLineInsertBefore(t *testing.T) {
	path := writeTemp(t, "a\nb\nc\n", 0o644)
	res, err := run(context.Background(), NewDispatcher(), LineInsertTool, []byte(`{"path":`+quote(path)+`,"start":2,"new_text":"X\nY\n"}`))
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if got, want := readTemp(t, path), "a\nX\nY\nb\nc\n"; got != want {
		t.Errorf("content = %q, want %q", got, want)
	}
	if !strings.Contains(res, "inserted 2 lines before line 2") {
		t.Errorf("echo missing action: %s", res)
	}
}

func TestLineInsertAutoTrailingNewline(t *testing.T) {
	// Inserted text that does not end in a newline must not glue onto the
	// following line.
	path := writeTemp(t, "a\nb\nc\n", 0o644)
	if _, err := run(context.Background(), NewDispatcher(), LineInsertTool, []byte(`{"path":`+quote(path)+`,"start":2,"new_text":"X\nY"}`)); err != nil {
		t.Fatalf("Run: %v", err)
	}
	if got, want := readTemp(t, path), "a\nX\nY\nb\nc\n"; got != want {
		t.Errorf("content = %q, want %q (trailing newline added, no glue)", got, want)
	}
}

func TestLineInsertAtTop(t *testing.T) {
	path := writeTemp(t, "a\nb\nc\n", 0o644)
	if _, err := run(context.Background(), NewDispatcher(), LineInsertTool, []byte(`{"path":`+quote(path)+`,"start":1,"new_text":"X\n"}`)); err != nil {
		t.Fatalf("Run: %v", err)
	}
	if got, want := readTemp(t, path), "X\na\nb\nc\n"; got != want {
		t.Errorf("content = %q, want %q", got, want)
	}
}

func TestLineInsertAppend(t *testing.T) {
	path := writeTemp(t, "a\nb\n", 0o644)
	res, err := run(context.Background(), NewDispatcher(), LineInsertTool, []byte(`{"path":`+quote(path)+`,"start":3,"new_text":"c\n"}`))
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if got, want := readTemp(t, path), "a\nb\nc\n"; got != want {
		t.Errorf("content = %q, want %q", got, want)
	}
	if !strings.Contains(res, "appended 1 line at the end of the file") {
		t.Errorf("echo missing action: %s", res)
	}
}

func TestLineInsertIntoEmptyFile(t *testing.T) {
	path := writeTemp(t, "", 0o644)
	if _, err := run(context.Background(), NewDispatcher(), LineInsertTool, []byte(`{"path":`+quote(path)+`,"start":1,"new_text":"a\n"}`)); err != nil {
		t.Fatalf("Run: %v", err)
	}
	if got, want := readTemp(t, path), "a\n"; got != want {
		t.Errorf("content = %q, want %q", got, want)
	}
}

func TestLineInsertAppendRejectedWithoutTrailingNewline(t *testing.T) {
	// There is no line boundary after an unterminated last line, so appending
	// there is rejected instead of gluing.
	path := writeTemp(t, "a\nb", 0o644)
	_, err := run(context.Background(), NewDispatcher(), LineInsertTool, []byte(`{"path":`+quote(path)+`,"start":3,"new_text":"c\n"}`))
	if err == nil || !strings.Contains(err.Error(), "does not end in a newline") {
		t.Errorf("append to unterminated file: err = %v, want a no-line-boundary error", err)
	}
	if got := readTemp(t, path); got != "a\nb" {
		t.Errorf("rejected append must not modify the file, got %q", got)
	}
}

func TestLineReplaceValidation(t *testing.T) {
	path := writeTemp(t, "a\nb\nc\n", 0o644)
	cases := []struct {
		name string
		args string
		want string
	}{
		{"start below 1", `{"path":` + quote(path) + `,"start":0,"end":1,"new_text":"x"}`, "start must be >= 1"},
		{"end before start", `{"path":` + quote(path) + `,"start":3,"end":2,"new_text":"x"}`, "end (2) must be >= start (3)"},
		{"end past last line", `{"path":` + quote(path) + `,"start":1,"end":4,"new_text":"x"}`, "end=4 is past the last line"},
		{"start past last line", `{"path":` + quote(path) + `,"start":4,"end":4,"new_text":"x"}`, "start=4 is past the last line"},
	}
	for _, c := range cases {
		if _, err := run(context.Background(), NewDispatcher(), LineReplaceTool, []byte(c.args)); err == nil || !strings.Contains(err.Error(), c.want) {
			t.Errorf("%s: err = %v, want containing %q", c.name, err, c.want)
		}
	}
	if _, err := run(context.Background(), NewDispatcher(), LineReplaceTool, []byte(`{"path":"/nonexistent/x","start":1,"end":1,"new_text":"x"}`)); err == nil || !strings.Contains(err.Error(), "no such file") {
		t.Errorf("missing file: err = %v", err)
	}
}

func TestLineInsertValidation(t *testing.T) {
	path := writeTemp(t, "a\nb\nc\n", 0o644)
	if _, err := run(context.Background(), NewDispatcher(), LineInsertTool, []byte(`{"path":`+quote(path)+`,"start":0,"new_text":"x"}`)); err == nil || !strings.Contains(err.Error(), "start must be >= 1") {
		t.Errorf("start below 1: err = %v", err)
	}
	if _, err := run(context.Background(), NewDispatcher(), LineInsertTool, []byte(`{"path":`+quote(path)+`,"start":5,"new_text":"x"}`)); err == nil || !strings.Contains(err.Error(), "past one past the last line") {
		t.Errorf("start past n+1: err = %v", err)
	}
	if _, err := run(context.Background(), NewDispatcher(), LineInsertTool, []byte(`{"path":"/nonexistent/x","start":1,"new_text":"x"}`)); err == nil || !strings.Contains(err.Error(), "no such file") {
		t.Errorf("missing file: err = %v", err)
	}
}

func TestLineReplacePreservesModeAndLeavesNoTemp(t *testing.T) {
	path := writeTemp(t, "a\nb\nc\n", 0o600)
	if _, err := run(context.Background(), NewDispatcher(), LineReplaceTool, []byte(`{"path":`+quote(path)+`,"start":2,"end":2,"new_text":"X\n"}`)); err != nil {
		t.Fatalf("Run: %v", err)
	}
	info, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	if perm := info.Mode().Perm(); perm != 0o600 {
		t.Errorf("mode = %v, want 0600 preserved", perm)
	}
	entries, err := os.ReadDir(filepath.Dir(path))
	if err != nil {
		t.Fatal(err)
	}
	for _, e := range entries {
		if strings.Contains(e.Name(), ".porter-edit-") {
			t.Errorf("temp file left behind: %s", e.Name())
		}
	}
}

func TestLineReplaceNoopLeavesFileUntouched(t *testing.T) {
	path := writeTemp(t, "a\nb\nc\n", 0o644)
	before, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := run(context.Background(), NewDispatcher(), LineReplaceTool, []byte(`{"path":`+quote(path)+`,"start":2,"end":2,"new_text":"b"}`)); err != nil {
		t.Fatalf("Run: %v", err)
	}
	after, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	if !after.ModTime().Equal(before.ModTime()) {
		t.Errorf("no-op edit rewrote the file (modtime changed)")
	}
}

func TestStringReplaceOnce(t *testing.T) {
	path := writeTemp(t, "hello world\n", 0o644)
	res, err := run(context.Background(), NewDispatcher(), StringReplace, []byte(`{"path":`+quote(path)+`,"old_text":"world","new_text":"there"}`))
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if got, want := readTemp(t, path), "hello there\n"; got != want {
		t.Errorf("content = %q, want %q", got, want)
	}
	if !strings.Contains(res, "before (old line numbers):") || !strings.Contains(res, "after (new line numbers):") {
		t.Errorf("echo missing before/after:\n%s", res)
	}
}

func TestStringReplaceEveryOccurrence(t *testing.T) {
	path := writeTemp(t, "a\nb a\nc a\n", 0o644)
	if _, err := run(context.Background(), NewDispatcher(), StringReplace, []byte(`{"path":`+quote(path)+`,"old_text":"a","new_text":"X","expected_count":3}`)); err != nil {
		t.Fatalf("Run: %v", err)
	}
	if got, want := readTemp(t, path), "X\nb X\nc X\n"; got != want {
		t.Errorf("content = %q, want %q", got, want)
	}
}

func TestStringReplaceCountMismatch(t *testing.T) {
	path := writeTemp(t, "a\nb a\nc\n", 0o644)
	_, err := run(context.Background(), NewDispatcher(), StringReplace, []byte(`{"path":`+quote(path)+`,"old_text":"a","new_text":"X","expected_count":1}`))
	if err == nil {
		t.Fatal("expected a count mismatch error")
	}
	if !strings.Contains(err.Error(), "found 2") {
		t.Errorf("error should report the real count: %v", err)
	}
	// The file must be unchanged after a rejected edit.
	if got := readTemp(t, path); got != "a\nb a\nc\n" {
		t.Errorf("rejected edit changed the file: %q", got)
	}
}

func TestStringReplaceValidation(t *testing.T) {
	path := writeTemp(t, "hello\n", 0o644)
	if _, err := run(context.Background(), NewDispatcher(), StringReplace, []byte(`{"path":`+quote(path)+`,"old_text":"","new_text":"x"}`)); err == nil || !strings.Contains(err.Error(), "must not be empty") {
		t.Errorf("empty old_text: err = %v", err)
	}
	if _, err := run(context.Background(), NewDispatcher(), StringReplace, []byte(`{"path":"/nonexistent/x","old_text":"a","new_text":"b"}`)); err == nil || !strings.Contains(err.Error(), "no such file") {
		t.Errorf("missing file: err = %v", err)
	}
}

func TestEditToolRelativeToRunDir(t *testing.T) {
	// Paths resolve against the run dir (like the shell's cwd), so a sandboxed
	// provider and the local dispatcher edit the same files.
	dir := t.TempDir()
	sub := filepath.Join(dir, "sub")
	if err := os.MkdirAll(sub, 0o755); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(sub, "f.txt")
	if err := os.WriteFile(path, []byte("a\nb\nc\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	d := NewDispatcher()
	res, err := runDir(context.Background(), d, ReadLinesTool, []byte(`{"path":"f.txt"}`), sub)
	if err != nil {
		t.Fatalf("read relative to dir: %v", err)
	}
	if !strings.Contains(res, "showing lines 1-3") {
		t.Errorf("read relative to dir = %s", res)
	}
	if _, err := runDir(context.Background(), d, LineReplaceTool, []byte(`{"path":"f.txt","start":2,"end":2,"new_text":"X\n"}`), sub); err != nil {
		t.Fatalf("edit relative to dir: %v", err)
	}
	if got := readTemp(t, path); got != "a\nX\nc\n" {
		t.Errorf("edit relative to dir content = %q", got)
	}
}

// quote returns the JSON string form of s for embedding in args.
func quote(s string) string {
	b, _ := json.Marshal(s)
	return string(b)
}

// runDir runs a tool against a dispatcher with an explicit working dir.
func runDir(ctx context.Context, p Provider, name string, args []byte, dir string) (string, error) {
	disp, ok := p.(*Dispatcher)
	if !ok {
		return "", errors.New("runDir needs a *Dispatcher")
	}
	stream, err := disp.RunDir(ctx, name, args, dir)
	if err != nil {
		return "", err
	}
	b, rerr := io.ReadAll(stream)
	_ = stream.Close()
	if rerr != nil {
		return "", rerr
	}
	return string(b), nil
}
