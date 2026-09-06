package tools

import (
	"encoding/json"
	"io"
	"os"
	"path/filepath"
	"strconv"
	"testing"
)

func TestRunSpoolWriteDir(t *testing.T) {
	dir := t.TempDir()
	content := "line1\nline2\nexit code: 0\n" // raw mirror: trailer kept verbatim
	payload := map[string]string{"path": ".porter/out/call_1", "content": content}
	args, err := json.Marshal(payload)
	if err != nil {
		t.Fatal(err)
	}
	rc, err := runSpoolWriteDir(args, dir)
	if err != nil {
		t.Fatalf("runSpoolWriteDir: %v", err)
	}
	confirm, _ := io.ReadAll(rc)
	want := "wrote " + strconv.Itoa(len(content)) + " bytes to " + filepath.Join(dir, ".porter/out/call_1")
	if string(confirm) != want {
		t.Errorf("confirmation = %q, want %q", confirm, want)
	}
	got, err := os.ReadFile(filepath.Join(dir, ".porter/out/call_1"))
	if err != nil {
		t.Fatalf("spooled file not written: %v", err)
	}
	if string(got) != content {
		t.Errorf("spooled content = %q, want the raw bytes %q", got, content)
	}
}

func TestRunSpoolWriteDirExplicitAbsolutePath(t *testing.T) {
	dir := t.TempDir()
	target := filepath.Join(dir, "deep", "nested", "out.json") // parents created
	payload := map[string]string{"path": target, "content": "{\"a\":1}"}
	args, _ := json.Marshal(payload)
	rc, err := runSpoolWriteDir(args, dir)
	if err != nil {
		t.Fatalf("runSpoolWriteDir: %v", err)
	}
	_ = rc.Close()
	if _, err := os.Stat(target); err != nil {
		t.Errorf("explicit path not written: %v", err)
	}
}

func TestRunSpoolWriteDirEmptyPath(t *testing.T) {
	dir := t.TempDir()
	args, _ := json.Marshal(map[string]string{"path": "", "content": "x"})
	if _, err := runSpoolWriteDir(args, dir); err == nil {
		t.Errorf("empty path: want error")
	}
}

// TestDispatcherServesSpoolWrite routes the private tool through RunDir, the
// path every execution site (local, remote, host sandbox) shares.
func TestDispatcherServesSpoolWrite(t *testing.T) {
	dir := t.TempDir()
	d := NewDispatcher()
	payload := map[string]string{"path": "spooled.txt", "content": "bytes"}
	args, _ := json.Marshal(payload)
	rc, err := d.RunDir(t.Context(), SpoolWriteTool, args, dir)
	if err != nil {
		t.Fatalf("RunDir(%s): %v", SpoolWriteTool, err)
	}
	_ = rc.Close()
	got, err := os.ReadFile(filepath.Join(dir, "spooled.txt"))
	if err != nil || string(got) != "bytes" {
		t.Errorf("spooled file via dispatcher = %q, %v; want bytes", got, err)
	}
}
