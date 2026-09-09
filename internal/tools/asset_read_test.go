package tools

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestAssetReadRoundTrip publishes binary bytes through the private read tool:
// the payload must come back byte-identical, base64-wrapped in the JSON
// envelope the agent decodes.
func TestAssetReadRoundTrip(t *testing.T) {
	dir := t.TempDir()
	content := []byte{0x89, 'P', 'N', 'G', 0x0d, 0x0a, 0x1a, 0x0a, 0x00, 0x01, 0xff} // binary, NULs included
	if err := os.WriteFile(filepath.Join(dir, "chart.png"), content, 0o644); err != nil {
		t.Fatalf("write: %v", err)
	}
	payload, _ := json.Marshal(map[string]string{"path": "chart.png"})
	stream, err := NewDispatcher().RunDir(context.Background(), AssetReadTool, payload, dir)
	if err != nil {
		t.Fatalf("RunDir: %v", err)
	}
	defer stream.Close()
	var b bytes.Buffer
	if _, err := b.ReadFrom(stream); err != nil {
		t.Fatalf("read stream: %v", err)
	}
	var rd struct {
		Path string `json:"path"`
		Size int    `json:"size"`
		Data string `json:"data"`
	}
	if err := json.Unmarshal(b.Bytes(), &rd); err != nil {
		t.Fatalf("unmarshal response %q: %v", b.String(), err)
	}
	if rd.Path != filepath.Join(dir, "chart.png") {
		t.Errorf("path = %q, want %q", rd.Path, filepath.Join(dir, "chart.png"))
	}
	got, err := base64.StdEncoding.DecodeString(rd.Data)
	if err != nil {
		t.Fatalf("decode base64: %v", err)
	}
	if rd.Size != len(content) || !bytes.Equal(got, content) {
		t.Errorf("bytes did not round-trip: size=%d len=%d", rd.Size, len(content))
	}
}

// TestAssetReadErrors covers the tool's refusal paths: a missing file and a
// directory both fail to start (with a message naming publish_asset, so the
// model sees why its call failed), never streaming empty content.
func TestAssetReadErrors(t *testing.T) {
	d := NewDispatcher()
	if _, err := d.RunDir(context.Background(), AssetReadTool, []byte(`{"path":"nope.png"}`), t.TempDir()); err == nil || !strings.Contains(err.Error(), "no such file") {
		t.Errorf("missing file error = %v, want a no-such-file error", err)
	}
	dir := t.TempDir()
	if _, err := d.RunDir(context.Background(), AssetReadTool, []byte(`{"path":"."}`), dir); err == nil || !strings.Contains(err.Error(), "is a directory") {
		t.Errorf("directory error = %v, want a directory error", err)
	}
}
