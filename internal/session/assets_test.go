package session

import (
	"encoding/base64"
	"os"
	"strings"
	"testing"

	"porter/internal/asset"
)

// TestPublishAssetRoundTrip publishes a file through a session and reads it
// back through the store's URL validation: the URL is well-formed, the bytes
// land on disk under the session's folder, and the store resolves the URL
// segments back to exactly that file.
func TestPublishAssetRoundTrip(t *testing.T) {
	root := t.TempDir()
	st := NewStore(memPersister(t), nil)
	st.SetAssetRoot(root)
	s := newTestSession(t, "session_7")
	s.assetRoot = root

	content := []byte("PNG-bytes")
	url, err := s.publishAsset("../odd name.png", content)
	if err != nil {
		t.Fatalf("publishAsset: %v", err)
	}
	const prefix = "/assets/sessions/session_7/"
	if !strings.HasPrefix(url, prefix) {
		t.Fatalf("url = %q, want prefix %q", url, prefix)
	}
	rest := strings.TrimPrefix(url, prefix) // <asset_id>/<filename>
	parts := strings.SplitN(rest, "/", 2)
	if len(parts) != 2 || parts[1] != "odd_name.png" {
		t.Fatalf("url tail = %q, want <asset_id>/odd_name.png", rest)
	}
	path, err := st.AssetFilePath("session_7", parts[0], parts[1])
	if err != nil {
		t.Fatalf("AssetFilePath: %v", err)
	}
	got, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}
	if string(got) != string(content) {
		t.Errorf("asset bytes = %q, want %q", got, content)
	}
}

// TestAssetFilePathRejectsTraversal verifies the serving lookup refuses
// anything that is not a sanitized segment: dot segments, unknown sessions,
// and unknown assets all error instead of touching the filesystem.
func TestAssetFilePathRejectsTraversal(t *testing.T) {
	st := NewStore(memPersister(t), nil)
	st.SetAssetRoot(t.TempDir())
	for _, tc := range []struct{ id, assetID, name string }{
		{"session_7", "abc", ".."},
		{"session_7", "..", "x.png"},
		{"../etc", "abc", "x.png"},
		{"nope", "abc", "x.png"}, // session folder never created
	} {
		if _, err := st.AssetFilePath(tc.id, tc.assetID, tc.name); err == nil {
			t.Errorf("AssetFilePath(%q, %q, %q) succeeded, want error", tc.id, tc.assetID, tc.name)
		}
	}
}

// TestSanitizeFilename covers the name reduction: separators, spaces, and
// control characters become underscores, dot segments collapse, and empty
// results fall back to "asset" so the file always publishes under a usable
// name.
func TestSanitizeFilename(t *testing.T) {
	cases := map[string]string{
		"chart.png":       "chart.png",
		"../odd name.png": "odd_name.png",
		"a/b/c.html":      "c.html",
		"...":             "asset",
		"/":               "asset",
		"":                "asset",
	}
	for in, want := range cases {
		if got := asset.SanitizeFilename(in); got != want {
			t.Errorf("SanitizeFilename(%q) = %q, want %q", in, got, want)
		}
	}
}

// TestAssetDataURLRoundTrip publishes bytes and reads them back as an inline
// data URL. That is how a message that names an asset sends the picture to a
// model provider that cannot reach this server.
func TestAssetDataURLRoundTrip(t *testing.T) {
	root := t.TempDir()
	st := NewStore(memPersister(t), nil)
	st.SetAssetRoot(root)
	s := newTestSession(t, "session_7")
	s.assetRoot = root

	// The PNG signature alone is enough for http.DetectContentType to call it
	// an image, which is all the round trip needs.
	png := []byte("\x89PNG\r\n\x1a\nand then some pixels")
	url, err := s.publishAsset("shot.png", png)
	if err != nil {
		t.Fatalf("publishAsset: %v", err)
	}
	got, err := st.AssetDataURL(url)
	if err != nil {
		t.Fatalf("AssetDataURL: %v", err)
	}
	const prefix = "data:image/png;base64,"
	if !strings.HasPrefix(got, prefix) {
		t.Fatalf("data URL = %q, want prefix %q", got, prefix)
	}
	decoded, err := base64.StdEncoding.DecodeString(strings.TrimPrefix(got, prefix))
	if err != nil {
		t.Fatalf("data URL body is not base64: %v", err)
	}
	if string(decoded) != string(png) {
		t.Errorf("decoded bytes = %q, want %q", decoded, png)
	}
}

// TestAssetDataURLRejectsForeignURL checks that only a published asset's URL is
// accepted, so a location invented by a client cannot make the server read a
// file outside the asset root.
func TestAssetDataURLRejectsForeignURL(t *testing.T) {
	st := NewStore(memPersister(t), nil)
	st.SetAssetRoot(t.TempDir())
	for _, url := range []string{
		"/etc/passwd",
		"/assets/sessions/session_7/../../etc/passwd",
		"/assets/sessions/session_7/abc",
		"data:image/png;base64,AAAA",
	} {
		if _, err := st.AssetDataURL(url); err == nil {
			t.Errorf("AssetDataURL(%q) succeeded, want an error", url)
		}
	}
}
