package session

import (
	"context"
	"encoding/base64"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"

	"porter/internal/asset"
	"porter/internal/imagegen"
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

// TestGenerateImageStoresResult covers the session's generate_image hook from
// end to end against a stub image server: the input picture is read out of the
// session's own assets, the prompt and the picture reach the model, and the
// picture it returns is stored as a new asset.
func TestGenerateImageStoresResult(t *testing.T) {
	st := NewStore(memPersister(t), nil)
	st.SetAssetRoot(t.TempDir())

	var gotPath, gotPrompt, gotFile string
	imgSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		if err := r.ParseMultipartForm(1 << 20); err != nil {
			t.Errorf("parse multipart: %v", err)
			return
		}
		gotPrompt = r.FormValue("prompt")
		f, _, err := r.FormFile("image")
		if err != nil {
			t.Errorf("no image field: %v", err)
			return
		}
		b, _ := io.ReadAll(f)
		f.Close()
		gotFile = string(b)
		b64 := base64.StdEncoding.EncodeToString([]byte("\x89PNG\r\n\x1a\nmade"))
		fmt.Fprintf(w, `{"data":[{"b64_json":%q,"media_type":"image/png"}]}`, b64)
	}))
	defer imgSrv.Close()
	st.SetImageClient(imagegen.NewClient(imgSrv.URL, "k", "img-model", imgSrv.Client()))

	s := newTestSession(t, "session_7")
	st.configureSession(s)

	baseURL, err := s.publishAsset("base.png", []byte("\x89PNG\r\n\x1a\nbase"))
	if err != nil {
		t.Fatalf("publishAsset: %v", err)
	}
	url, err := s.generateImage(context.Background(), "make it blue", []string{baseURL})
	if err != nil {
		t.Fatalf("generateImage: %v", err)
	}
	if gotPath != "/images/edits" {
		t.Errorf("path = %q, want /images/edits (the shape that carries pictures)", gotPath)
	}
	if gotPrompt != "make it blue" {
		t.Errorf("prompt = %q, want make it blue", gotPrompt)
	}
	if gotFile != "\x89PNG\r\n\x1a\nbase" {
		t.Errorf("input picture = %q, want the base asset's bytes", gotFile)
	}
	if !strings.HasPrefix(url, "/assets/sessions/session_7/") {
		t.Errorf("url = %q, want a published asset URL", url)
	}
	_, data, err := st.AssetBytes(url)
	if err != nil {
		t.Fatalf("AssetBytes: %v", err)
	}
	if string(data) != "\x89PNG\r\n\x1a\nmade" {
		t.Errorf("stored picture = %q, want the model's bytes", data)
	}
}

// TestGenerateImageTextOnlyUsesGenerationsRoute checks that a prompt with no
// input picture goes to the plain generation route, not the edit route.
func TestGenerateImageTextOnlyUsesGenerationsRoute(t *testing.T) {
	st := NewStore(memPersister(t), nil)
	st.SetAssetRoot(t.TempDir())

	var gotPath, gotContentType string
	imgSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		gotContentType = r.Header.Get("Content-Type")
		b64 := base64.StdEncoding.EncodeToString([]byte("\x89PNG\r\n\x1a\nfresh"))
		fmt.Fprintf(w, `{"data":[{"b64_json":%q}]}`, b64)
	}))
	defer imgSrv.Close()
	st.SetImageClient(imagegen.NewClient(imgSrv.URL, "k", "img-model", imgSrv.Client()))

	s := newTestSession(t, "session_7")
	st.configureSession(s)

	url, err := s.generateImage(context.Background(), "a red apple", nil)
	if err != nil {
		t.Fatalf("generateImage: %v", err)
	}
	if gotPath != "/images/generations" {
		t.Errorf("path = %q, want /images/generations", gotPath)
	}
	if !strings.HasPrefix(gotContentType, "application/json") {
		t.Errorf("content type = %q, want JSON", gotContentType)
	}
	// The type is missing from the reply, so the name falls back to png.
	if !strings.HasSuffix(url, "/generated.png") {
		t.Errorf("url = %q, want a generated.png name", url)
	}
}

// TestGenerateImageWithoutModel checks the message when the server has no image
// model: it names the setting rather than failing obscurely.
func TestGenerateImageWithoutModel(t *testing.T) {
	st := NewStore(memPersister(t), nil)
	st.SetAssetRoot(t.TempDir())
	s := newTestSession(t, "session_7")
	st.configureSession(s)

	_, err := s.generateImage(context.Background(), "a red apple", nil)
	if err == nil {
		t.Fatal("generateImage succeeded with no image model")
	}
	if !strings.Contains(err.Error(), "PORTER_IMAGE_MODEL") {
		t.Errorf("error = %q, want it to name the setting to change", err)
	}
}
