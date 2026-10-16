package session

import (
	"context"
	"crypto/rand"
	"encoding/base64"
	"encoding/hex"
	"errors"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"strings"

	"porter/internal/asset"
	"porter/internal/imagegen"
)

// SetAssetRoot configures where published assets are stored: every file the
// publish_asset tool moves to the server lands under <root>/<session id>/ and
// is served back from there by the /assets route. The server calls this once
// at startup, beside SetSandboxRoot, before Load; tests and embedders that
// opt out never call it, and their publish_asset calls fail with a clear
// error instead of silently dropping bytes.
func (st *Store) SetAssetRoot(root string) {
	st.assetRoot = root
}

// SetImageClient configures the image model the generate_image tool calls. The
// server calls this once at startup, beside SetAssetRoot, before Load. A nil
// client leaves the tool with no model, and a call says so plainly instead of
// failing obscurely.
func (st *Store) SetImageClient(c *imagegen.Client) {
	st.images = c
}

// configureSession copies the store's session-scoped asset and image settings
// onto a session: where published assets live, how to read one back by its URL,
// and which image model generate_image should call. It is called whenever the
// store creates or loads a session, so the two paths cannot drift apart.
func (st *Store) configureSession(s *Session) {
	s.assetRoot = st.assetRoot
	s.images = st.images
	s.readAsset = st.AssetBytes
	// The sidebar watchers live on the store; point the session at them so a
	// turn starting or ending can wake them (see notifyListChanged).
	s.notifyChange = st.notifyListChanged
}

// assetDirFor returns the asset folder for a session id ("" when no asset
// root is configured). The folder name is the session's public id, matching
// the sandbox layout, so assets are discoverable at restart with no lookup.
func (st *Store) assetDirFor(id string) string {
	if st.assetRoot == "" {
		return ""
	}
	return filepath.Join(st.assetRoot, id)
}

// PublishAsset stores one asset for a session and returns its hosted URL. It
// is the store-level entry point (the session's publish hook wraps the same
// write with the session's own root), used by callers that hold the store —
// e.g. tests — instead of a live session.
func (st *Store) PublishAsset(id, filename string, content []byte) (string, error) {
	root := st.assetRoot
	if root == "" {
		return "", errors.New("assets are not configured on this server")
	}
	return writeAsset(root, id, filename, content)
}

// writeAsset stores one published asset under the asset root and returns its
// hosted URL. root is the store's asset root (the session folder is appended
// here, so callers pass either the store root or the session's recorded copy
// of it); filename is sanitized to a safe single path segment; the asset id
// is fresh random hex, so each publish gets its own immutable URL and a
// republish of the same file never collides. The URL is relative to the
// porter origin, which is exactly what the model should embed in a reply: it
// works from any device that can reach the server, with no CORS involved.
func writeAsset(root, id, filename string, content []byte) (string, error) {
	name := asset.SanitizeFilename(filename)
	var b [8]byte
	if _, err := rand.Read(b[:]); err != nil {
		return "", fmt.Errorf("generate asset id: %w", err)
	}
	assetID := hex.EncodeToString(b[:])
	dir := filepath.Join(root, id, assetID)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return "", fmt.Errorf("create asset folder: %w", err)
	}
	if err := os.WriteFile(filepath.Join(dir, name), content, 0o644); err != nil {
		return "", fmt.Errorf("write asset: %w", err)
	}
	return fmt.Sprintf("/assets/sessions/%s/%s/%s", id, assetID, name), nil
}

// safeSegment reports whether s is a safe single URL path segment: it matches
// its sanitized form (letters, digits, . - _), so a request can never smuggle
// separators or dot segments into the filesystem path.
func safeSegment(s string) bool {
	return s != "" && s == asset.SanitizeFilename(s)
}

// AssetFilePath validates a published asset's URL segments and returns the
// file's absolute path on disk, or an error when the session, asset id, or
// file is unknown. The server's /assets handler calls it so the lookup and
// the path construction live in one place, next to the store that wrote the
// file.
func (st *Store) AssetFilePath(id, assetID, filename string) (string, error) {
	root := st.assetDirFor(id)
	if root == "" {
		return "", errors.New("assets are not configured on this server")
	}
	if !safeSegment(id) || !safeSegment(assetID) || !safeSegment(filename) {
		return "", fmt.Errorf("invalid asset path")
	}
	path := filepath.Join(root, assetID, filename)
	// The sanitized segments cannot contain separators or dot segments, so
	// path is already confined to the asset root; the Rel check is a backstop
	// so a future change to one of the checks cannot open a hole.
	if rel, err := filepath.Rel(root, path); err != nil || rel == ".." || len(rel) > 2 && rel[:3] == "../" {
		return "", fmt.Errorf("invalid asset path")
	}
	info, err := os.Stat(path)
	if err != nil || info.IsDir() {
		return "", fmt.Errorf("asset not found")
	}
	return path, nil
}

// publishAsset is the Session's hook for the agent's publish_asset tool: it
// stores the bytes under the session's asset folder and returns the URL.
func (s *Session) publishAsset(filename string, content []byte) (string, error) {
	if s.assetRoot == "" {
		return "", errors.New("assets are not configured on this server")
	}
	return writeAsset(s.assetRoot, s.id, filename, content)
}

// AssetDataURL returns a published asset's bytes as an inline data URL, given
// the hosted URL a message carries: "/assets/sessions/<session>/<id>/<file>".
// A message names an image by that URL, but the model provider cannot reach the
// porter server, so the bytes have to travel inside the request. The media type
// comes from the bytes themselves, the same way an upload is checked.
func (st *Store) AssetDataURL(url string) (string, error) {
	_, data, err := st.AssetBytes(url)
	if err != nil {
		return "", err
	}
	mime := http.DetectContentType(data)
	return "data:" + mime + ";base64," + base64.StdEncoding.EncodeToString(data), nil
}

// AssetBytes reads a published asset's bytes, given the hosted URL a message
// carries: "/assets/sessions/<session>/<id>/<file>". It returns the file name
// and the bytes.
//
// The URL is untrusted (it can come from a client or from the model), so every
// segment is validated by AssetFilePath before any file is opened.
func (st *Store) AssetBytes(url string) (string, []byte, error) {
	rest, ok := strings.CutPrefix(url, asset.URLPrefix)
	if !ok {
		return "", nil, fmt.Errorf("not a published asset URL: %q", url)
	}
	parts := strings.Split(rest, "/")
	if len(parts) != 3 {
		return "", nil, fmt.Errorf("malformed asset URL: %q", url)
	}
	path, err := st.AssetFilePath(parts[0], parts[1], parts[2])
	if err != nil {
		return "", nil, err
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return "", nil, fmt.Errorf("read asset: %w", err)
	}
	return parts[2], data, nil
}

// generateImage is the Session's hook for the agent's generate_image tool: it
// reads any input pictures out of the session's own assets, asks the image
// model for a new picture, and stores the result as a new asset.
//
// The model never carries the picture bytes: inputs are named by asset URL on
// the way in, and the answer is an asset URL on the way out. That keeps a
// generation off the model's context and inside the server's own storage.
func (s *Session) generateImage(ctx context.Context, prompt string, images []string) (string, error) {
	if s.images == nil {
		return "", errors.New("generate_image: no image model is configured on this server (set PORTER_IMAGE_MODEL)")
	}
	if s.assetRoot == "" || s.readAsset == nil {
		return "", errors.New("generate_image: no asset store is configured on this server, so the picture could not be saved")
	}
	inputs := make([]imagegen.Input, 0, len(images))
	for _, url := range images {
		name, data, err := s.readAsset(url)
		if err != nil {
			return "", fmt.Errorf("generate_image: input picture %s: %w", url, err)
		}
		inputs = append(inputs, imagegen.Input{Name: name, Data: data})
	}
	result, err := s.images.Generate(ctx, prompt, inputs)
	if err != nil {
		return "", fmt.Errorf("generate_image: %w", err)
	}
	return s.publishAsset(generatedName(result.MediaType), result.Data)
}

// generatedName names a stored picture from the media type the provider
// reported, so it serves back with the right type. The type is often missing,
// and png is what the image models return by default.
func generatedName(mediaType string) string {
	switch mediaType {
	case "image/jpeg":
		return "generated.jpg"
	case "image/webp":
		return "generated.webp"
	case "image/gif":
		return "generated.gif"
	default:
		return "generated.png"
	}
}
