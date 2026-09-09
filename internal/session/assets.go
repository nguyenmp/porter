package session

import (
	"crypto/rand"
	"encoding/hex"
	"errors"
	"fmt"
	"os"
	"path/filepath"

	"porter/internal/asset"
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
