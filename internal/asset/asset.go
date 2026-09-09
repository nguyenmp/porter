// Package asset defines the publish_asset tool: it moves a file out of the
// execution environment and onto the server, where the session can serve it,
// and returns the file's hosted URL so the model can show it in a reply
// (an <img> for a picture, render_iframe for HTML). It is the mirror image of
// spool_output: spool writes server-side bytes to the provider's disk;
// publish_asset reads a provider-side file and stores the bytes on the
// server, keyed by session. The read itself is executed by the active
// provider through a private provider tool (tools.AssetReadTool), so
// publish_asset works for any execution context — local sandbox or a remote
// host — and the file never has to stay on the machine that made it.
package asset

import (
	"encoding/json"
	"errors"
	"fmt"
	"path/filepath"
	"strings"

	"porter/internal/llm"
	"porter/internal/tools"
)

// PublishTool is the model-facing name of the publish tool.
const PublishTool = "publish_asset"

// MaxBytes is the largest single asset publish_asset accepts. It mirrors the
// provider-side cap (tools.AssetMaxBytes) that guards the read; the agent
// re-checks after decoding so the two ends of the private channel agree.
const MaxBytes = tools.AssetMaxBytes

// publishArgs is the parsed model-facing publish_asset call.
type publishArgs struct {
	Path string `json:"path"`
}

// ParseArgs validates a publish_asset call and returns the file path to read.
// The path is resolved by the provider against its working directory, exactly
// like the file tools, so a relative path names a file next to the ones shell
// and the editors touch.
func ParseArgs(raw string) (string, error) {
	var in publishArgs
	if err := json.Unmarshal([]byte(raw), &in); err != nil {
		return "", fmt.Errorf("parse publish_asset arguments: %w", err)
	}
	if strings.TrimSpace(in.Path) == "" {
		return "", errors.New("publish_asset: path is required (the file to publish, relative to the working directory or absolute)")
	}
	return in.Path, nil
}

// ReadPayload renders the arguments for the provider's private read tool
// (tools.AssetReadTool). Marshaling cannot fail for these fields.
func ReadPayload(path string) []byte {
	b, _ := json.Marshal(publishArgs{Path: path})
	return b
}

// Def is the model-facing definition of the publish_asset tool. The agent
// declares it on every request alongside recall_tool_output and spool_output.
func Def() llm.Tool {
	return llm.Tool{
		Type: "function",
		Function: llm.Function{
			Name: PublishTool,
			Description: "Publish a file from the execution environment to a hosted URL, so you can show it in your reply. " +
				"First create the file with shell or another rendering tool in the working directory (e.g. a PNG from a screenshot, mermaid-cli, or matplotlib, or an HTML file you generated). " +
				"Then call publish_asset with the file's path (relative to the working directory, or absolute). " +
				"The file is uploaded to the server and the result is its hosted URL, which stays available as long as the session exists. " +
				"To show the file in your reply, reference the URL as an image: ![alt text](URL) or <img src=\"URL\">. " +
				"Content type is guessed from the file extension (png, jpg, gif, webp, svg, html, pdf, and more).",
			Parameters: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"path": map[string]any{
						"type":        "string",
						"description": "The file to publish: relative to the working directory, or absolute.",
					},
				},
				"required": []string{"path"},
			},
		},
	}
}

// SanitizeFilename reduces a file name to a safe single path segment for the
// asset URL: no separators, no dot segments, only letters, digits, and
// . - _. An empty result becomes "asset", so a file with no usable name still
// publishes under a stable name.
func SanitizeFilename(name string) string {
	name = filepath.Base(name)
	var b strings.Builder
	for _, r := range name {
		switch {
		case r >= 'a' && r <= 'z', r >= 'A' && r <= 'Z', r >= '0' && r <= '9', r == '.', r == '-', r == '_':
			b.WriteRune(r)
		default:
			b.WriteByte('_')
		}
	}
	out := strings.Trim(b.String(), "._")
	if out == "" || out == ".." {
		return "asset"
	}
	if len(out) > 80 {
		// Keep the tail, which usually holds the extension.
		out = out[len(out)-80:]
	}
	return out
}

// MimeType returns the Content-Type for a published asset by extension. The
// map is deliberately small and safe: unknown types serve as
// application/octet-stream rather than guessing, and the server always sends
// X-Content-Type-Options: nosniff so a browser never sniffs past it.
func MimeType(name string) string {
	ext := strings.ToLower(filepath.Ext(name))
	switch ext {
	case ".png":
		return "image/png"
	case ".jpg", ".jpeg":
		return "image/jpeg"
	case ".gif":
		return "image/gif"
	case ".webp":
		return "image/webp"
	case ".avif":
		return "image/avif"
	case ".svg":
		return "image/svg+xml"
	case ".html", ".htm":
		return "text/html; charset=utf-8"
	case ".pdf":
		return "application/pdf"
	case ".txt", ".log":
		return "text/plain; charset=utf-8"
	case ".css":
		return "text/css; charset=utf-8"
	case ".csv":
		return "text/csv; charset=utf-8"
	case ".json":
		return "application/json"
	case ".xml":
		return "application/xml"
	default:
		return "application/octet-stream"
	}
}
