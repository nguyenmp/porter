// Package asset defines the tools that move files the model creates in the
// execution environment onto the server, where the session can serve them:
// publish_asset uploads a file and returns its hosted URL (so the model can
// show it as an <img> in a reply), and render_iframe displays an HTML asset
// in a sandboxed iframe. publish_asset is the mirror image of spool_output:
// spool writes server-side bytes to the provider's disk; publish_asset reads
// a provider-side file and stores the bytes on the server, keyed by session.
// The read is executed by the active provider through a private provider tool
// (tools.AssetReadTool), so publish_asset works for any execution context —
// local sandbox or a remote host — and the file never has to stay on the
// machine that made it.
package asset

import (
	"encoding/json"
	"errors"
	"fmt"
	"path/filepath"
	"strconv"
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

// IframeTool is the model-facing name of the iframe rendering tool.
const IframeTool = "render_iframe"

// DefaultIframeHeight is the height in px the UI gives an iframe when the
// model does not pick one.
const DefaultIframeHeight = 480

// MaxIframeHeight caps the height so a careless call cannot request a
// viewport taller than the page it renders in.
const MaxIframeHeight = 2000

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
				"To show an HTML file as a live page instead of a static image, pass the URL to render_iframe — render_iframe displays HTML that publish_asset uploaded. " +
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

// IframeArgs is the parsed render_iframe call: the asset URL publish_asset
// returned, and an optional height in px.
type IframeArgs struct {
	Src    string `json:"src"`
	Height *int   `json:"height"`
}

// ParseIframeArgs validates a render_iframe call: the src must be a relative
// asset URL (an absolute or foreign URL would defeat the sandbox's purpose —
// the frame is only safe because the served HTML carries a sandboxing CSP).
func ParseIframeArgs(raw string) (IframeArgs, error) {
	var in IframeArgs
	if err := json.Unmarshal([]byte(raw), &in); err != nil {
		return in, fmt.Errorf("parse render_iframe arguments: %w", err)
	}
	if !strings.HasPrefix(in.Src, "/assets/sessions/") {
		return in, errors.New("render_iframe: src must be the URL publish_asset returned (a relative /assets/sessions/... URL)")
	}
	if in.Height != nil && (*in.Height < 100 || *in.Height > MaxIframeHeight) {
		return in, fmt.Errorf("render_iframe: height must be between 100 and %d px (got %d)", MaxIframeHeight, *in.Height)
	}
	return in, nil
}

// IframeDef is the model-facing definition of the render_iframe tool. It is
// served by the agent (it needs no execution provider): the call only records
// which asset to show, and the web UI draws the committed result as an
// expanded sandboxed iframe.
func IframeDef() llm.Tool {
	return llm.Tool{
		Type: "function",
		Function: llm.Function{
			Name: IframeTool,
			Description: "Display an HTML page you generated in the chat, in a sandboxed iframe that is always expanded. " +
				"render_iframe requires an asset you published with publish_asset first: call publish_asset with the path to an HTML file you created, " +
				"then pass the URL it returned as src. The page is shown in an iframe sandboxed with allow-scripts only (no same-origin), so its " +
				"scripts run but it cannot touch the rest of the chat or the site. height is optional, in px (default " +
				strconv.Itoa(DefaultIframeHeight) + ").",
			Parameters: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"src": map[string]any{
						"type":        "string",
						"description": "The /assets/sessions/... URL that publish_asset returned for the HTML file.",
					},
					"height": map[string]any{
						"type":        "integer",
						"description": "Optional height of the iframe in px (default " + strconv.Itoa(DefaultIframeHeight) + ").",
					},
				},
				"required": []string{"src"},
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
