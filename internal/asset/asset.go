// Package asset defines the tools that move files between the execution
// environment and the server: publish_asset uploads a file and returns its
// hosted URL (so the model can show it as an <img> in a reply), read_image
// reads a picture and hands it to the model to look at, and render_iframe
// displays an HTML asset in a sandboxed iframe. publish_asset and read_image
// are the mirror image of spool_output: spool writes server-side bytes to the
// provider's disk; they read a provider-side file and store the bytes on the
// server, keyed by session. The read is executed by the active provider through
// a private provider tool (tools.AssetReadTool), so both work for any execution
// context — local sandbox or a remote host — and the file never has to stay on
// the machine that made it.
package asset

import (
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
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

// URLPrefix is the path prefix of a published asset's hosted URL:
// /assets/sessions/<session id>/<asset id>/<file name>. It is the one shape
// that names an asset, so a message can carry the URL and the server can find
// the bytes again from it.
const URLPrefix = "/assets/sessions/"

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
				"First create the file with shell or another rendering tool in the working directory (for example a PNG from a screenshot, mermaid-cli, or matplotlib, or an HTML file you generated). " +
				"Then call publish_asset with the file's path (relative to the working directory, or absolute). " +
				"The file is uploaded to the server, and the result is its hosted URL. That URL is a path from the base URL. Use it as-is; you don't need to add a prefix, and it works. The URL stays available as long as the session exists. " +
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

// ReadImageTool is the model-facing name of the tool that reads a picture from
// the execution environment so the model can look at it. Like publish_asset it
// is served by the agent: the file read runs on the active execution provider
// (the same private _porter_asset_read tool), and the bytes are published as an
// asset. Unlike publish_asset the picture is also handed to the model.
const ReadImageTool = "read_image"

// ParseReadImageArgs validates a read_image call and returns the file path to
// read. The path is resolved by the provider against its working directory,
// exactly like the file tools, so a relative path names a picture next to the
// ones shell and the editors touch.
func ParseReadImageArgs(raw string) (string, error) {
	var in publishArgs
	if err := json.Unmarshal([]byte(raw), &in); err != nil {
		return "", fmt.Errorf("parse read_image arguments: %w", err)
	}
	if strings.TrimSpace(in.Path) == "" {
		return "", errors.New("read_image: path is required (the picture to read, relative to the working directory or absolute)")
	}
	return in.Path, nil
}

// ReadImageDef is the model-facing definition of the read_image tool.
func ReadImageDef() llm.Tool {
	return llm.Tool{
		Type: "function",
		Function: llm.Function{
			Name: ReadImageTool,
			Description: "Look at a picture that is already a file in the execution environment (a screenshot, a chart, a photo). " +
				"Pass its path, relative to the working directory or absolute. " +
				"The picture is saved as an asset and attached to your context, so you see it in your next turn; the result is the asset's hosted URL. " +
				"You cannot see the picture in the same turn you ask for it — call this, then answer once the picture arrives. " +
				"To show a picture to the user instead of only looking at it yourself, use publish_asset.",
			Parameters: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"path": map[string]any{
						"type":        "string",
						"description": "The picture file to read: relative to the working directory, or absolute. Must be a png, jpeg, gif, or webp.",
					},
				},
				"required": []string{"path"},
			},
		},
	}
}

// GenerateImageTool is the model-facing name of the tool that makes a picture
// from a prompt, optionally guided by input pictures.
const GenerateImageTool = "generate_image"

// GenerateImageArgs is the parsed generate_image call: the prompt, and any
// input pictures named by their asset URLs.
type GenerateImageArgs struct {
	Prompt string   `json:"prompt"`
	Images []string `json:"images"`
}

// ParseGenerateImageArgs validates a generate_image call. A prompt is required;
// the input pictures are asset URLs the server already holds, so the bytes
// never have to travel through the model.
func ParseGenerateImageArgs(raw string) (GenerateImageArgs, error) {
	var in GenerateImageArgs
	if err := json.Unmarshal([]byte(raw), &in); err != nil {
		return GenerateImageArgs{}, fmt.Errorf("parse generate_image arguments: %w", err)
	}
	if strings.TrimSpace(in.Prompt) == "" {
		return GenerateImageArgs{}, errors.New("generate_image: prompt is required (describe the picture you want)")
	}
	for _, u := range in.Images {
		if !strings.HasPrefix(u, URLPrefix) {
			return GenerateImageArgs{}, fmt.Errorf("generate_image: input picture %q must be an asset URL (a %s path), not a file path", u, URLPrefix)
		}
	}
	return in, nil
}

// GenerateImageDef is the model-facing definition of the generate_image tool.
func GenerateImageDef() llm.Tool {
	return llm.Tool{
		Type: "function",
		Function: llm.Function{
			Name: GenerateImageTool,
			Description: "Make a picture from a text prompt. " +
				"Pass input pictures (as asset URLs, the same /assets/sessions/... paths publish_asset and read_image return) to guide the result or to edit one of them; with no input pictures you get a fresh picture. " +
				"The new picture is saved as an asset, and the result is its hosted URL. " +
				"To show it to the user, reference that URL in your reply the usual markdown way: ![alt text](URL). " +
				"To change a picture you already made, call this again and pass its asset URL as an input picture.",
			Parameters: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"prompt": map[string]any{
						"type":        "string",
						"description": "What the picture should show. Be specific: subject, style, colors, framing, and any text to render.",
					},
					"images": map[string]any{
						"type":        "array",
						"items":       map[string]any{"type": "string"},
						"description": "Optional input pictures, as asset URLs, to guide or edit. Omit for a brand new picture.",
					},
				},
				"required": []string{"prompt"},
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

// ImageMimeType returns the media type of picture bytes, or "" when the bytes
// are not a raster image the model providers accept. The type comes from the
// bytes themselves, never from a file name, so a text file called .png is not
// an image. SVG is left out on purpose: it is a script-carrying document
// rather than a picture.
func ImageMimeType(data []byte) string {
	mime := http.DetectContentType(data)
	if IsImageMimeType(mime) {
		return mime
	}
	return ""
}

// IsImageMimeType reports whether a media type is one the model providers
// accept as an image input.
func IsImageMimeType(mime string) bool {
	switch mime {
	case "image/png", "image/jpeg", "image/gif", "image/webp":
		return true
	default:
		return false
	}
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
