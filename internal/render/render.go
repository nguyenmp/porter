// Package render owns how message text becomes HTML for the web UI. It is the
// single source of truth for the committed render path — the initial history
// render (GET /view) and the message_committed envelopes the live SSE stream
// swaps in when a message finishes — so the two can never drift apart. The
// live in-progress view is rendered client-side with markdown-it (see
// index.tmpl) and is deliberately transient: the server-rendered committed
// copy replaces it on commit. Assistant replies are markdown (rendered with
// goldmark); user messages and tool output are escaped plaintext with <br>
// line breaks.
package render

import (
	"bytes"
	"html"
	"strings"

	"github.com/yuin/goldmark"
	"github.com/yuin/goldmark/extension"
)

// md is the markdown engine used for assistant replies. goldmark handles its
// own HTML escaping, so its output is safe to insert into a page as-is.
var md = goldmark.New(goldmark.WithExtensions(extension.GFM))

// Markdown renders markdown text to HTML. On a parse error it falls back to
// escaped plaintext with line breaks rather than emitting partial markup.
func Markdown(s string) string {
	var buf bytes.Buffer
	if err := md.Convert([]byte(s), &buf); err != nil {
		return Plaintext(s)
	}
	return buf.String()
}

// Plaintext escapes s for HTML and converts newlines to <br>. It is used for
// user messages and tool output, where the source text is not markdown.
func Plaintext(s string) string {
	return strings.ReplaceAll(html.EscapeString(s), "\n", "<br>")
}
