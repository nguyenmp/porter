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
	"github.com/yuin/goldmark/ast"
	"github.com/yuin/goldmark/extension"
	"github.com/yuin/goldmark/parser"
	"github.com/yuin/goldmark/text"
	"github.com/yuin/goldmark/util"
)

// md is the markdown engine used for assistant replies. goldmark handles its
// own HTML escaping, so its output is safe to insert into a page as-is.
var md = goldmark.New(
	goldmark.WithExtensions(extension.GFM),
	goldmark.WithParserOptions(parser.WithASTTransformers(
		util.Prioritized(imageLinker{}, 100),
	)),
)

// Markdown renders markdown text to HTML. On a parse error it falls back to
// escaped plaintext with line breaks rather than emitting partial markup.
func Markdown(s string) string {
	var buf bytes.Buffer
	if err := md.Convert([]byte(s), &buf); err != nil {
		return Plaintext(s)
	}
	return buf.String()
}

// imageLinker wraps each reply image in a link to the image file, so a click
// opens the full-size picture. The web UI caps every reply image at a fixed
// height, so the link is how a reader reaches the whole image. goldmark emits a
// bare <img> with no such link, so this adds one.
//
// The link points at the same URL as the image and opens in a new tab, so the
// chat stays where it is. An image the author already wrapped in a link keeps
// the author's destination and gets no second link.
type imageLinker struct{}

// Transform implements parser.ASTTransformer.
func (imageLinker) Transform(doc *ast.Document, _ text.Reader, _ parser.Context) {
	// Collect first, then wrap. Wrapping an image moves it in the tree, and
	// goldmark's ast.Walk iterates a node's children as it goes, so a wrap
	// during the walk would skip the images that follow in the same block.
	var images []*ast.Image
	_ = ast.Walk(doc, func(n ast.Node, entering bool) (ast.WalkStatus, error) {
		if !entering {
			return ast.WalkContinue, nil
		}
		img, ok := n.(*ast.Image)
		if !ok {
			return ast.WalkContinue, nil
		}
		if _, inLink := img.Parent().(*ast.Link); !inLink {
			images = append(images, img)
		}
		return ast.WalkContinue, nil
	})

	for _, img := range images {
		parent := img.Parent()
		if parent == nil {
			continue
		}
		link := ast.NewLink()
		link.Destination = img.Destination
		link.SetAttributeString("target", "_blank")
		link.SetAttributeString("rel", "noopener noreferrer")
		// Replace first, which detaches the image, then move the detached image
		// inside the link. AppendChild rejects a node that still has a parent.
		parent.ReplaceChild(parent, img, link)
		link.AppendChild(link, img)
	}
}

// Plaintext escapes s for HTML and converts newlines to <br>. It is used for
// user messages and tool output, where the source text is not markdown.
func Plaintext(s string) string {
	return strings.ReplaceAll(html.EscapeString(s), "\n", "<br>")
}
