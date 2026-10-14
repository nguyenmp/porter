package render

import (
	"strings"
	"testing"
)

func TestMarkdownRendersGFM(t *testing.T) {
	got := Markdown("**bold**\n\n# Heading\n\n- item\n\n> quote\n\n| a | b |\n|---|---|\n| 1 | 2 |")
	for _, want := range []string{"<strong>bold</strong>", "<h1>Heading</h1>", "<li>item</li>", "<blockquote>", "<table>"} {
		if !strings.Contains(got, want) {
			t.Errorf("Markdown output missing %q:\n%s", want, got)
		}
	}
	if strings.Contains(got, "**bold**") {
		t.Errorf("Markdown left literal markdown markers in output:\n%s", got)
	}
}

func TestMarkdownEscapesRawHTML(t *testing.T) {
	// goldmark omits raw HTML by default (it is not configured with the raw
	// HTML extension), so a <script> tag must never reach the page verbatim.
	got := Markdown("hello <script>alert(1)</script>")
	if strings.Contains(got, "<script>") {
		t.Errorf("Markdown leaked raw HTML: %s", got)
	}
	if !strings.Contains(got, "raw HTML omitted") {
		t.Errorf("Markdown should have stripped the raw HTML block: %s", got)
	}
}

func TestMarkdownEmpty(t *testing.T) {
	if got := Markdown(""); got != "" {
		t.Errorf("Markdown(\"\") = %q, want empty", got)
	}
}

func TestPlaintextEscapesAndBreaks(t *testing.T) {
	got := Plaintext("a & b\nline2 <b>")
	if !strings.Contains(got, "a &amp; b") || !strings.Contains(got, "&lt;b&gt;") {
		t.Errorf("Plaintext did not escape HTML: %s", got)
	}
	if !strings.Contains(got, "<br>line2") {
		t.Errorf("Plaintext did not convert newline to <br>: %s", got)
	}
}

// TestMarkdownLinksReplyImageToFullSize checks that a reply image is wrapped in
// a link to the image file, opening in a new tab, so the reader can reach the
// full-size picture that the web UI crops to a fixed height.
func TestMarkdownLinksReplyImageToFullSize(t *testing.T) {
	got := Markdown("![shot](/assets/sessions/s/shot.png)")
	want := `<a href="/assets/sessions/s/shot.png" target="_blank" rel="noopener noreferrer">` +
		`<img src="/assets/sessions/s/shot.png" alt="shot"></a>`
	if !strings.Contains(got, want) {
		t.Errorf("Markdown output missing %q:\n%s", want, got)
	}
}

// TestMarkdownLinksEveryImageInABlock checks that each image in one paragraph
// gets its own link. Wrapping an image moves it in the syntax tree, so a naive
// single pass would stop early and leave the later images unlinked.
func TestMarkdownLinksEveryImageInABlock(t *testing.T) {
	got := Markdown("![one](1.png) ![two](2.png)")
	if n := strings.Count(got, "<a href="); n != 2 {
		t.Errorf("Markdown wrapped %d of 2 images:\n%s", n, got)
	}
	for _, want := range []string{`href="1.png"`, `href="2.png"`, `src="1.png"`, `src="2.png"`} {
		if !strings.Contains(got, want) {
			t.Errorf("Markdown output missing %q:\n%s", want, got)
		}
	}
}

// TestMarkdownKeepsAuthorLinkAroundImage checks that an image the author
// already wrapped in a link keeps that destination, with no second link and no
// forced new tab on top of the author's choice.
func TestMarkdownKeepsAuthorLinkAroundImage(t *testing.T) {
	got := Markdown("[![shot](img.png)](https://example.com/page)")
	if n := strings.Count(got, "<a href="); n != 1 {
		t.Errorf("Markdown wrapped the image again; %d links:\n%s", n, got)
	}
	if !strings.Contains(got, `<a href="https://example.com/page">`) {
		t.Errorf("Markdown dropped the author's link destination:\n%s", got)
	}
	if strings.Contains(got, "target=") {
		t.Errorf("Markdown forced a new tab on the author's link:\n%s", got)
	}
}
