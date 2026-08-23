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
