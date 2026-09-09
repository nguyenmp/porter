package asset

import (
	"strings"
	"testing"
)

// TestParseIframeArgs covers the render_iframe argument contract: a relative
// asset URL is accepted (with optional height clamping to the allowed range),
// while an absolute or foreign src is refused — the sandbox is only as good
// as the served HTML's CSP, so the frame must point at a published asset.
func TestParseIframeArgs(t *testing.T) {
	good, err := ParseIframeArgs(`{"src":"/assets/sessions/session_1/ab12cd34/page.html"}`)
	if err != nil {
		t.Fatalf("valid args rejected: %v", err)
	}
	if good.Src != "/assets/sessions/session_1/ab12cd34/page.html" {
		t.Errorf("src = %q", good.Src)
	}
	if good.Height != nil {
		t.Errorf("height should be nil when omitted, got %d", *good.Height)
	}
	h := 700
	tall, err := ParseIframeArgs(`{"src":"/assets/sessions/session_1/ab12cd34/page.html","height":700}`)
	if err != nil || tall.Height == nil || *tall.Height != h {
		t.Errorf("height 700 not parsed: %+v err=%v", tall, err)
	}
	bad := []string{
		`{"src":"https://evil.example/x.html"}`,
		`{"src":"/api/sessions/1"}`,
		`{"src":"/assets/sessions/session_1/ab12cd34/page.html","height":50}`,
		`{}`,
		``,
	}
	for _, in := range bad {
		if _, err := ParseIframeArgs(in); err == nil {
			t.Errorf("ParseIframeArgs(%q) succeeded, want error", in)
		}
	}
}

// TestToolDescriptionsCrossReference locks in the mutually-referential tool
// descriptions the model reads: publish_asset names render_iframe for HTML,
// and render_iframe says to publish with publish_asset first.
func TestToolDescriptionsCrossReference(t *testing.T) {
	pub := Def().Function.Description
	frame := IframeDef().Function.Description
	if !strings.Contains(pub, "render_iframe") {
		t.Errorf("publish_asset description should mention render_iframe:\n%s", pub)
	}
	if !strings.Contains(frame, "publish_asset") {
		t.Errorf("render_iframe description should mention publish_asset:\n%s", frame)
	}
	if !strings.Contains(pub, "<img") {
		t.Errorf("publish_asset description should show the <img> use case:\n%s", pub)
	}
}
