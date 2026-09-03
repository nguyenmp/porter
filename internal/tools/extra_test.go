package tools

import (
	"context"
	"strings"
	"testing"
)

// TestLineReplaceRemovedBlockShowsOnlyRemoved verifies the "removed" echo shows
// exactly the removed lines (no context under a heading that implies they were
// all removed).
func TestLineReplaceRemovedBlockShowsOnlyRemoved(t *testing.T) {
	path := writeTemp(t, "a\nb\nc\nd\ne\nf\n", 0o644)
	res, err := run(context.Background(), NewDispatcher(), LineReplaceTool, []byte(`{"path":`+quote(path)+`,"start":3,"end":4,"new_text":"X\n"}`))
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if !strings.Contains(res, "removed (old line numbers):\n     3\tc\n") {
		t.Errorf("removed block must contain only line 3 (c):\n%s", res)
	}
	// Lines 2 and 4 (kept, not removed) must not appear in the removed block.
	if strings.Contains(res, "removed (old line numbers):\n     2\t") {
		t.Errorf("removed block leaked a kept line:\n%s", res)
	}
}

// TestRenderRangeNumberingContinuity verifies a long capped range keeps correct
// numbers on both sides of the omission marker.
func TestRenderRangeNumberingContinuity(t *testing.T) {
	lines := make([]string, 20)
	for i := range lines {
		lines[i] = "L\n"
	}
	out := renderRange(lines, 0, 19)
	if !strings.HasPrefix(out, "     1\tL\n") || !strings.Contains(out, "\n    20\tL\n") {
		t.Errorf("renderRange must keep first and last line numbers:\n%s", out)
	}
	if !strings.Contains(out, "[... 12 lines omitted ...]\n") {
		t.Errorf("renderRange must mark the omitted middle (20-8 = 12):\n%s", out)
	}
	if strings.Contains(out, "\n     5\tL\n") {
		t.Errorf("renderRange should not render the omitted middle lines 5-8:\n%s", out)
	}
}
