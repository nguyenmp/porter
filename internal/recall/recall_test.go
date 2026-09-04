package recall

import (
	"encoding/json"
	"fmt"
	"strings"
	"testing"
	"time"

	"porter/internal/llm"
)

// buildContent returns a deterministic byte payload: headBytes 'h', middle m
// 'm's, tailBytes 't's, plus the shell tool's trailing exit line. Useful for
// asserting exact head/tail/middle separation.
func buildContent(m int) string {
	return strings.Repeat("h", HeadBytes) + strings.Repeat("m", m) + strings.Repeat("t", TailBytes) + "\nexit code: 0\n"
}

func TestMetaTruncationBoundaries(t *testing.T) {
	// Small output: not truncated; shown == total.
	small := "hi"
	if m := Meta(small); m.Truncated || m.TotalBytes != 2 || m.ShownBytes != 2 {
		t.Errorf("Meta(small) = %+v, want not truncated, 2/2", m)
	}
	// Exactly the head+tail budget: still not truncated (the whole thing fits).
	exact := strings.Repeat("x", HeadBytes+TailBytes)
	if m := Meta(exact); m.Truncated || m.ShownBytes != HeadBytes+TailBytes {
		t.Errorf("Meta(exact budget) = %+v, want not truncated", m)
	}
	// One byte over: truncated, shown is exactly head+tail.
	over := strings.Repeat("x", HeadBytes+TailBytes+1)
	if m := Meta(over); !m.Truncated || m.ShownBytes != HeadBytes+TailBytes {
		t.Errorf("Meta(over) = %+v, want truncated with shown=head+tail", m)
	}
}

func TestTruncateFormat(t *testing.T) {
	content := buildContent(1000)
	meta := Meta(content)
	got := Truncate(content, "call_1", meta)

	// The header states sizes (comma-formatted) and how to load the rest.
	expectedHeader := fmt.Sprintf("[tool output: %s of %s bytes (head); last %s shown below.  To load more: recall_tool_output(call_id=\"call_1\", offset=%d, max_bytes=%d).]",
		comma(HeadBytes), comma(len(content)), bytesLabel(TailBytes), HeadBytes, len(content)-HeadBytes)
	if !strings.Contains(got, expectedHeader) {
		t.Errorf("truncation header missing/wrong:\n%s", got)
	}
	// Head bytes present at the start of the body.
	if !strings.HasPrefix(got, "[tool output:") {
		t.Errorf("truncated output must start with the header")
	}
	// The middle is gone, replaced by an omitted marker.
	if strings.Contains(got, strings.Repeat("m", 1000)) {
		t.Errorf("truncated output must omit the middle bytes")
	}
	expectedOmitted := fmt.Sprintf("[... %s omitted ...]", bytesLabel(len(content)-HeadBytes-TailBytes))
	if !strings.Contains(got, expectedOmitted) {
		t.Errorf("truncated output missing omitted marker:\n%s", got)
	}
	// The tail (with the exit line) is preserved at the very end.
	if !strings.HasSuffix(got, content[len(content)-TailBytes:]) {
		t.Errorf("truncated output must end with the tail + exit line:\n%s", got)
	}
	// The head and a recall_tool_output at offset=head must line up exactly.
	head := content[:HeadBytes]
	if !strings.Contains(got, head) {
		t.Errorf("truncated output missing the head bytes")
	}
	// Idempotent: truncating an already-truncated form is a no-op.
	if again := Truncate(got, "call_1", meta); again != got {
		t.Errorf("Truncate is not idempotent")
	}
}

func TestTruncateFitsBudgetUnchanged(t *testing.T) {
	small := "hello\n"
	if got := Truncate(small, "call_1", Meta(small)); got != small {
		t.Errorf("Truncate on small output changed it: %q", got)
	}
}

func TestProjectModelView(t *testing.T) {
	full := buildContent(1000)
	big := llm.ToolResult("call_1", full)
	big.ToolOutput = Meta(full)

	small := llm.ToolResult("call_2", "tiny")
	small.ToolOutput = Meta("tiny")

	recallMsg := llm.ToolResult("call_3", "the window bytes should stay")
	recallMsg.ToolOutput = &llm.ToolOutputMeta{Recall: true, SourceCallID: "call_1", Offset: 1024, MaxBytes: 1000, TotalBytes: len(full), ShownBytes: 1000}

	noMeta := llm.ToolResult("call_4", buildContent(500))

	user := llm.UserMessage("hi")
	history := []llm.ChatMessage{user, big, small, recallMsg, noMeta}

	out := ProjectModelView(history)

	// Input is never mutated: history keeps the full output.
	if history[1].Content != full {
		t.Errorf("ProjectModelView mutated its input; history should hold the full output")
	}
	// User message unchanged.
	if out[0].Content != "hi" {
		t.Errorf("user message changed: %q", out[0].Content)
	}
	// Big tool message truncated in the view.
	if !strings.HasPrefix(out[1].Content, "[tool output:") {
		t.Errorf("big tool message not truncated in the view: %.60s", out[1].Content)
	}
	// Small tool message unchanged.
	if out[2].Content != "tiny" {
		t.Errorf("small tool message changed: %q", out[2].Content)
	}
	// Recall message kept intact (the one exception to truncation).
	if out[3].Content != "the window bytes should stay" {
		t.Errorf("recall message truncated: %q", out[3].Content)
	}
	// A tool message without metadata is truncated too (uniform model view).
	if !strings.HasPrefix(out[4].Content, "[tool output:") {
		t.Errorf("meta-less big tool message not truncated: %.60s", out[4].Content)
	}
}

func TestServeWindow(t *testing.T) {
	full := buildContent(1000)
	history := []llm.ChatMessage{
		llm.UserMessage("hi"),
		llm.AssistantMessage("", "", []llm.ToolCall{{ID: "call_1", Type: "function", Function: llm.ToolFunction{Name: "shell", Arguments: "{}"}}}),
		llm.ToolResult("call_1", full),
	}

	// A window covering exactly the middle bytes.
	window, meta, err := ServeWindow(history, fmt.Sprintf(`{"call_id":"call_1","offset":%d,"max_bytes":1000}`, HeadBytes))
	if err != nil {
		t.Fatalf("ServeWindow: %v", err)
	}
	if meta.SourceCallID != "call_1" || meta.Offset != HeadBytes || meta.MaxBytes != 1000 || meta.TotalBytes != len(full) || meta.ShownBytes != 1000 {
		t.Errorf("meta = %+v", meta)
	}
	if !strings.Contains(window, strings.Repeat("m", 1000)) {
		t.Errorf("window missing the middle bytes")
	}
	if !strings.Contains(window, "[recall: recall_tool_output") {
		t.Errorf("window missing the recall header:\n%s", window)
	}
	// The window is the raw bytes; the head and this window line up exactly.
	if windowBytes := window[strings.Index(window, "\n")+1:]; windowBytes != full[HeadBytes:HeadBytes+1000] {
		t.Errorf("window bytes = %q..., want full[HeadBytes:HeadBytes+1000]", windowBytes[:20])
	}

	// Omitting max_bytes reads to the end of the output.
	toEnd, metaEnd, err := ServeWindow(history, fmt.Sprintf(`{"call_id":"call_1","offset":%d}`, HeadBytes))
	if err != nil {
		t.Fatalf("ServeWindow to-end: %v", err)
	}
	if metaEnd.MaxBytes != len(full)-HeadBytes {
		t.Errorf("to-end max_bytes = %d, want %d", metaEnd.MaxBytes, len(full)-HeadBytes)
	}
	if !strings.HasSuffix(toEnd, full[HeadBytes:]) {
		t.Errorf("to-end window must be the rest of the output")
	}

	// Unknown call_id is an error (the model may only read what it has seen).
	if _, _, err := ServeWindow(history, `{"call_id":"nope"}`); err == nil {
		t.Errorf("unknown call_id should error")
	}
	// Invalid arguments error.
	for _, args := range []string{`{"call_id":""}`, `{"call_id":"call_1","offset":-1}`, `{"call_id":"call_1","max_bytes":-1}`, `not json`} {
		if _, _, err := ServeWindow(history, args); err == nil {
			t.Errorf("args %q should error", args)
		}
	}
	// Offset beyond the end clamps to the end (an empty window), not an error.
	clamped, metaC, err := ServeWindow(history, `{"call_id":"call_1","offset":999999}`)
	if err != nil {
		t.Fatalf("ServeWindow clamp: %v", err)
	}
	if metaC.MaxBytes != 0 || metaC.Offset != len(full) {
		t.Errorf("clamped meta = %+v, want offset=total, max=0", metaC)
	}
	if !strings.HasSuffix(clamped, "(0 bytes) served to the model]\n") {
		t.Errorf("clamped window header = %q", clamped)
	}
}

func TestWindowTextAndPlaceholder(t *testing.T) {
	meta := &llm.ToolOutputMeta{Recall: true, SourceCallID: "call_1", Offset: 1024, MaxBytes: 1000, TotalBytes: 2549, ShownBytes: 1000}
	window := WindowText(meta, "MIDDLE")
	if !strings.HasPrefix(window, "[recall: recall_tool_output(call_id=\"call_1\", offset=1024, max_bytes=1000) -> bytes 1024-2024 of 2549 (1000 bytes) served to the model]\n") {
		t.Errorf("window text = %q", window)
	}
	if !strings.HasSuffix(window, "MIDDLE") {
		t.Errorf("window text must carry the window bytes")
	}
	placeholder := Placeholder(meta)
	if !strings.Contains(placeholder, "returned bytes 1024-2024 of 2549 (1000 bytes) to the model context") {
		t.Errorf("placeholder = %q", placeholder)
	}
	// The placeholder must NOT carry the window bytes (no duplication).
	if strings.Contains(placeholder, "MIDDLE") {
		t.Errorf("placeholder must not duplicate the window bytes")
	}
}

func TestDef(t *testing.T) {
	d := Def()
	if d.Function.Name != "recall_tool_output" {
		t.Errorf("Def name = %q", d.Function.Name)
	}
	params := d.Function.Parameters["properties"].(map[string]any)
	if _, ok := params["call_id"]; !ok {
		t.Errorf("Def missing call_id parameter")
	}
	if _, ok := params["offset"]; !ok {
		t.Errorf("Def missing offset parameter")
	}
	if _, ok := params["max_bytes"]; !ok {
		t.Errorf("Def missing max_bytes parameter")
	}
	required := d.Function.Parameters["required"].([]string)
	if len(required) != 1 || required[0] != "call_id" {
		t.Errorf("Def required = %v, want [call_id]", required)
	}
	// The schema must round-trip through JSON (the LLM sees it).
	if _, err := json.Marshal(d); err != nil {
		t.Errorf("Def does not marshal: %v", err)
	}
}

// TestTimingNote pins the exact bracketed wording the model view uses, so a
// format change is a deliberate, reviewed edit (the model may rely on it).
func TestTimingNote(t *testing.T) {
	at := time.Date(2025, 9, 4, 15, 42, 11, 0, time.UTC).UnixMilli()
	end := at + 8200 // 8.2s later (sub-10s keeps the one-decimal form)

	cases := []struct {
		name string
		msg  llm.ChatMessage
		want string
	}{
		{"user", func() llm.ChatMessage { m := llm.UserMessage("hi"); m.StartedAt = at; return m }(), "[sent 2025-09-04 15:42:11 UTC]"},
		{"user no clock", llm.UserMessage("hi"), ""},
		{"assistant reply", func() llm.ChatMessage {
			m := llm.AssistantMessage("hello", "", nil)
			m.StartedAt = at
			m.FinishedAt = end
			return m
		}(), "[replied 2025-09-04 15:42:19 UTC, took 8.2s]"},
		{"assistant tool-call frame (no prose)", func() llm.ChatMessage {
			m := llm.AssistantMessage("", "", []llm.ToolCall{{ID: "c1"}})
			m.StartedAt = at
			m.FinishedAt = end
			return m
		}(), ""},
		{"assistant started only", func() llm.ChatMessage { m := llm.AssistantMessage("hi", "", nil); m.StartedAt = at; return m }(), "[replied 2025-09-04 15:42:11 UTC]"},
		{"tool", func() llm.ChatMessage {
			m := llm.ToolResult("c", "out")
			m.StartedAt = at
			m.FinishedAt = end
			return m
		}(), "[ran 8.2s, finished 2025-09-04 15:42:19 UTC]"},
		{"tool cancelled", func() llm.ChatMessage {
			m := llm.ToolResult("c", "partial")
			m.StartedAt = at
			m.FinishedAt = end
			m.Cancelled = true
			return m
		}(), "[cancelled after 8.2s at 2025-09-04 15:42:19 UTC]"},
		{"tool no clock", llm.ToolResult("c", "out"), ""},
		{"system", func() llm.ChatMessage { m := llm.SystemMessage("notice"); m.StartedAt = at; return m }(), "[at 2025-09-04 15:42:11 UTC]"},
		{"system no clock", llm.SystemMessage("notice"), ""},
	}
	for _, c := range cases {
		if got := timingNote(c.msg); got != c.want {
			t.Errorf("%s: timingNote = %q, want %q", c.name, got, c.want)
		}
	}
}

// TestDurText covers the compact model-view duration forms.
func TestDurText(t *testing.T) {
	cases := []struct {
		ms   int64
		want string
	}{
		{0, ""},
		{800, "0.8s"},
		{2500, "2.5s"},
		{9500, "9.5s"},
		{12000, "12s"},
		{45000, "45s"},
		{192000, "3m 12s"},
		{180000, "3m"},
		{3900000, "1h 5m"},
		{7200000, "2h"},
	}
	for _, c := range cases {
		if got := durText(c.ms); got != c.want {
			t.Errorf("durText(%d) = %q, want %q", c.ms, got, c.want)
		}
	}
}

// TestProjectModelViewTiming verifies the projection annotates timed messages
// with the bracket header, leaves clock-less messages alone, never mutates its
// input, and puts the timing note in front of a truncation header when both
// apply.
func TestProjectModelViewTiming(t *testing.T) {
	at := time.Date(2025, 9, 4, 15, 42, 11, 0, time.UTC).UnixMilli()
	full := buildContent(1000)
	big := llm.ToolResult("call_big", full)
	big.StartedAt = at
	big.FinishedAt = at + 192000 // 3m 12s
	big.ToolOutput = Meta(full)

	user := llm.UserMessage("hi")
	user.StartedAt = at

	plain := llm.ToolResult("call_plain", "tiny") // no clocks

	history := []llm.ChatMessage{user, big, plain}
	out := ProjectModelView(history)

	// Input is never mutated (history still holds untimed, full content).
	if history[0].Content != "hi" || history[1].Content != full {
		t.Errorf("ProjectModelView mutated its input")
	}
	// A timed user message is annotated.
	if want := "[sent 2025-09-04 15:42:11 UTC]\nhi"; out[0].Content != want {
		t.Errorf("user view = %q, want %q", out[0].Content, want)
	}
	// A big timed tool result is truncated AND annotated, timing note first.
	if !strings.HasPrefix(out[1].Content, "[ran 3m 12s, finished 2025-09-04 15:45:23 UTC]\n[tool output:") {
		t.Errorf("big tool view = %.120s..., want timing note before truncation header", out[1].Content)
	}
	// A clock-less tool message is untouched (no fabricated note).
	if out[2].Content != "tiny" {
		t.Errorf("clock-less tool message changed: %q", out[2].Content)
	}
	// Re-projecting the OUTPUT would double-annotate, so the caller must always
	// project from stored history (which the agent does fresh each request).
	// Guard that the output of a single pass is deterministic.
	again := ProjectModelView(history)
	if again[0].Content != out[0].Content {
		t.Errorf("projection not deterministic across passes")
	}
}
