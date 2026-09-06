package spool

import (
	"encoding/json"
	"strings"
	"testing"

	"porter/internal/llm"
	"porter/internal/tools"
)

func TestParseArgs(t *testing.T) {
	// No path: default under DefaultDir, keyed by call id.
	callID, path, err := ParseArgs(`{"call_id":"call_1"}`)
	if err != nil || callID != "call_1" || path != ".porter/out/call_1" {
		t.Errorf("ParseArgs(no path) = (%q, %q, %v), want (call_1, .porter/out/call_1, nil)", callID, path, err)
	}
	// Explicit path is kept as given (the provider resolves it).
	callID, path, err = ParseArgs(`{"call_id":"call_2","path":"data/out.json"}`)
	if err != nil || callID != "call_2" || path != "data/out.json" {
		t.Errorf("ParseArgs(explicit) = (%q, %q, %v), want (call_2, data/out.json, nil)", callID, path, err)
	}
	// Missing call_id and malformed JSON are errors.
	if _, _, err := ParseArgs(`{"path":"x"}`); err == nil {
		t.Errorf("ParseArgs without call_id: want error")
	}
	if _, _, err := ParseArgs(`not json`); err == nil {
		t.Errorf("ParseArgs with bad json: want error")
	}
}

func TestLookup(t *testing.T) {
	full := llm.ToolResult("call_1", "the full output bytes")
	recallMsg := llm.ToolResult("call_2", "window bytes")
	recallMsg.ToolOutput = &llm.ToolOutputMeta{Recall: true, SourceCallID: "call_1", Offset: 0, MaxBytes: 5, TotalBytes: len("the full output bytes"), ShownBytes: 5}
	history := []llm.ChatMessage{full, recallMsg}

	if got, err := Lookup(history, "call_1"); err != nil || got != "the full output bytes" {
		t.Errorf("Lookup(full) = (%q, %v), want the full output bytes, nil", got, err)
	}
	if _, err := Lookup(history, "call_2"); err == nil {
		t.Errorf("Lookup(recall window): want error (not a full result)")
	}
	if _, err := Lookup(history, "nope"); err == nil {
		t.Errorf("Lookup(unknown): want error")
	}
}

func TestShouldHint(t *testing.T) {
	// Data-producing tools get the hint.
	for _, name := range []string{"shell", "CallMCP", "FindMCP", ""} {
		if !ShouldHint(name) {
			t.Errorf("ShouldHint(%q) = false, want true", name)
		}
	}
	// Prose / on-disk producers do not.
	for _, name := range []string{tools.ReadLinesTool, tools.LineInsertTool, tools.LineReplaceTool, tools.StringReplace, tools.LoadSkillTool} {
		if ShouldHint(name) {
			t.Errorf("ShouldHint(%q) = true, want false", name)
		}
	}
}

func TestWritePayload(t *testing.T) {
	b := WritePayload("out/x.json", "content")
	var got WriteArgs
	if err := json.Unmarshal(b, &got); err != nil {
		t.Fatalf("WritePayload is not valid json: %v", err)
	}
	if got.Path != "out/x.json" || got.Content != "content" {
		t.Errorf("WritePayload = %+v, want out/x.json/content", got)
	}
}

func TestDef(t *testing.T) {
	d := Def()
	if d.Function.Name != OutputTool {
		t.Errorf("Def().Function.Name = %q, want %q", d.Function.Name, OutputTool)
	}
	if !strings.Contains(d.Function.Description, DefaultDir) {
		t.Errorf("Def description should mention the default directory %s", DefaultDir)
	}
}

func TestFooterHint(t *testing.T) {
	h := FooterHint("call_9")
	if !strings.Contains(h, OutputTool) || !strings.Contains(h, `"call_9"`) || !strings.Contains(h, DefaultDir) {
		t.Errorf("FooterHint missing tool/path/default: %q", h)
	}
}
