package remoteedit

import (
	"strings"
	"testing"

	"porter/internal/api"
)

// TestBuiltinSkillShape verifies the skill metadata: a sentinel path the
// dispatcher recognizes as compiled-in, a non-empty description for the
// load_skill listing, and a name that matches the sentinel suffix.
func TestBuiltinSkillShape(t *testing.T) {
	s := BuiltinSkill()
	if s.Name != SkillName {
		t.Errorf("name = %q, want %q", s.Name, SkillName)
	}
	if s.Path != api.BuiltinPrefix+SkillName {
		t.Errorf("path = %q, want sentinel %q", s.Path, api.BuiltinPrefix+SkillName)
	}
	if s.Description == "" {
		t.Error("description is empty")
	}
}

// TestPromptCoversWorkflow verifies the body gives a model the actionable core
// of the guidance: pull a copy down, edit locally, push it back — not sed or
// perl one-liners over ssh.
func TestPromptCoversWorkflow(t *testing.T) {
	b := Prompt()
	for _, want := range []string{"How to edit files on a remote host", "scp", "read_with_line_numbers", "line_insert", "line_replace", "string_replace", "git diff", "push it back"} {
		if !strings.Contains(b, want) {
			t.Errorf("prompt missing %q:\n%s", want, b)
		}
	}
	if strings.Contains(b, "sed") == false {
		t.Errorf("prompt should name sed/perl as the anti-pattern to avoid")
	}
}
