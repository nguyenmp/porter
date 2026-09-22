package handoff

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
	if !strings.Contains(s.Description, "plain language") {
		t.Errorf("description should name plain language: %q", s.Description)
	}
}

// TestPromptCoversWhatToWrite verifies the body opens with the request that
// defines the skill and lists what a handoff summary must contain for someone
// who was not in the session.
func TestPromptCoversWhatToWrite(t *testing.T) {
	b := Prompt()
	for _, want := range []string{
		"Could you write a summary using plain-language?",
		"pick up where we left off",
		"## What to cover",
		"Where things stand",
		"Where the work lives",
		"The next step",
		"uncommitted edits",
		"## How to write it",
		"plain-language skill",
	} {
		if !strings.Contains(b, want) {
			t.Errorf("prompt missing %q:\n%s", want, b)
		}
	}
}

// TestPromptDefersToPlainLanguageSkill verifies the body refers to the
// plain-language rules instead of restating them. The rules already ride in
// every request, and the plain-language skill holds the full list, so a copy
// here would only drift from them.
func TestPromptDefersToPlainLanguageSkill(t *testing.T) {
	b := Prompt()
	for _, unwanted := range []string{
		"Use the active voice", // a rule from the plain-language body
		"short sentences",
		"a story, not a place to start",
		"and the command that starts it",
		"gets skimmed",
	} {
		if strings.Contains(b, unwanted) {
			t.Errorf("prompt still contains %q:\n%s", unwanted, b)
		}
	}
}
