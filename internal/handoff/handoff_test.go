package handoff

import (
	"os"
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
// defines the skill, warns that the sandbox goes away, and lists what a handoff
// summary must contain for someone who was not in the session: why the work
// exists and what the conversation settled come before where it stands.
func TestPromptCoversWhatToWrite(t *testing.T) {
	b := flat(Prompt())
	for _, want := range []string{
		"Could you write a summary using plain-language?",
		"pick up where we left off",
		"## The sandbox goes away",
		"may be on another machine",
		"Save what would be lost",
		"Ask before you push",
		"no remote",
		"git can't carry",
		"## What to cover",
		"Why we did this",
		"what to protect",
		"What we settled",
		"A requirement without a reason gets dropped at the first obstacle",
		"what we rejected and why",
		"what is still open",
		"Where things stand",
		"Where the work lives",
		"commit SHA",
		"The next step",
		"unpushed commits",
		"## How to write it",
		"plain-language skill",
	} {
		if !strings.Contains(b, want) {
			t.Errorf("prompt missing %q:\n%s", want, b)
		}
	}
}

// TestPromptSetsCarryOverBalance verifies the body tells the model where to
// draw the line between the conversation and the repo: carry what the next
// person cannot look up, and skip what they can read for themselves. Without
// that line the summary either buries the reader in transcript or loses the
// goal and the requirements behind it.
func TestPromptSetsCarryOverBalance(t *testing.T) {
	b := flat(Prompt())
	for _, want := range []string{
		"The conversation shows why",
		"Carry over what the next person cannot look up",
		"Skip what they can read",
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
	b := flat(Prompt())
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

// flat collapses the prompt's line wrapping to single spaces, so a test can
// assert on a whole sentence that prompt.md splits across lines.
func flat(s string) string { return strings.Join(strings.Fields(s), " ") }

// TestPromptEmbedded verifies the served body is the embedded prompt.md. That
// file is the copy a reader can link to, so the text load_skill serves must
// match it byte for byte.
func TestPromptEmbedded(t *testing.T) {
	want, err := os.ReadFile("prompt.md")
	if err != nil {
		t.Fatalf("read prompt.md: %v", err)
	}
	if got := Prompt(); got != string(want) {
		t.Errorf("Prompt() does not match prompt.md:\ngot:  %q\nwant: %q", got, want)
	}
}
