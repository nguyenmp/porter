package visual

import (
	"strings"
	"testing"

	"porter/internal/api"
)

// TestBuiltinSkillSentinel checks the skill advertises the sentinel path the
// dispatcher resolves from memory, and a description a model can choose from.
// A real path here would make the server look for a file it never ships.
func TestBuiltinSkillSentinel(t *testing.T) {
	s := BuiltinSkill()
	if s.Name != SkillName {
		t.Errorf("name = %q, want %q", s.Name, SkillName)
	}
	if s.Path != api.BuiltinPrefix+SkillName {
		t.Errorf("path = %q, want sentinel %q", s.Path, api.BuiltinPrefix+SkillName)
	}
	if s.Description == "" {
		t.Error("description is empty, so load_skill cannot say what the skill holds")
	}
}

// TestFourPhasesInRulesBody checks the phases live in the shared rule body and
// run in order: condense, expand, write plainly, verify. The order is the
// point, and a silent reshuffle should fail here rather than reach a prompt.
func TestFourPhasesInRulesBody(t *testing.T) {
	phases := []string{"1. Condense.", "2. Expand.", "3. Apply the rules below.", "4. Verify."}
	last := -1
	for _, phase := range phases {
		i := strings.Index(rules, phase)
		if i < 0 {
			t.Fatalf("rules.md is missing %q:\n%s", phase, rules)
		}
		if i < last {
			t.Errorf("%q comes out of order in rules.md", phase)
		}
		last = i
		if !strings.Contains(Prompt(), phase) {
			t.Errorf("Prompt() does not carry %q", phase)
		}
	}
}

// TestRulesRunLargestConcernFirst checks the body reads biggest to smallest,
// from choosing the form down to the check, with the dated taste sections last
// because every rule above them outranks them.
func TestRulesRunLargestConcernFirst(t *testing.T) {
	sections := []string{
		"## Choose the form before you draw",
		"## Lead with the main point",
		"## Group so the eye reads in order",
		"## Encode by position",
		"## Label directly",
		"## Use color as a cue, never as the only cue",
		"## Set the page and the type for reading",
		"## Keep the accent for one job",
		"## Avoid the AI look",
		"## Check your work",
	}
	last := -1
	for _, s := range sections {
		i := strings.Index(rules, s)
		if i < 0 {
			t.Fatalf("rules.md is missing the %q section:\n%s", s, rules)
		}
		if i < last {
			t.Errorf("section %q runs before the one above it; sections must run largest concern first", s)
		}
		last = i
	}
}

// TestHouseRulesAreDated checks the house rules carry a date and a name, so a
// reader can tell an agreement from a guide.
func TestHouseRulesAreDated(t *testing.T) {
	want := []string{
		"(House rule, 19 November 2026)",
		"checked on 19 November 2026",
		"(Refactoring UI; house rule, 19 November 2026)",
	}
	for _, w := range want {
		if !strings.Contains(rules, w) {
			t.Errorf("rules.md is missing the dated marker %q", w)
		}
	}
}
