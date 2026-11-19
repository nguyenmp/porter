// Package visual owns the built-in "visual-design" skill: rules for figures,
// diagrams, screenshots, tables, slides, posters, and the style layer of a
// page. It is a sibling of humanize's plain-language skill, written in the same
// shape (a definition, a rule that settles a conflict, four phases, then
// sections from the whole surface down to type and color) and served the same
// way: the body is compiled into the binary, so exec's discovery lists it with
// a sentinel Path (api.BuiltinPrefix) and tools' dispatcher serves Prompt() from
// memory. The words on a visual stay plain language; this package owns the
// pixels, and plain_rules.md points here for type, layout, and color.
package visual

import (
	_ "embed"
	"strings"

	"porter/internal/api"
)

// SkillName is the name the built-in visual-design skill is exposed under, both
// in the load_skill listing and as the sentinel path (api.BuiltinPrefix + Name)
// that tells the dispatcher to serve its body from memory.
const SkillName = "visual-design"

// BuiltinSkill returns the visual-design prompt as a built-in skill. Name and
// Description are what the model sees in load_skill; the sentinel Path tells the
// dispatcher to serve the body from memory (Prompt) instead of a file, so the
// skill is available in every build even when no skill files exist.
func BuiltinSkill() api.Skill {
	return api.Skill{
		Name:        SkillName,
		Description: "Design or fix a figure, diagram, chart, screenshot, table, slide, poster, or the style layer of a page so a reader gets the point from a glance. Load for the rules, the sources behind them, and the tests that check the result.",
		Path:        api.BuiltinPrefix + SkillName,
	}
}

// Prompt returns the body of the built-in visual-design skill: the rule body in
// rules.md. The rules live in that file rather than in Go source, so anyone can
// read them as prose and link to them, and the build embeds the file so the
// binary still carries its own copy.
func Prompt() string { return rules }

//go:embed rules.md
var rulesFile string

// rules is rulesFile with the file's trailing newline removed, so a caller can
// wrap the body without inheriting a stray blank line.
var rules = strings.TrimSuffix(rulesFile, "\n")
