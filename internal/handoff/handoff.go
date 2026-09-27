// Package handoff owns the built-in "handoff" skill: how to write a summary
// that lets someone else pick up the work where this session left off. Like
// humanize's plain-language skill and remoteedit's editing-remote-files skill,
// the body is compiled into the binary from the embedded prompt.md, so the
// server ships as a single binary with no skill files on disk and exec's
// discovery lists the skill with a sentinel Path. Tools' dispatcher then
// serves Prompt() from memory. The sentinel prefix is api.BuiltinPrefix,
// shared by every built-in skill and its plumbing.
//
// The body deliberately does not restate the plain-language rules. The
// standing writing directive already carries them into every request, and the
// plain-language skill holds the full list, so the body points at the rules
// instead of copying them into a second place that could drift.
package handoff

import (
	_ "embed"

	"porter/internal/api"
)

// SkillName is the name the built-in handoff skill is exposed under, both in
// the load_skill listing and as the sentinel path (api.BuiltinPrefix + Name)
// that tells the dispatcher to serve its body from memory.
const SkillName = "handoff"

// BuiltinSkill returns the handoff prompt as a built-in skill. Name and
// Description are what the model sees in load_skill; the sentinel Path tells
// the dispatcher to serve the body from memory (Prompt) instead of a file, so
// the skill is available in every build even when no skill files exist.
func BuiltinSkill() api.Skill {
	return api.Skill{
		Name:        SkillName,
		Description: "Write a summary of the work in plain language: you're handing this off to someone else to pick up where you left off. Load for what the summary must cover.",
		Path:        api.BuiltinPrefix + SkillName,
	}
}

// Prompt returns the body of the built-in handoff skill: the guidance a model
// should follow when the session ends and someone else takes the work over,
// either another person or a later session with no memory of this one.
func Prompt() string { return prompt }

// prompt is the skill body. It lives in prompt.md rather than in Go source, so
// anyone can read it as prose and link to the file; the build embeds that file,
// so the binary still carries its own copy and the server needs no skill files
// on disk. It is the authoritative copy: exec's discovery reserves built-in
// names, so a filesystem skill of the same name is skipped and can never shadow
// this guidance. Embedding the file keeps its trailing newline in the body, so
// the served text matches the file byte for byte.
//
//go:embed prompt.md
var prompt string
