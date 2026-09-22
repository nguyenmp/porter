// Package handoff owns the built-in "handoff" skill: how to write a summary
// that lets someone else pick up the work where this session left off. Like
// humanize's plain-language skill and remoteedit's editing-remote-files skill,
// the body is compiled into the binary (the server ships as a single binary
// with no skill files in its build), so exec's discovery lists it with a
// sentinel Path and tools' dispatcher serves Prompt() from memory. The
// sentinel prefix is api.BuiltinPrefix, shared by every built-in skill and its
// plumbing.
//
// The body deliberately does not restate the plain-language rules. The
// standing writing directive already carries them into every request, and the
// plain-language skill holds the full list, so the body points at the rules
// instead of copying them into a second place that could drift.
package handoff

import "porter/internal/api"

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

// prompt is the skill body, kept as a raw string so it reads like a SKILL.md
// that never ships as a file. It is the authoritative copy: exec's discovery
// reserves built-in names, so a filesystem skill of the same name is skipped
// and can never shadow this guidance.
const prompt = `Could you write a summary using plain-language? You're handing this off to someone else to pick up where we left off.

Write for the person who picks the work up next, not for the person who did
it. They did not sit in this session and cannot read the conversation.
Everything they know about the work comes from your summary.

## What to cover

1. What we were doing, and why.
2. Where things stand: what is done, what is left to do, and what we already
   decided. Say when something is unverified, failing, or blocked.
3. Where the work lives: file paths, branch and worktree names, links, and the
   commands that matter (build, run, test).
4. The next step: the one thing the next person should do first.
5. Traps: what looks finished but is not, such as uncommitted edits, failing
   checks, or a step that only works in one order.

## How to write it

Follow the plain-language rules. Load the plain-language skill for the full
list.
`
