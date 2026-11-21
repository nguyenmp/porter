// Package historysearch owns the built-in "history-search" skill: how to find
// an earlier conversation by what was said in it. Every message this server
// holds lives in porter.db, and the v14 migration gives that text an FTS5
// index, so the feature is a written-down query rather than a tool: the agent
// already has shell, the runtime image ships the sqlite3 CLI, and the database
// sits on the machine the commands run on. One query serves every question
// shape the user brings ("where did I ask about the cron job", "which chat has
// that error"), instead of a new Go path for each.
//
// Like humanize's plain-language skill and handoff, the body is compiled into
// the binary from the embedded skill.md, so the server still ships as a single
// binary with no skill files on disk: exec's discovery lists the skill with a
// sentinel Path (api.BuiltinPrefix) and tools' dispatcher serves Prompt() from
// memory.
package historysearch

import (
	_ "embed"

	"porter/internal/api"
)

// SkillName is the name the built-in history-search skill is exposed under,
// both in the load_skill listing and as the sentinel path (api.BuiltinPrefix +
// Name) that tells the dispatcher to serve its body from memory.
const SkillName = "history-search"

// BuiltinSkill returns the history-search prompt as a built-in skill. Name and
// Description are what the model sees in load_skill; the sentinel Path tells
// the dispatcher to serve the body from memory (Prompt) instead of a file, so
// the skill is available in every build even when no skill files exist.
func BuiltinSkill() api.Skill {
	return api.Skill{
		Name:        SkillName,
		Description: "Search every stored chat for something the user remembers: a phrase, a decision, a command's output. Load for the database schema, a read-only FTS5 query with relevance ranking, and the rule that writing to the database is never allowed.",
		Path:        api.BuiltinPrefix + SkillName,
	}
}

// Prompt returns the body of the built-in history-search skill: the schema, the
// query, and the rules around both.
func Prompt() string { return skill }

// skill is the skill body. It lives in skill.md rather than in Go source, so
// anyone can read it as prose and link to the file; the build embeds that file,
// so the binary still carries its own copy and the server needs no skill files
// on disk. It is the authoritative copy: exec's discovery reserves built-in
// names, so a filesystem skill of the same name is skipped and can never shadow
// this guidance. Embedding the file keeps its trailing newline in the body, so
// the served text matches the file byte for byte.
//
//go:embed skill.md
var skill string
