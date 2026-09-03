// Package remoteedit owns the built-in "editing-remote-files" skill: guidance
// for changing a file that lives on a machine other than the one the tools run
// on. Like humanize's built-in plain-language skill, the body is compiled into
// the binary (the server has no skill files in its build), so exec's discovery
// lists it with a sentinel Path and tools' dispatcher serves Prompt() from
// memory. The sentinel prefix is api.BuiltinPrefix, shared by every built-in
// skill and its plumbing.
package remoteedit

import "porter/internal/api"

// SkillName is the name the built-in editing-remote-files skill is exposed
// under, both in the load_skill listing and as the sentinel path
// (api.BuiltinPrefix + Name) that tells the dispatcher to serve its body from
// memory.
const SkillName = "editing-remote-files"

// BuiltinSkill returns the editing-remote-files prompt as a built-in skill.
// Name and Description are what the model sees in load_skill; the sentinel
// Path tells the dispatcher to serve the body from memory (Prompt) instead of
// a file, so the skill is available in every build even when no skill files
// exist.
func BuiltinSkill() api.Skill {
	return api.Skill{
		Name:        SkillName,
		Description: "Edit a file that lives on another machine (over ssh) without fighting quoting and escaping layers: copy it into the working directory, edit the local copy with the file tools, and copy it back. Load for the workflow.",
		Path:        api.BuiltinPrefix + SkillName,
	}
}

// Prompt returns the body of the built-in editing-remote-files skill: the
// guidance a model should follow when a task needs to change a file that is not
// on the execution provider's local filesystem.
func Prompt() string { return prompt }

// prompt is the skill body, kept as a raw string so it reads like a SKILL.md
// that never ships as a file. When a filesystem skill of the same name exists,
// it shadows this built-in (exec's discovery gives filesystem skills
// priority), which lets a user override the guidance with their own copy.
const prompt = `# Editing files on a remote host

When you need to change a file that lives on another machine (reachable over
ssh), edit a local copy and push it back. Do not edit it in place by piping
sed, perl, python, or shell one-liners over ssh: every layer you add — your
shell's quoting, the remote shell's parsing, the tool's own escaping — is a
place the edit can fail or silently corrupt the file. It is slower and far
less reliable than the workflow below.

1. Pull the file into the working directory:

   scp user@host:/path/to/file.txt ./file.txt

   For a whole directory use rsync (-av). For a whole repo, git clone or check
   out the branch when you can, so the pull is a known-good state.

2. Edit the local copy with the normal editing tools:

   - read_with_line_numbers to inspect it (the header reports the line count,
     the byte count, and whether the file ends in a newline);
   - line_replace to change a numbered range, insert lines, or delete lines;
   - string_replace for a distinctive exact text change;
   - the shell tool to create a brand-new file.

   These tools and the shell share one working directory and see the same
   files, so a local copy edits exactly like any other local file — no quoting
   through remote layers, and the exact-match and numbered-edit rules behave
   predictably.

3. Review the change before sending it back: the editing tools echo what
   changed with line numbers, or run git diff when the directory is a checkout.

4. Push the edited file back:

   scp ./file.txt user@host:/path/to/file.txt

When the working directory is a git checkout this doubles as a safety net: git
already holds the pre-edit state, so pulling, editing, and pushing is
rollback-friendly.

Reserve remote one-liners for operations that must run on the remote machine
itself — restarting a service, moving files that only exist there. For editing
file content, prefer the local-copy workflow.
`
