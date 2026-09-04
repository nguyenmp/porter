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
const prompt = `# How to edit files on a remote host

To change a file on another machine (one you reach over ssh), pull it into
the working directory, edit the local copy, and push it back.

Don't edit a remote file in place by piping sed, perl, python, or shell
one-liners over ssh. Every layer you add can break the edit or silently
corrupt the file: your shell's quoting, the remote shell's parsing, the
tool's own escaping. This way is slower and far less reliable than the
steps below.

1. Pull the file into the working directory:

   scp user@host:/path/to/file.txt ./file.txt

   For a whole directory, use rsync -av. For a whole repo, use git clone or
   check out the branch when you can, so you start from a known-good state.

2. Edit the local copy with the normal editing tools.

   The editing tools and the shell work in the same directory and see the
   same files. You edit the local copy just like any other local file: no
   quoting passes through remote layers, and the exact-match and
   numbered-edit rules behave predictably. Use:

   - read_with_line_numbers reads the file with line numbers. Its header
     shows the line count, byte count, and whether the file ends in a
     newline.
   - line_insert adds whole lines before a numbered line, or at the end of
     the file.
   - line_replace replaces a numbered range of whole lines. An empty
     new_text deletes the range.
   - string_replace replaces a distinctive, exact string with new text.
   - the shell tool creates a new file.

3. Review the change before you send it back. The editing tools show what
   changed, with line numbers. If the directory is a git checkout, run
   git diff to see the change.

4. Push the edited file back:

   scp ./file.txt user@host:/path/to/file.txt

In a git checkout, this workflow is a safety net too: git keeps the
pre-edit state, so you can undo a bad pull, edit, or push.

Use remote one-liners only for tasks that must run on the remote machine
itself — restarting a service, moving files that only exist there. For
editing file content, use the local-copy workflow.
`
