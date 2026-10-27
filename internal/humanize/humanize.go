// Package humanize rewrites assistant replies in plain language, in the
// background, so assistant text blocks get a "Humanized" variant tab in the
// web UI without interrupting the turn. The pass is deliberately light: one
// LLM request carrying the text plus a full transcript of the conversation
// leading up to it (grounding for the rewrite, prefix-stable so provider
// prompt caching makes repeat passes cheap), driven by a prompt built from the
// plain-language rules this package embeds — a silent background step must be
// fast, deterministic, and cheap, not a skill-loading multi-fetch ritual. The
// same rewrite rules are also exposed as a built-in skill (BuiltinSkill /
// Prompt): the server ships as a single binary with no skill files on disk, so
// the skill body is compiled in (the embedded plain_rules.md) and load_skill
// serves it from memory. A shared
// rule base means the background pass and the loadable skill can never drift apart;
// each mode appends only the behavior its own job needs. The standing style
// the agent injects into every request (Directive) stays lean: instead of
// restating the rules, it tells the model to load this skill and thoroughly
// apply its rules to everything it writes, so replies, drafted reports, and code
// comments are written plainly the first time and the rewrite pass is a safety
// net rather than the only plain-language path. A one-line reminder
// (UserMessageReminder) rides on the conversation too: the model view appends
// it to every user message, so the style sits next to the request being
// answered and not only in the system
// prefix. Like the model view's timing headers, the reminder is derived on the
// outgoing copy only — the stored message, the database, and the UI keep
// exactly what the user typed.
package humanize

import (
	"context"
	_ "embed"
	"fmt"
	"strings"

	"porter/internal/api"
	"porter/internal/codec"
	"porter/internal/db"
	"porter/internal/llm"
)

// PromptVersion identifies the prompt revision that produced a variant. It is
// stamped on every pass so the UI can explain why a tab reads the way it does,
// and should be bumped whenever the prompt below changes.
const PromptVersion = "plain-language-v20"

// SkillName is the name the built-in plain-language skill is exposed under,
// both in the load_skill listing and as the sentinel path (api.BuiltinPrefix +
// Name) that tells the dispatcher to serve its body from memory.
const SkillName = "plain-language"

// BuiltinSkill returns the plain-language prompt as a built-in skill. Name and
// Description are what the model sees in load_skill; the sentinel Path tells
// the dispatcher to serve the body from memory (Prompt) instead of a file, so
// the skill is available in every build even when no skill files exist. The
// shared sentinel prefix lives in api (every built-in skill and its exec/tools
// plumbing agree on it); this package only contributes its own name.
func BuiltinSkill() api.Skill {
	return api.Skill{
		Name:        SkillName,
		Description: "Rewrite or write text in plain language: simple, direct, reader-first prose (from the US government plainlanguage.gov guidelines). Load to get the full rewrite rules and the loop that applies them, rule by rule, until the text is thoroughly plain.",
		Path:        api.BuiltinPrefix + SkillName,
	}
}

// Prompt returns the body of the built-in plain-language skill:the shared
// rewrite rules (systemPrompt) plus an interactive-use suffix. A model
// that loads this skill asks for text when none is given, runs its passes on a
// scratch file (rereading the file after each pass), edits a given file in
// place, answers in chat with no code block or quotation marks, and reports how
// many passes it took when the user asked for plain language - things a
// background pass must never do. The citation and self-review rules live in
// the shared rule body (plain_rules.md), so this prompt and the background
// pass cannot differ on them.
func Prompt() string { return systemPrompt + interactiveSuffix }

// humanizePrompt returns the prompt the background humanize pass uses: the
// shared rewrite rules (systemPrompt) plus a silent-output suffix. The pass
// runs unattended and its result lands directly in the UI as a variant tab, so
// the model must return only the rewritten text, with no preamble, no summary
// of changes, and no commentary.
func humanizePrompt() string { return systemPrompt + silentSuffix }

// Auto-pass thresholds: an assistant reply qualifies when it has at least
// MinWords words AND at least MinSentences sentences, and is not dominated by
// code blocks. Short replies and code dumps are not worth an automatic
// rewrite (they can still be humanized manually from the web UI's tab bar).
const (
	MinWords     = 60
	MinSentences = 4
)

// plainRules is the plain-language rule body the skill and the background pass
// share: the definition, the rule that settles a conflict (clarity wins, and
// length is a rule of thumb), the four rules that steer an edit (where to spend
// effort and in what order, how many words to cut, what to protect, and what
// must not change), and the writing rules grouped by structure, sentences,
// words, links, emphasis, page layout, and review. The sections run from the
// largest concern to the smallest, so a model reading top to bottom fixes the
// shape of the text before it polishes words, and review comes last because it
// loops back over every rule.
// Sections hold one idea per rule and run a few dozen words each, so the body
// models the style it asks for. systemPrompt wraps it with the framing line both
// modes need, so the shared guidance can never drift between the skill and the
// pass. The standing directive (systemDirective) does not wrap it: it points at
// the skill instead.
//
// The rules live in plain_rules.md rather than in Go source, so anyone can read
// them as prose and link to the file. The build embeds that file, so the binary
// still carries its own copy and the server needs no skill files on disk. The
// text is distilled from the markdown files of the gsa/plainlanguage.gov git
// repository (https://github.com/gsa/plainlanguage.gov; the guidelines live
// under _pages/guidelines/, covering audience, words, concise, conversational,
// voice, organization, design, web, and test), so prompt edits stay anchored in
// the source material rather than in memory. It condenses the guidance into one
// LLM-request-sized block with no web fetches.
//
//go:embed plain_rules.md
var plainRulesFile string

// plainRules is plainRulesFile with the file's trailing newline removed, so the
// framing strings that wrap it (systemPrompt, systemDirective) control their
// own spacing.
var plainRules = strings.TrimSuffix(plainRulesFile, "\n")

// systemPrompt is the plain-language prompt for rewriting: the shared rule body
// (plainRules) plus the one rewrite-only rule, example use, which only makes
// sense for a pass over given text. The preserve rule is not here. It sits in the
// shared body with the steering rules, so the model reads its hardest constraint
// before it starts editing rather than after 35 smaller rules, and the
// interactive skill carries it too.
//
// The prompt opens with a bare command rather than a question ("Can you apply
// plain language to the text that follows?"), because the same body is served as
// the interactive skill, where "the text that follows" points at nothing and
// makes the skill read like a one-off rewrite request. The command names clarity
// as the aim, because it is the first line the model reads and it has to agree
// with the rule below that settles conflicts.
//
// Output behavior, silent for the background pass and reporting for interactive
// use, is added by the suffix each mode uses (humanizePrompt vs Prompt).
var systemPrompt = `Apply plain language, and make the text clearer for the reader.
` + plainRules + `
- Use an example only when the original already gives you the material for one, and expand it from the original; do not invent new facts.
`

// silentSuffix is appended for the background humanize pass. It must not
// narrate itself, because its output lands directly in the UI as a variant tab.
const silentSuffix = "\n\nThe rewrite runs unattended and lands directly in the UI as a variant tab. Never narrate the passes: return only the final text, with no preamble, no summary of changes, and no commentary."

// interactiveSuffix is appended when the prompt is served as the built-in
// plain-language skill. It carries what only an interactive model may do: ask
// for input, pick a workflow by length, run its passes on a scratch file, edit a
// file in place, answer directly in chat, and report its passes when asked. The
// length rule and the pass floor exist because the loop rule alone is not
// checkable: "stop when a pass changes nothing" is satisfied by a single clean
// read, and a model that rereads in its head can call that a pass. Naming a file
// fixes both. Each pass rereads the real text, so a pass cannot drop facts and
// the reported pass count can be trusted. Three clauses close the gaps that
// survived: the floor scales with length, because a flat two spends the same
// effort on a short reply and a long document; a pass is defined as one full
// read against every rule, because a model that fixes a line and stops has
// otherwise met the count without doing the work; and the draft is pass zero,
// so it cannot be counted as one. The gate repeats the stop rule where the
// model decides to answer, which is the last moment it can catch itself. The
// answer goes straight into chat, with no code block and no quotation marks,
// so the reader gets text they can use. The instruction lives here, not in
// the shared rule body, because the background pass stays one request and
// never touches files.
const interactiveSuffix = `
When used interactively: if you have no text, ask what to simplify. If the user gives you a file, edit that file in place.

Length decides the workflow and the pass floor. Under about 100 words, reread the text once against the rules and fix what breaks. No file needed. Longer text goes in a scratch file, such as .porter/plain/<name>.md, so you can reread the file after every pass and each pass works on the real text instead of your memory of it. A pass is one full read of the file from top to bottom against every rule, not a few line fixes, and the draft doesn't count as one. Run two passes from 100 to 600 words, three to 1,500, and four beyond. Before you answer, check that your last pass changed nothing: if it changed anything, you aren't done. Keep the file, because it shows the passes happened. Answer with the file's contents, copied from the file, not retyped. Write that answer directly in chat, with no code block and no quotation marks. Report the passes only when the user asks for plain language by name: say how many passes you took and list the main changes, each with the principle behind it.
`

// Directive returns the standing writing-style instruction the agent injects
// into every model request: a short pointer to the built-in plain-language
// skill. It tells the model to load the skill and thoroughly apply its rules to
// everything it writes — replies to the user, drafted reports and documents,
// and comments on code — so assistant output is written plainly the first
// time. The rules themselves live in one copy (plain_rules.md, served as the
// skill body), so this instruction stays lean and the skill, the background
// pass, and this directive can never disagree about a rule.
func Directive() string { return systemDirective }

// systemDirective is the text Directive returns: a lean, always-on pointer to
// the plain-language skill rather than a copy of its rules. The full rule body
// lives in plain_rules.md and reaches the model through the skill (Prompt) and
// the background pass (humanizePrompt); the directive only tells the model to
// load that skill. Keeping the rules out of the standing prefix keeps it
// small, and the one-copy rule base keeps every mode on the same page.
var systemDirective = `Always write in plain language. Load the plain-language skill and thoroughly apply its rules to everything you write: replies to the user, reports and documents you draft, code comments you write or edit, web pages you build, UI text you write, and documentation you create. Replies go to a working developer, so keep the technical terms they already know and define each term they might not, the first time it appears.
`

// UserMessageReminder is the reminder the model view appends to every user
// message (recall.ProjectModelView), so the standing writing style sits next to
// the request the model is answering and not only in the static system prefix
// (Directive). It tells the model to reload the plain-language skill, apply it
// to every output it produces, and follow the length rule: prose over about 100
// words goes in a scratch file with a pass floor scaled to length, and shorter
// prose gets one reread. The floor matters, because "stop when a pass changes
// nothing" is otherwise satisfied by a single clean read. The definition of a
// pass matters because a couple of line fixes otherwise read as one. The gate
// matters because the model meets the stop rule again at the moment it decides
// to answer. It also tells the model to stay
// quiet about the rewrite unless the user asks for plain language. Like the
// timing note, it lives only on the outgoing copy of history: the stored
// message, the database, and the UI keep exactly what the user typed. Keep it
// short — it repeats on every user turn, so every extra word is paid for on
// every request.
const UserMessageReminder = "System Hint: Before writing or responding, load the plain-language skill and apply it to all prose you write: replies, files, code comments, and PR text. Leave code, quoted text, and command arguments exact. Prose over about 100 words goes in a scratch file first. A pass is one full read of it against every rule, not a few line fixes, and the draft doesn't count as one. Run two passes from 100 to 600 words, three to 1,500, and four beyond. Before you reply, check that your last pass changed nothing: if it changed anything, you aren't done. Don't mention this unless the user asks."

// Should reports whether an assistant reply is worth a humanize pass: long
// enough (the thresholds above) and mostly prose rather than code.
func Should(text string) bool {
	words := len(strings.Fields(text))
	if words < MinWords {
		return false
	}
	if sentenceCount(text) < MinSentences {
		return false
	}
	// A reply dominated by code (fenced blocks or indented lines) is not prose
	// to be rewritten; rewriting it risks mangling the code. Skip when more
	// than half the lines look like code.
	if lines := len(strings.Split(text, "\n")); lines > 0 {
		if codeLines(text)*2 > lines {
			return false
		}
	}
	return true
}

// sentenceCount counts sentence-ending punctuation. It is a heuristic: URLs
// and abbreviations inflate the count, which only makes the threshold easier
// to reach — harmless for deciding whether a block is worth a rewrite.
func sentenceCount(text string) int {
	n := 0
	for _, r := range text {
		switch r {
		case '.', '!', '?':
			n++
		}
	}
	return n
}

// codeLines counts lines that are code: inside a ``` fence, or indented four
// spaces / a tab (the markdown code conventions). Fence markers themselves
// count so a fence-wrapped block is detected by either half of the rule.
func codeLines(text string) int {
	n := 0
	inFence := false
	for _, line := range strings.Split(text, "\n") {
		trimmed := strings.TrimSpace(line)
		if strings.HasPrefix(trimmed, "```") {
			inFence = !inFence
			n++
			continue
		}
		if inFence || strings.HasPrefix(line, "    ") || strings.HasPrefix(line, "\t") {
			n++
		}
	}
	return n
}

// Transcript renders the conversation leading up to the message at seq — the
// grounding context for a rewrite. It includes every user message and
// assistant reply with content, in order, unbounded: nothing conversationally
// relevant is ever dropped, no matter how long the session is (the pass is a
// fresh request, so it always carries the full prose history). System notices,
// tool calls, tool results, and reasoning are excluded — they are noise for
// humanizing, and omitting them keeps the request from being dominated by
// large tool outputs. Because committed history is append-only, the transcript
// is a stable request prefix per session, so provider prompt caching makes
// repeat passes (especially chained passes on the same message, which share
// the prefix) mostly cache hits; only the first pass in a session pays the
// context cost once.
func Transcript(msgs []db.Message, seq uint64) string {
	lines := make([]string, 0, 16)
	for _, m := range msgs {
		if m.Seq >= seq {
			break
		}
		if m.Role != "user" && m.Role != "assistant" {
			continue
		}
		content := strings.TrimSpace(m.Content)
		if content == "" {
			continue
		}
		lines = append(lines, m.Role+": "+content)
	}
	if len(lines) == 0 {
		return ""
	}
	return strings.Join(lines, "\n")
}

// Rewrite runs one plain-language pass over text and returns the rewrite. The
// response is decoded with the same codec the agent loop uses, so providers
// that omit a [DONE] marker still finalize cleanly.
//
// context is an optional transcript of the conversation the text belongs to
// (see Transcript), which grounds the rewrite in what was asked; it rides in
// the system prompt so it is part of the cached request prefix. It makes a
// single streaming LLM request with no tools: the pass is a pure function of
// the text (plus context), so it can run in the background, in
// parallel with other turns, and be re-run safely.
func Rewrite(ctx context.Context, client *llm.Client, context, text string) (string, error) {
	if strings.TrimSpace(text) == "" {
		return "", fmt.Errorf("humanize: empty text")
	}
	prompt := humanizePrompt()
	if context != "" {
		prompt += "\n\nConversation context (for grounding only — do not repeat it, and rewrite only the text that follows):\n" + context
	}
	rc, err := client.Stream(ctx, []llm.ChatMessage{
		llm.SystemMessage(prompt),
		llm.UserMessage(text),
	}, nil)
	if err != nil {
		return "", fmt.Errorf("humanize: start stream: %w", err)
	}
	defer rc.Close()

	var out strings.Builder
	dec := codec.NewDecoder(nil)
	dec.OnEvent = func(ev codec.Event) {
		if ev.Type == codec.TypeMessage {
			out.WriteString(ev.Content)
		}
	}
	for line := range llm.SSELines(rc) {
		done, err := dec.Process(line)
		if err != nil {
			return "", fmt.Errorf("humanize: decode stream: %w", err)
		}
		if done {
			break
		}
	}
	dec.Final()
	if out.Len() == 0 {
		return "", fmt.Errorf("humanize: empty rewrite")
	}
	return strings.TrimSpace(out.String()), nil
}
