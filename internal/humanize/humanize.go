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
// each mode appends only the behavior its own job needs. The same rule body
// is the standing style the agent injects into every request (Directive):
// replies, drafted reports, and code comments are written plainly the first
// time, and the rewrite pass becomes a safety net rather than the only
// plain-language path. A one-line reminder (UserMessageReminder) rides on the
// conversation too: the model view appends it to every user message, so the
// style sits next to the request being answered and not only in the system
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
const PromptVersion = "plain-language-v8"

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
		Description: "Rewrite or write text in plain language: simple, direct, reader-first prose (from the US government plainlanguage.gov guidelines). Load to get the full rewrite rules.",
		Path:        api.BuiltinPrefix + SkillName,
	}
}

// Prompt returns the body of the built-in plain-language skill:the shared
// rewrite rules (systemPrompt) plus an interactive-use suffix. A model
// that loads this skill asks for text when none is given, edits files in
// place,and reports how many passes it took and what it changed - things a
// background pass must never do.
func Prompt() string { return systemPrompt + interactiveSuffix }

// humanizePrompt returns the prompt the background humanize pass uses:the
// shared rewrite rules (systemPrompt) plus a silent-output suffix. The
// pass runs unattended and its result lands directly in the UI as a variant
// tab,so the model must return only the rewritten text:no preamble,no
// summary of changes,no commentary.
func humanizePrompt() string { return systemPrompt + silentSuffix }

// Auto-pass thresholds: an assistant reply qualifies when it has at least
// MinWords words AND at least MinSentences sentences, and is not dominated by
// code blocks. Short replies and code dumps are not worth an automatic
// rewrite (they can still be humanized manually from the web UI's tab bar).
const (
	MinWords     = 60
	MinSentences = 4
)

// plainRules is the plain-language rule body both prompts share: the
// definition, the rule that decides where to spend effort, and the writing
// rules. systemPrompt (rewriting) and systemDirective (fresh writing) wrap it
// with their own framing — a rewrite pass must preserve the given text, while
// fresh writing just follows the rules as it goes — so the shared guidance can
// never drift between the two.
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

// systemPrompt is the plain-language prompt for rewriting: the shared rule
// body (plainRules) plus the rewrite-only rules — example use and preserving
// the message — which only make sense for a pass over given text. It opens as
// a request rather than a bare command ("Can you apply plain language ..."),
// which restates the effort rule that leads plainRules: aim at the
// improvements a reader would notice rather than reworking every sentence.
// Output behavior, silent for the background pass and reporting for
// interactive use, is added by the suffix each mode uses (humanizePrompt vs
// Prompt).
var systemPrompt = `Can you apply plain language to the text that follows, focusing on the biggest improvements?
` + plainRules + `
- Use an example only when the original already gives you the material for one, and expand it from the original; do not invent new facts.
- Preserve the message. You are free to reformat, rephrase, and reorganize, and to add headings, lists, or simple visuals, to improve clarity, but do not change what the original conveys. Keep every fact, name, number, and date as given: do not add, drop, or alter anything in the original. Keep code, URLs, and quoted text exact.
`

// silentSuffix is appended for the background humanize pass:it must not
// narrate itself,ecause its output lands directly in the UI as a variant tab.
const silentSuffix = "\n\nThe rewrite runs unattended and lands directly in the UI as a variant tab.Never narrate the passes; return only the final text:no preamble,no summary of changes,no commentary."

// interactiveSuffix is appended when the prompt is served as the built-in
// plain-language skill:an interactive model can ask for input,edit files,
// and report its passes - things the silent background pass must never do.
const interactiveSuffix = "\n\nWhen used interactively:if you have no text,ask what to simplify;if the input is a file,edit the file in place. Tell the user when you start a new pass;after the rewrite,say how many passes you took,and list the main changes,each with the principle behind it."

// Directive returns the standing writing-style instruction the agent injects
// into every model request, so assistant output — replies to the user, drafted
// reports, and comments on code — is written in plain language the first time
// rather than only rewritten afterwards. It wraps the shared rule body
// (plainRules) that systemPrompt also uses, adding only the generation
// framing: where the style applies and for whom. Every rule is one copy, so
// the two prompts cannot drift.
func Directive() string { return systemDirective }

// systemDirective is the text Directive returns: the standing plain-language
// writing style for generation — the shared rule body (plainRules) plus the
// generation-only framing. See Directive for why it wraps plainRules like
// systemPrompt does.
var systemDirective = `Always speak in plain language. Before replying, reflect on what you are about to say and revise it. Apply this style to everything you write, including replies to the user, reports and documents you draft, code comments you write or edit, web pages you build, UI text you write, and documentation you create.
` + plainRules + `
Audience: replies and UI text go to the user, a working developer, so keep the technical terms they already know — do not define them, and do not pad to fill space. Text you draft for other readers, like a report, web page, or update, fits that audience instead: keep the technical terms they need and define each one the first time you use it. Code comments are prose too: short, direct, and about the code they sit in.
`

// UserMessageReminder is the one-line plain-language reminder the model view
// appends to every user message (recall.ProjectModelView), so the style also
// sits next to the request the model is answering and not only in the static
// system prefix (Directive). Like the timing note, it lives only on the
// outgoing copy of history: the stored message, the database, and the UI keep
// exactly what the user typed. Keep it short — it repeats on every user turn,
// so every extra word is paid for on every request.
const UserMessageReminder = "Please write prose simply and in plain language"

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
