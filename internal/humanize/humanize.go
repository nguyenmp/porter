// Package humanize rewrites assistant replies in plain language, in the
// background, so assistant text blocks get a "Humanized" variant tab in the
// web UI without interrupting the turn. The pass is deliberately light: one
// LLM request carrying the text plus a full transcript of the conversation
// leading up to it (grounding for the rewrite, prefix-stable so provider
// prompt caching makes repeat passes cheap), driven by a hard-coded
// prompt distilled from the plain-language skill — a silent background step
// must be fast, deterministic, and cheap, not a skill-loading multi-fetch
// ritual.
package humanize

import (
	"context"
	"fmt"
	"strings"

	"porter/internal/codec"
	"porter/internal/db"
	"porter/internal/llm"
)

// PromptVersion identifies the prompt revision that produced a variant. It is
// stamped on every pass so the UI can explain why a tab reads the way it does,
// and should be bumped whenever the prompt below changes.
const PromptVersion = "plain-language-v2"

// Auto-pass thresholds: an assistant reply qualifies when it has at least
// MinWords words AND at least MinSentences sentences, and is not dominated by
// code blocks. Short replies and code dumps are not worth an automatic
// rewrite (they can still be humanized manually from the web UI's tab bar).
const (
	MinWords     = 60
	MinSentences = 4
)

// systemPrompt is the hard-coded plain-language instruction,distilled from
// the markdown files of the gsa/plainlanguage.gov git repository
// (https://github.com/gsa/plainlanguage.gov;the guidelines live under
// _pages/guidelines/,covering audience,words,sentences,voice,organization,
// and design),so prompt edits stay anchored in the source material rather
// than in memory. It condenses the guidance into a single-shot rewrite:one
// LLM request,no web fetches,and no pass commentary,because the pass runs
// unattended and its output lands directly in the UI.

const systemPrompt = `Rewrite the following text in plain language.
Plain language means an average reader of the intended audience understands the text on the first read. It is simple and direct, not simplistic or patronizing: do not talk down to the reader. Follow these rules:
- Speak directly to the reader: use you, and prefer the present tense. Use contractions where they sound natural.
- Choose simple, concrete, familiar words. Replace jargon, vague words, and filler (load-bearing, survived, rides, nobody, genuinely, outright, owed, rests, asymmetry, died, buys, settles, lever, byte-identical, delve, real) with everyday words. Cut unnecessary words: challenge every word, omit padding, and compress wordy phrases (in order to becomes to, at this point in time becomes now, is able to becomes can). Use must, not shall: must for an obligation, must not for a prohibition, may for a choice, should for a recommendation. Avoid and/or; write either, or, or both as intended. Define a technical term only when you must, and avoid abbreviations unless they are widely known; never redefine a common word to mean something other than its usual meaning.
- Use the active voice and the strongest, most direct form of each verb. Make the actor the subject: say we manage the program, not we are responsible for management of the program; say apply, not make an application. Avoid noun strings: say developing procedures to protect workers, not underground mine worker safety protection procedures development. Keep the subject, verb, and object close together, and put modifiers next to the words they modify: say you are required to provide only the following, not you are only required to provide the following. Put long conditions and exceptions after the main clause.
- Prefer short sentences carrying one clear idea each and short paragraphs covering one topic each. Use positive language: say what is true rather than what is not, and avoid double negatives and exceptions piled on exceptions. Use the same term for the same thing throughout; do not reach for synonyms just to sound varied.
- Lead with the main point: start with the answer and the top task for the reader, then the reasoning, exceptions, and details, and say why it matters to the reader. Make the structure skimmable: use useful headings, preferring question-form or specific headings over vague noun headings; use lists, and put steps in the order they happen; add simple visuals or tables (an if-then table suits conditional rules) where they help. Avoid nesting lists more than one level deep, and use bold sparingly for emphasis; do not use ALL CAPS or underlining. For longer, report-style replies, give the bottom line up front, separate distinct audiences and topics into their own sections, and keep related material together rather than referring readers elsewhere. Write link text that says exactly where the link leads; do not use click here or read more. Use transition words sparingly, and write out for example or such as instead of e.g. or i.e. Use an example only when the original already gives you the material for one, and expand it from the original; do not invent new facts.
- Self-review before finishing: read the rewrite as if it were the original and apply every rule above again. Each fresh pass catches what the last one missed, so keep iterating until a further pass would change nothing. Never narrate the passes; return only the final text.
- Preserve the message. You are free to reformat, rephrase, and reorganize, and to add headings, lists, or simple visuals, to improve clarity, but do not change what the original conveys. Keep every fact, name, number, and date as given: do not add, drop, or alter anything in the original. Keep code, URLs, and quoted text exact. Return only the rewritten text: no preamble, no summary of changes, no commentary.
`

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
	prompt := systemPrompt
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
