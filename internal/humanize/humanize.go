// Package humanize rewrites assistant replies in plain language, in the
// background, so assistant text blocks get a "Humanized" variant tab in the
// web UI without interrupting the turn. The pass is deliberately light: one
// LLM request carrying the text plus a compact transcript of the conversation
// leading up to it (grounding for the rewrite, bounded and prefix-stable so
// provider prompt caching makes repeat passes cheap), driven by a hard-coded
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
const PromptVersion = "plain-language-v1"

// Auto-pass thresholds: an assistant reply qualifies when it has at least
// MinWords words AND at least MinSentences sentences, and is not dominated by
// code blocks. Short replies and code dumps are not worth an automatic
// rewrite (they can still be humanized manually from the web UI's tab bar).
const (
	MinWords     = 60
	MinSentences = 4
	// MaxContextMessages bounds how many prior conversation messages the
	// rewrite's transcript includes, and MaxContextRunes caps each message's
	// contribution, so the context prefix stays cheap. Because the transcript
	// is built from committed history — append-only — it is a stable prefix
	// per session, so provider prompt caching makes repeat passes (especially
	// chained passes on the same message, which share the prefix) mostly cache
	// hits; only the first pass in a session pays the context cost once.
	MaxContextMessages = 12
	MaxContextRunes    = 2000
)

// systemPrompt is the hard-coded plain-language instruction. It distills the
// plain-language skill's steps (simplify the words, not the content; keep
// facts and technical precision; split long sentences; avoid jargon) into a
// single-shot rewrite with no web fetches and no pass commentary, because the
// pass runs unattended and its output lands directly in the UI.
const systemPrompt = `Rewrite the following text in plain language.

Plain language means an average reader understands the text on the first read. Follow these rules:

- Simplify the words, not the content. Keep every fact, name, number, date, and technical term, and define a technical term the first time you use it.
- Prefer short, clear sentences and the active voice. Split long sentences.
- Replace jargon, vague words, and AI-sounding filler (delve, furthermore, in conclusion, it is important to note, etc.) with everyday words.
- Keep the meaning and the original structure: preserve markdown headings, lists, bold, links, and code blocks exactly. Do not change code, URLs, or quoted text.
- Do not add anything that is not in the original, and do not drop anything that is.
- Return only the rewritten text. No preamble, no summary of changes, no commentary.`

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

// Transcript renders a compact, human-readable transcript of the conversation
// leading up to the message at seq — the grounding context for a rewrite.
// Only user messages and assistant replies with content are included: system
// notices, tool calls, tool results, and reasoning are excluded (they are
// noise for humanizing, and omitting them keeps the request small). Only the
// most recent MaxContextMessages are kept, each capped at MaxContextRunes, so
// the prefix is bounded no matter how long the session is.
func Transcript(msgs []db.Message, seq uint64) string {
	lines := make([]string, 0, MaxContextMessages)
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
		lines = append(lines, m.Role+": "+truncateRunes(content, MaxContextRunes))
	}
	if len(lines) > MaxContextMessages {
		lines = lines[len(lines)-MaxContextMessages:]
	}
	if len(lines) == 0 {
		return ""
	}
	return strings.Join(lines, "\n")
}

// truncateRunes caps s at max runes, appending a marker when it cut anything.
func truncateRunes(s string, max int) string {
	r := []rune(s)
	if len(r) <= max {
		return s
	}
	return string(r[:max]) + " …[truncated]"
}

// Rewrite runs one plain-language pass over text and returns the rewrite. The
// response is decoded with the same codec the agent loop uses, so providers
// that omit a [DONE] marker still finalize cleanly.
//
// context is an optional transcript of the conversation the text belongs to
// (see Transcript), which grounds the rewrite in what was asked; it rides in
// the system prompt so it is part of the cached request prefix. It makes a
// single streaming LLM request with no tools: the pass is a pure function of
// the text (plus bounded context), so it can run in the background, in
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
