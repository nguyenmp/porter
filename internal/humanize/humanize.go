// Package humanize rewrites assistant replies in plain language, in the
// background, so long assistant text blocks get a "Humanized" variant tab in
// the web UI without interrupting the turn. The pass is deliberately
// self-contained: one LLM request carrying just the text (no session context,
// no tools, no history), driven by a hard-coded prompt distilled from the
// plain-language skill — a silent background step must be fast, deterministic,
// and cheap, not a skill-loading multi-fetch ritual.
package humanize

import (
	"context"
	"fmt"
	"strings"

	"porter/internal/codec"
	"porter/internal/llm"
)

// PromptVersion identifies the prompt revision that produced a variant. It is
// stamped on every pass so the UI can explain why a tab reads the way it does,
// and should be bumped whenever the prompt below changes.
const PromptVersion = "plain-language-v1"

// Auto-pass thresholds: an assistant reply qualifies when it has at least
// MinWords words AND at least MinSentences sentences, and is not dominated by
// code blocks. Short replies and code dumps are not worth a rewrite.
const (
	MinWords     = 60
	MinSentences = 4
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

// Rewrite runs one plain-language pass over text and returns the rewrite. It
// makes a single streaming LLM request with no tools and no conversation
// context: the pass is a pure function of the text, so it can run in the
// background, in parallel with other turns, and be re-run safely. The response
// is decoded with the same codec the agent loop uses, so providers that omit a
// [DONE] marker still finalize cleanly.
func Rewrite(ctx context.Context, client *llm.Client, text string) (string, error) {
	if strings.TrimSpace(text) == "" {
		return "", fmt.Errorf("humanize: empty text")
	}
	rc, err := client.Stream(ctx, []llm.ChatMessage{
		llm.SystemMessage(systemPrompt),
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
