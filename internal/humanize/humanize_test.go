package humanize

import (
	"strings"
	"testing"

	"porter/internal/db"
	"porter/internal/llm"
)

// longProse builds a reply that clears both thresholds: 6 sentences, well over
// 60 words, no code.
func longProse() string {
	return strings.Repeat("This is a perfectly ordinary sentence with more than enough words to qualify. ", 6)
}

func TestShouldQualifiesLongProse(t *testing.T) {
	if !Should(longProse()) {
		t.Errorf("long prose reply should qualify for a humanize pass")
	}
}

func TestShouldSkipsShortReplies(t *testing.T) {
	if Should("Just a quick answer.") {
		t.Errorf("short reply should not qualify")
	}
}

func TestShouldRequiresBothThresholds(t *testing.T) {
	// Many words but no sentence punctuation (a run-on): fails the sentence count.
	runon := strings.Repeat("no punctuation here just words ", 40)
	if Should(runon) {
		t.Errorf("run-on with no sentence punctuation should not qualify")
	}
	// Many sentences but too few words.
	tiny := strings.Repeat("Hi. ", 10)
	if Should(tiny) {
		t.Errorf("high-sentence low-word reply should not qualify")
	}
}

func TestShouldSkipsCodeDominated(t *testing.T) {
	var b strings.Builder
	b.WriteString("```go\n")
	for i := 0; i < 20; i++ {
		b.WriteString("func f() int { return 1 }\n")
	}
	b.WriteString("```\n")
	b.WriteString(longProse())
	if Should(b.String()) {
		t.Errorf("code-dominated reply should not qualify")
	}
}

func TestShouldAllowsProseWithSomeCode(t *testing.T) {
	var b strings.Builder
	b.WriteString("Here is an explanation with some code.\n\n")
	b.WriteString("```go\nx := 1\n```\n\n")
	b.WriteString(longProse())
	if !Should(b.String()) {
		t.Errorf("prose reply with a small code block should still qualify")
	}
}

// TestTranscriptIncludesConversation verifies the context builder keeps the
// human-readable thread (user + assistant prose) up to the target message,
// excluding tool traffic and system notices.
func TestTranscriptIncludesConversation(t *testing.T) {
	msgs := []db.Message{
		{Seq: 1, ChatMessage: llm.UserMessage("explain this")},
		{Seq: 2, ChatMessage: llm.SystemMessage("exec notice")},
		{Seq: 3, ChatMessage: llm.AssistantMessage("a draft answer", "", nil)},
		{Seq: 4, ChatMessage: llm.ToolResult("c1", "some tool output")},
		{Seq: 5, ChatMessage: llm.AssistantMessage("the reply being rewritten", "", nil)},
	}
	got := Transcript(msgs, 5)
	if !strings.Contains(got, "user: explain this") {
		t.Errorf("transcript missing user message: %q", got)
	}
	if !strings.Contains(got, "assistant: a draft answer") {
		t.Errorf("transcript missing earlier assistant reply: %q", got)
	}
	if strings.Contains(got, "exec notice") || strings.Contains(got, "tool output") {
		t.Errorf("transcript leaked tool/system traffic: %q", got)
	}
	if strings.Contains(got, "reply being rewritten") {
		t.Errorf("transcript included the target message itself: %q", got)
	}
}

// TestTranscriptUnbounded ensures the full prose history survives: no message
// count cap and no per-message truncation — a long session's entire
// user/assistant thread is included, so grounding is never dropped.
func TestTranscriptUnbounded(t *testing.T) {
	// A long session: every prose message must appear, oldest first.
	var msgs []db.Message
	for i := 1; i <= 20; i++ {
		msgs = append(msgs, db.Message{Seq: uint64(i), ChatMessage: llm.UserMessage("m")})
	}
	got := Transcript(msgs, 21)
	if strings.Count(got, "user: m") != 20 {
		t.Errorf("transcript has %d messages, want all 20", strings.Count(got, "user: m"))
	}
	if !strings.HasPrefix(got, "user: m") {
		t.Errorf("transcript does not start with the oldest message: %q", got)
	}

	// A single oversized message is not truncated.
	long := strings.Repeat("x", 10000)
	msgs2 := []db.Message{{Seq: 1, ChatMessage: llm.UserMessage(long)}}
	got2 := Transcript(msgs2, 2)
	if !strings.Contains(got2, strings.Repeat("x", 10000)) {
		t.Errorf("transcript dropped content from an oversized message (len %d)", len(got2))
	}
	if strings.Contains(got2, "[truncated]") {
		t.Errorf("transcript marked truncation; the unbounded transcript never truncates: %q", got2[:50])
	}
}

// TestTranscriptEmptyWithoutContext returns "" for no qualifying messages.
func TestTranscriptEmptyWithoutContext(t *testing.T) {
	if got := Transcript(nil, 1); got != "" {
		t.Errorf("empty transcript = %q, want \"\"", got)
	}
	var msgs []db.Message
	for i := 1; i <= 3; i++ {
		msgs = append(msgs, db.Message{Seq: uint64(i), ChatMessage: llm.ToolResult("c", "out")})
	}
	if got := Transcript(msgs, 4); got != "" {
		t.Errorf("tool-only transcript = %q, want \"\"", got)
	}
}
