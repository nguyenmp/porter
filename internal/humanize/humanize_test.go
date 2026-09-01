package humanize

import (
	"strings"
	"testing"
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
