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

// TestDirectiveNamesSurfaces verifies the standing writing style explicitly
// names the surfaces it governs — replies, drafted reports, and code comments —
// so the always-on instruction is concrete about where it applies rather than a
// vague "be clear" note. It also checks the directive carries no rewrite-pass
// language: the pass path may narrate passes and variants, but the generation
// directive must not leak that machinery into everyday writing instructions.
func TestDirectiveNamesSurfaces(t *testing.T) {
	d := Directive()
	for _, want := range []string{"plain language", "replies", "reports", "comments"} {
		if !strings.Contains(d, want) {
			t.Errorf("directive missing %q:\n%s", want, d)
		}
	}
	for _, banned := range []string{"passes", "rewrite", "variant", "Humanized"} {
		if strings.Contains(d, banned) {
			t.Errorf("directive leaked rewrite-pass language %q:\n%s", banned, d)
		}
	}
}

// TestPlainRulesEmbedded covers the embedded rule body and the way the two
// rule-carrying prompts (the skill and the background pass) join it: the rules
// must load from plain_rules.md (so that file is the one copy a reader can
// link to), and the text each prompt adds after the rules must start on its
// own line rather than glue onto the last rule. The standing directive is not
// in this set: it points at the skill instead of carrying the rules.
func TestPlainRulesEmbedded(t *testing.T) {
	if !strings.Contains(plainRules, "Plain language means an average reader") {
		t.Fatalf("embedded rules are missing the definition:\n%s", plainRules)
	}
	if strings.HasSuffix(plainRules, "\n") {
		t.Errorf("plainRules kept the file's trailing newline; the framing strings set the spacing")
	}
	for name, prompt := range map[string]string{
		"humanizePrompt": humanizePrompt(),
		"Prompt":         Prompt(),
	} {
		if strings.Contains(prompt, "goals?-") {
			t.Errorf("%s glued the text after the rules onto the last rule:\n%s", name, prompt)
		}
		if !strings.Contains(prompt, "meet their goals?") {
			t.Errorf("%s does not carry the end of the rule body", name)
		}
	}
}

// TestCiteRuleInRulesBody checks the citation rule lives in the shared rule
// body, plain_rules.md, so the rule-carrying prompts get it from one copy: the
// skill and the background rewrite pass. The standing directive points at the
// skill rather than carrying the rules, so it is not checked here.
func TestCiteRuleInRulesBody(t *testing.T) {
	if !strings.Contains(plainRules, "Cite your sources") {
		t.Fatalf("plain_rules.md is missing the citation rule:\n%s", plainRules)
	}
	for name, prompt := range map[string]string{
		"humanizePrompt": humanizePrompt(),
		"Prompt":         Prompt(),
	} {
		if !strings.Contains(prompt, "Cite your sources") {
			t.Errorf("%s does not carry the citation rule", name)
		}
	}
}

// TestDirectivePointsAtSkill verifies the standing style stays lean: it names
// the plain-language skill and tells the model to load it, rather than
// restating the rules. The rule body lives in one copy (plain_rules.md),
// carried by the skill and the background pass, so the directive must not
// duplicate it.
func TestDirectivePointsAtSkill(t *testing.T) {
	d := Directive()
	if !strings.Contains(d, "plain-language skill") {
		t.Errorf("directive does not name the plain-language skill:\n%s", d)
	}
	if !strings.Contains(d, "Load") {
		t.Errorf("directive does not tell the model to load the skill:\n%s", d)
	}
	for _, rule := range []string{"Cite your sources", "meet their goals?", "Plain language means an average reader"} {
		if strings.Contains(d, rule) {
			t.Errorf("directive restates the rule body (%q); it should point at the skill instead:\n%s", rule, d)
		}
	}
}

// TestInteractiveRunsPassesOnFile checks the skill's interactive suffix tells
// the model to run its passes on a scratch file and reread it after each pass,
// so the loop rule works on the real text, not the model's memory of it.
// The background pass must stay one request, so this instruction lives only in
// the interactive suffix and never in the shared rule body.
func TestInteractiveRunsPassesOnFile(t *testing.T) {
	p := Prompt()
	for _, want := range []string{"scratch file", "reread the file after every pass"} {
		if !strings.Contains(p, want) {
			t.Errorf("skill prompt missing %q:\n%s", want, p)
		}
	}
	if strings.Contains(humanizePrompt(), "scratch file") {
		t.Errorf("background pass prompt leaked the interactive file workflow:\n%s", humanizePrompt())
	}
}

// TestInteractiveSetsWorkflowAndLengthRule checks the skill carries the workflow
// clauses it needs: the length cutoff that keeps short replies out of a file,
// what a pass is, that the draft is not one, the gate before the answer, and the
// answer format. Without the definition, two line fixes pass for a pass. Without
// the cutoff, short replies pay for a file they do not need. It sets no count of
// passes: the tradeoff test ends the loop.
func TestInteractiveSetsWorkflowAndLengthRule(t *testing.T) {
	p := Prompt()
	for _, want := range []string{
		"A pass is one full read",
		"the draft doesn't count as one",
		"Before you answer, check",
		"straight fix means one more pass",
		"about 100 words",
		"Keep the file",
		"copied from the file",
	} {
		if !strings.Contains(p, want) {
			t.Errorf("skill prompt missing %q:\n%s", want, p)
		}
	}
	// The background pass stays one request, so it must not carry the file
	// workflow or the interactive length rule.
	for _, banned := range []string{"scratch file", "100 words", "pass is one full read"} {
		if strings.Contains(humanizePrompt(), banned) {
			t.Errorf("background pass prompt leaked %q:\n%s", banned, humanizePrompt())
		}
	}
}

// TestUserMessageReminderCarriesWorkflow checks the per-turn reminder states the
// file, the definition of a pass, the length rule, and the stop test. The
// reminder repeats on every turn, so it has to carry them in a few words.
func TestUserMessageReminderCarriesWorkflow(t *testing.T) {
	for _, want := range []string{
		"scratch file",
		"pass is one full read",
		"draft doesn't count as one",
		stopRule,
		"100 words",
	} {
		if !strings.Contains(UserMessageReminder, want) {
			t.Errorf("UserMessageReminder missing %q: %s", want, UserMessageReminder)
		}
	}
}

// TestReminderMatchesSkillOnStopRule checks the reminder and the skill agree on
// the two parts of the stop test a model can talk itself out of: what counts as
// a pass, and the test that ends the loop. Both surfaces are written by hand at
// different lengths, so a drift here would let the shorter one quietly weaken
// the rule.
func TestReminderMatchesSkillOnStopRule(t *testing.T) {
	shared := []string{"a pass is one full read", "trades one quality for another", "understandability first"}
	for _, want := range shared {
		for name, text := range map[string]string{"reminder": UserMessageReminder, "skill": Prompt()} {
			if !strings.Contains(strings.ToLower(text), want) {
				t.Errorf("%s missing %q: %s", name, want, text)
			}
		}
	}
}

// TestPromptSetsNoPassCount checks neither surface sets a number of passes. A
// count keyed to length spends the same passes on a short reply as on a long
// one, and a text that reads well after two passes paid for four. The stop test
// (stopRule) ends the loop instead, so a phrase like "four passes" here means a
// schedule has crept back in.
func TestPromptSetsNoPassCount(t *testing.T) {
	numbers := []string{"two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"}
	for name, text := range map[string]string{
		"skill":      Prompt(),
		"reminder":   UserMessageReminder,
		"background": humanizePrompt(),
	} {
		lower := strings.ToLower(text)
		for _, n := range numbers {
			if strings.Contains(lower, n+" passes") {
				t.Errorf("%s sets a pass count (%q):\n%s", name, n+" passes", text)
			}
		}
	}
}

// TestLoopRuleInRulesBody checks the rule to work through every rule, one at a
// time, in a loop until the text is thoroughly plain lives in the shared rule
// body (plain_rules.md), so the skill and the background pass both carry it
// from one copy.
func TestLoopRuleInRulesBody(t *testing.T) {
	if !strings.Contains(plainRules, "loop") {
		t.Fatalf("plain_rules.md is missing the loop rule:\n%s", plainRules)
	}
	if !strings.Contains(Prompt(), "loop") {
		t.Errorf("the plain-language skill does not carry the loop rule:\n%s", Prompt())
	}
}

// TestBrevityRuleInRulesBody checks the brevity tiebreaker — prefer fewer
// words unless more words add clarity — lives in the shared rule body
// (plain_rules.md), so the skill and the background pass both carry it from one
// copy. The standing directive points at the skill rather than carrying the
// rules, so it is not checked here.
func TestBrevityRuleInRulesBody(t *testing.T) {
	if !strings.Contains(plainRules, "Prefer fewer words") {
		t.Fatalf("plain_rules.md is missing the brevity rule:\n%s", plainRules)
	}
	for name, prompt := range map[string]string{
		"humanizePrompt": humanizePrompt(),
		"Prompt":         Prompt(),
	} {
		if !strings.Contains(prompt, "Prefer fewer words") {
			t.Errorf("%s does not carry the brevity rule", name)
		}
	}
}

// TestRulesRunLargestConcernFirst checks the rule body reads biggest to
// smallest: the sections start with structure and end with review, and the
// preserve rule sits ahead of the sections so the model meets its hardest
// constraint before it starts editing. A reorder is the point of the file, so a
// silent reshuffle should fail here rather than only in a reply.
func TestRulesRunLargestConcernFirst(t *testing.T) {
	sections := []string{"## Structure", "## Sentences", "## Words", "## Links", "## Emphasis", "## When the text is a page", "## Review"}
	last := 0
	for _, s := range sections {
		i := strings.Index(plainRules, s)
		if i < 0 {
			t.Fatalf("plain_rules.md is missing the %q section:\n%s", s, plainRules)
		}
		if i < last {
			t.Errorf("section %q runs before the section above it; sections must run largest concern first", s)
		}
		last = i
	}
	preserve := strings.Index(plainRules, "Preserve the message")
	if preserve < 0 {
		t.Fatalf("plain_rules.md is missing the preserve rule:\n%s", plainRules)
	}
	if preserve > strings.Index(plainRules, "## Structure") {
		t.Errorf("the preserve rule trails the section rules; it must sit in the steering block at the top")
	}
}

// TestNegationRulesInRulesBody checks the negation rules live in the shared rule
// body (plain_rules.md), so the skill and the background pass both carry them
// from one copy: flag every sentence that says what something is not, rewrite it
// in the positive, and reread the text for negation words before finishing. A
// "not X but Y" sentence reads smoothly and survives a reread, which is why the
// sweep is mechanical rather than left to the read.
func TestNegationRulesInRulesBody(t *testing.T) {
	for _, want := range []string{
		"Flag every sentence that says what something is not",
		"Run a keyword sweep",
	} {
		if !strings.Contains(plainRules, want) {
			t.Fatalf("plain_rules.md is missing %q:\n%s", want, plainRules)
		}
		for name, prompt := range map[string]string{
			"humanizePrompt": humanizePrompt(),
			"Prompt":         Prompt(),
		} {
			if !strings.Contains(prompt, want) {
				t.Errorf("%s does not carry %q", name, want)
			}
		}
	}
}

// TestScorecardRuleInRulesBody checks the scorecard step lives in the shared
// rule body, next to the loop rule it makes checkable: a writer that has to
// quote what is left cannot call a text done on a hunch. The step names no
// output format, because the skill keeps its scorecard in the scratch file and
// the background pass must stay silent.
func TestScorecardRuleInRulesBody(t *testing.T) {
	const want = "score the sections above in turn"
	if !strings.Contains(plainRules, want) {
		t.Fatalf("plain_rules.md is missing the scorecard step:\n%s", plainRules)
	}
	for name, prompt := range map[string]string{
		"humanizePrompt": humanizePrompt(),
		"Prompt":         Prompt(),
	} {
		if !strings.Contains(prompt, want) {
			t.Errorf("%s does not carry the scorecard step", name)
		}
	}
}

// TestSilentPassSuppressesScorecard checks the unattended pass is told to keep
// the scorecard out of its reply. The pass returns text straight into a variant
// tab, so a card that leaks is user-visible.
func TestSilentPassSuppressesScorecard(t *testing.T) {
	if !strings.Contains(humanizePrompt(), "no scorecard") {
		t.Errorf("the background pass does not suppress the scorecard:\n%s", humanizePrompt())
	}
	if strings.Contains(Prompt(), "no scorecard") {
		t.Errorf("the interactive skill must not suppress the scorecard; it keeps one in the scratch file")
	}
}
