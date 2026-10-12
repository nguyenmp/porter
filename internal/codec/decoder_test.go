package codec

import (
	"bytes"
	"strings"
	"testing"
)

func feed(t *testing.T, lines ...string) (string, error) {
	t.Helper()
	var buf bytes.Buffer
	dec := NewDecoder(NewEncoder(&buf))
	for i, line := range lines {
		done, err := dec.Process(line)
		if err != nil {
			return buf.String(), err
		}
		if done {
			if i != len(lines)-1 {
				t.Logf("stopped early at line %d of %d", i+1, len(lines))
			}
			break
		}
	}
	return buf.String(), nil
}

func TestDecoderAccumulatesAndTerminates(t *testing.T) {
	got, err := feed(t,
		`data: {"type":"response.output_text.delta","output_index":0,"delta":"Hel"}`,
		`data: {"type":"response.output_text.delta","output_index":0,"delta":"lo"}`,
		`data: {"type":"response.completed","response":{"usage":{"input_tokens":10,"output_tokens":20}}}`,
		`data: [DONE]`,
	)
	if err != nil {
		t.Fatalf("Process: %v", err)
	}
	for _, want := range []string{
		`"type":"message_delta"`, `"delta":"Hel"`, `"delta":"lo"`,
		`"type":"message"`, `"content":"Hello"`,
		`"type":"usage"`, `"input_tokens":10`, `"output_tokens":20`,
	} {
		if !strings.Contains(got, want) {
			t.Errorf("output missing %q; got:\n%s", want, got)
		}
	}
}

func TestDecoderDoneMarkerStopsStream(t *testing.T) {
	got, err := feed(t,
		`data: {"type":"response.output_text.delta","output_index":0,"delta":"hi"}`,
		`data: [DONE]`,
	)
	if err != nil {
		t.Fatalf("Process: %v", err)
	}
	if !strings.Contains(got, `"type":"message"`) || !strings.Contains(got, `"content":"hi"`) {
		t.Errorf("missing message event; got:\n%s", got)
	}
}

func TestDecoderReasoningStaysSeparate(t *testing.T) {
	got, err := feed(t,
		// Reasoning and content arrive as separate streams of deltas.
		`data: {"type":"response.reasoning_summary_text.delta","output_index":0,"delta":"think"}`,

		`data: {"type":"response.output_text.delta","output_index":0,"delta":"Answer"}`,
		`data: {"type":"response.reasoning_summary_text.delta","output_index":0,"delta":"ing"}`,
		`data: {"type":"response.reasoning_summary_text.delta","output_index":0,"delta":"."}`,

		`data: {"type":"response.output_text.delta","output_index":0,"delta":" is 4"}`,
		`data: [DONE]`,
	)
	if err != nil {
		t.Fatalf("Process: %v", err)
	}
	for _, want := range []string{
		`"type":"reasoning_delta"`, `"reasoning":"think"`, `"reasoning":"ing"`, `"reasoning":"."`,
		`"type":"message"`, `"content":"Answer is 4"`, `"reasoning":"thinking."`,
	} {
		if !strings.Contains(got, want) {
			t.Errorf("output missing %q; got:\n%s", want, got)
		}
	}
}

func TestDecoderToolCallAccumulatesAcrossDeltas(t *testing.T) {
	got, err := feed(t,
		// Tool call split over three deltas: id/name first, then args chunks.
		`data: {"type":"response.output_item.added","output_index":0,"item":{"type":"function_call","call_id":"call_1","name":"shell"}}`,
		`data: {"type":"response.function_call_arguments.delta","output_index":0,"delta":"echo "}`,
		`data: {"type":"response.function_call_arguments.delta","output_index":0,"delta":"hi"}`,
		`data: [DONE]`,
	)
	if err != nil {
		t.Fatalf("Process: %v", err)
	}
	for _, want := range []string{
		`"type":"tool_call"`, `"name":"shell"`, `"arguments":"echo hi"`,
	} {
		if !strings.Contains(got, want) {
			t.Errorf("output missing %q; got:\n%s", want, got)
		}
	}
}

func TestDecoderEmitsToolCallDeltasLive(t *testing.T) {
	got, err := feed(t,
		// A tool call split over three deltas: id/name first, then args chunks.
		`data: {"type":"response.output_item.added","output_index":0,"item":{"type":"function_call","call_id":"call_1","name":"shell"}}`,
		`data: {"type":"response.function_call_arguments.delta","output_index":0,"delta":"echo "}`,
		`data: {"type":"response.function_call_arguments.delta","output_index":0,"delta":"hi"}`,
		`data: [DONE]`,
	)
	if err != nil {
		t.Fatalf("Process: %v", err)
	}
	// Each non-empty fragment must surface as a live delta carrying exactly its
	// slice of the call, in addition to the assembled final tool_call.
	for _, want := range []string{
		`"type":"tool_call_delta"`, `"tool_call_id":"call_1"`, `"name":"shell"`,
		`"type":"tool_call_delta"`, `"arguments":"echo "`,
		`"type":"tool_call_delta"`, `"arguments":"hi"`,
		`"type":"tool_call"`, `"arguments":"echo hi"`,
	} {
		if !strings.Contains(got, want) {
			t.Errorf("output missing %q; got:\n%s", want, got)
		}
	}
}

// TestDecoderSkipsEventsWithNothingNew keeps providers from causing spurious
// re-renders: an event that carries no fresh field must not emit a delta. Here
// the completed item repeats the id and name porter already has, so only the
// two events that did carry something new may emit.
func TestDecoderSkipsEventsWithNothingNew(t *testing.T) {
	got, err := feed(t,
		`data: {"type":"response.output_item.added","output_index":0,"item":{"type":"function_call","call_id":"call_1","name":"shell"}}`,
		`data: {"type":"response.function_call_arguments.delta","output_index":0,"delta":"hi"}`,
		// A completed item that carries nothing new must not emit a delta.
		`data: {"type":"response.output_item.done","output_index":0,"item":{"type":"function_call","call_id":"call_1","name":"shell"}}`,
		`data: [DONE]`,
	)
	if err != nil {
		t.Fatalf("Process: %v", err)
	}
	if n, want := strings.Count(got, `"type":"tool_call_delta"`), 2; n != want {
		t.Errorf("delta count = %d, want %d (an event with nothing new must not emit); got:\n%s", n, want, got)
	}
}

func TestDecoderRejectsNonDataLine(t *testing.T) {
	if _, err := feed(t, "event: ping"); err == nil {
		t.Fatal("expected error for non-data line")
	}
}

func TestDecoderRejectsMalformedJSON(t *testing.T) {
	if _, err := feed(t, "data: {not json}"); err == nil {
		t.Fatal("expected error for malformed JSON")
	}
}

// TestDecoderReadsUsageFromCompletedEvent covers the Responses ending: the
// token totals ride on the response.completed event, which is also the end of
// the stream.
func TestDecoderReadsUsageFromCompletedEvent(t *testing.T) {
	got, err := feed(t,
		`data: {"type":"response.output_text.delta","output_index":0,"delta":"Hi"}`,
		`data: {"type":"response.completed","response":{"usage":{"input_tokens":10,"output_tokens":20}}}`,
		`data: [DONE]`,
	)
	if err != nil {
		t.Fatalf("Process: %v", err)
	}
	for _, want := range []string{
		`"type":"message"`, `"content":"Hi"`,
		`"type":"usage"`, `"input_tokens":10`, `"output_tokens":20`,
	} {
		if !strings.Contains(got, want) {
			t.Errorf("output missing %q; got:\n%s", want, got)
		}
	}
}

// TestDecoderCompletesAtTheCompletedEvent pins when Process reports done: a
// content delta is not the end of a stream, and response.completed is.
func TestDecoderCompletesAtTheCompletedEvent(t *testing.T) {
	dec := NewDecoder(nil)
	done, err := dec.Process(`data: {"type":"response.output_text.delta","output_index":0,"delta":"hi"}`)
	if err != nil {
		t.Fatalf("Process: %v", err)
	}
	if done {
		t.Fatal("Process must not report done while the model is still writing")
	}
	done, err = dec.Process(`data: {"type":"response.completed","response":{"usage":{"input_tokens":1,"output_tokens":2}}}`)
	if err != nil {
		t.Fatalf("Process: %v", err)
	}
	if !done {
		t.Fatal("Process must report done at response.completed")
	}
}

// TestDecoderFinalFlushesWithoutDone covers the EOF path: a transport that
// closes without a terminal event. Final() must flush the accumulated message
// and tool call, and must be idempotent.
func TestDecoderFinalFlushesWithoutDone(t *testing.T) {
	var buf bytes.Buffer
	dec := NewDecoder(NewEncoder(&buf))
	for _, line := range []string{
		`data: {"type":"response.output_text.delta","output_index":0,"delta":"yo"}`,
		`data: {"type":"response.output_item.added","output_index":0,"item":{"type":"function_call","call_id":"call_1","name":"shell"}}`,
		`data: {"type":"response.function_call_arguments.delta","output_index":0,"delta":"echo hi"}`,
	} {
		done, err := dec.Process(line)
		if err != nil {
			t.Fatalf("Process: %v", err)
		}
		if done {
			t.Fatalf("Process reported done with no terminal event")
		}
	}
	if !dec.Final() {
		t.Fatal("Final reported not done")
	}
	got := buf.String()
	for _, want := range []string{
		`"type":"message"`, `"content":"yo"`,
		`"type":"tool_call"`, `"arguments":"echo hi"`,
	} {
		if !strings.Contains(got, want) {
			t.Errorf("output missing %q; got:\n%s", want, got)
		}
	}
	// Final is idempotent: a second call must not duplicate events.
	before := buf.String()
	dec.Final()
	if after := buf.String(); after != before {
		t.Errorf("Final duplicated events; got:\n%s\nwant:\n%s", after, before)
	}
}

// TestDecoderSplitsCachedAndUncachedInput pins the explicit cache split: the
// provider reports how many input tokens were served from cache
// (input_tokens_details.cached_tokens). The emitted usage event carries both
// parts, which sum to the total input, plus the unchanged total for wire
// compatibility.
func TestDecoderSplitsCachedAndUncachedInput(t *testing.T) {
	got, err := feed(t,
		`data: {"type":"response.output_text.delta","output_index":0,"delta":"Hi"}`,

		`data: {"type":"response.completed","response":{"usage":{"input_tokens":10,"output_tokens":20,"input_tokens_details":{"cached_tokens":4}}}}`,
		`data: [DONE]`,
	)
	if err != nil {
		t.Fatalf("Process: %v", err)
	}
	for _, want := range []string{
		`"type":"usage"`,
		`"input_tokens":10`,         // total kept for compatibility
		`"output_tokens":20`,        //
		`"cached_input_tokens":4`,   // the explicit split: cache hits
		`"uncached_input_tokens":6`, // and the misses (10 - 4)
	} {
		if !strings.Contains(got, want) {
			t.Errorf("output missing %q; got:\n%s", want, got)
		}
	}
}

// TestDecoderNoCacheSplitDefaultsAllUncached pins the "provider did not report
// a cache split" default: everything is uncached, and no zero-value cache
// fields leak into the JSON.
func TestDecoderNoCacheSplitDefaultsAllUncached(t *testing.T) {
	got, err := feed(t,
		`data: {"type":"response.output_text.delta","output_index":0,"delta":"Hi"}`,

		`data: {"type":"response.completed","response":{"usage":{"input_tokens":5,"output_tokens":3}}}`,
		`data: [DONE]`,
	)
	if err != nil {
		t.Fatalf("Process: %v", err)
	}
	for _, want := range []string{
		`"uncached_input_tokens":5`,
	} {
		if !strings.Contains(got, want) {
			t.Errorf("output missing %q; got:\n%s", want, got)
		}
	}
	if strings.Contains(got, `"cached_input_tokens"`) {
		t.Errorf("zero cache split leaked into JSON (omitempty should drop it); got:\n%s", got)
	}
}

// TestDecoderReadsWholeTextReasoning covers the upstream shape where the chain
// of thought is never streamed as deltas: it arrives whole, inside the
// reasoning item's content array, on output_item.added / output_item.done. A
// Reka endpoint serving z-ai/glm-5.3-flash through OpenRouter was caught doing
// this, and porter committed an empty reply when a turn spent all its tokens
// on reasoning.
func TestDecoderReadsWholeTextReasoning(t *testing.T) {
	got, err := feed(t,
		`data: {"type":"response.output_item.added","output_index":0,"item":{"id":"rs_tmp_x","summary":[],"type":"reasoning","content":[{"text":"thinking here","type":"reasoning_text"}]}}`,
		`data: {"type":"response.output_item.done","output_index":0,"item":{"id":"rs_tmp_x","summary":[],"type":"reasoning","content":[{"text":"thinking here","type":"reasoning_text"}]}}`,
		`data: {"type":"response.completed","response":{"usage":{"input_tokens":14,"output_tokens":131}}}`,
	)
	if err != nil {
		t.Fatalf("Process: %v", err)
	}
	for _, want := range []string{
		`"type":"reasoning_delta"`, `"reasoning":"thinking here"`,
		// The reasoning must reach the assembled message too.
		`"type":"message"`, `"reasoning":"thinking here"`,
		`"type":"usage"`,
	} {
		if !strings.Contains(got, want) {
			t.Errorf("output missing %q; got:\n%s", want, got)
		}
	}
	// added and done both carried the same text; it must not be doubled.
	// Once in the live delta, once in the assembled message.
	if n := strings.Count(got, "thinking here"); n != 2 {
		t.Errorf("reasoning appears %d times, want 2 (delta + assembled); got:\n%s", n, got)
	}
}

// TestDecoderReasoningEchoOnItemIsNotDoubled covers the provider that both
// streams reasoning deltas and echoes the whole text on the completed item:
// the streamed copy wins, the echo is dropped.
func TestDecoderReasoningEchoOnItemIsNotDoubled(t *testing.T) {
	got, err := feed(t,
		`data: {"type":"response.reasoning_summary_text.delta","output_index":0,"delta":"think"}`,
		`data: {"type":"response.output_item.done","output_index":0,"item":{"type":"reasoning","content":[{"text":"think whole","type":"reasoning_text"}]}}`,
		`data: {"type":"response.completed","response":{"usage":{"input_tokens":1,"output_tokens":1}}}`,
	)
	if err != nil {
		t.Fatalf("Process: %v", err)
	}
	if strings.Contains(got, `"whole"`) {
		t.Errorf("the item's echo overwrote or joined the streamed reasoning; got:\n%s", got)
	}
	if !strings.Contains(got, `"reasoning":"think"`) {
		t.Errorf("the streamed reasoning is missing; got:\n%s", got)
	}
}

// TestDecoderReadsWholeTextReplyFromMessageItem covers the dead-end shape from
// the production hunt: an upstream that never streams output_text.delta
// delivers the whole reply inside the message item. Before this fallback, the
// decoder ignored that item, porter committed an empty reply, and the turn
// ended silently — the user had to type "continue" over and over.
func TestDecoderReadsWholeTextReplyFromMessageItem(t *testing.T) {
	got, err := feed(t,
		`data: {"type":"response.output_item.added","output_index":0,"item":{"id":"rs_1","type":"reasoning","content":[{"text":"plan","type":"reasoning_text"}]}}`,
		`data: {"type":"response.output_item.added","output_index":1,"item":{"id":"msg_1","content":[{"text":"Let me check your battery.","type":"output_text"}],"role":"assistant","type":"message"}}`,
		`data: {"type":"response.output_item.done","output_index":1,"item":{"id":"msg_1","content":[{"text":"Let me check your battery.","type":"output_text"}],"role":"assistant","type":"message"}}`,
		`data: {"type":"response.completed","response":{"usage":{"input_tokens":10,"output_tokens":30}}}`,
	)
	if err != nil {
		t.Fatalf("Process: %v", err)
	}
	for _, want := range []string{
		`"type":"message_delta"`, `"delta":"Let me check your battery."`,
		`"type":"message"`, `"content":"Let me check your battery."`,
		`"reasoning":"plan"`,
		`"type":"usage"`,
	} {
		if !strings.Contains(got, want) {
			t.Errorf("output missing %q; got:\n%s", want, got)
		}
	}
	// added and done both carried the reply; it must not be doubled.
	if n := strings.Count(got, "Let me check your battery."); n != 2 {
		t.Errorf("reply appears %d times, want 2 (delta + assembled); got:\n%s", n, got)
	}
}

// TestDecoderStreamedTextWinsOverItemEcho keeps the double-write from the other
// side: a provider that streams output_text.delta AND carries the whole reply
// in the item must commit it once.
func TestDecoderStreamedTextWinsOverItemEcho(t *testing.T) {
	got, err := feed(t,
		`data: {"type":"response.output_text.delta","output_index":0,"delta":"streamed"}`,
		`data: {"type":"response.output_item.added","output_index":0,"item":{"id":"msg_1","content":[{"text":"whole","type":"output_text"}],"role":"assistant","type":"message"}}`,
		`data: {"type":"response.output_item.done","output_index":0,"item":{"id":"msg_1","content":[{"text":"whole","type":"output_text"}],"role":"assistant","type":"message"}}`,
		`data: {"type":"response.completed","response":{"usage":{"input_tokens":1,"output_tokens":1}}}`,
	)
	if err != nil {
		t.Fatalf("Process: %v", err)
	}
	if strings.Contains(got, `"whole"`) {
		t.Errorf("the item echo joined the streamed reply; got:\n%s", got)
	}
	if !strings.Contains(got, `"content":"streamed"`) {
		t.Errorf("the streamed reply is missing; got:\n%s", got)
	}
}

// TestDecoderCollectsCallFromFinalOutput covers the shape that made porter stop
// the chat mid-task: an upstream (LiteLLM's chat-completions bridge) announces
// no output_item events for a function call; the call rides only in the
// terminal response's output list. The decoder must take it from there, or
// porter sees a plain reply, ends the turn, and the model's follow-up call is
// never run.
func TestDecoderCollectsCallFromFinalOutput(t *testing.T) {
	got, err := feed(t,
		`data: {"type":"response.output_item.added","output_index":0,"item":{"id":"rs_1","type":"reasoning","content":[{"text":"recall the rest","type":"reasoning_text"}]}}`,
		`data: {"type":"response.output_item.added","output_index":1,"item":{"id":"msg_1","content":[{"text":"Reading the rest of it now.","type":"output_text"}],"role":"assistant","type":"message"}}`,
		`data: {"type":"response.completed","response":{"output":[{"id":"rs_1","type":"reasoning"},{"id":"msg_1","type":"message"},{"id":"fc_1","call_id":"call_x9","type":"function_call","name":"recall_tool_output","arguments":"{\"call_id\":\"chatcmpl-tool-847\",\"offset\":0}"}],"usage":{"input_tokens":10,"output_tokens":77}}}`,
	)
	if err != nil {
		t.Fatalf("Process: %v", err)
	}
	for _, want := range []string{
		`"type":"tool_call"`, `"tool_call_id":"call_x9"`, `"name":"recall_tool_output"`,
		`"arguments":"{\"call_id\":\"chatcmpl-tool-847\",\"offset\":0}"`,
		`"content":"Reading the rest of it now."`,
	} {
		if !strings.Contains(got, want) {
			t.Errorf("output missing %q; got:\n%s", want, got)
		}
	}
}

// TestDecoderKeepsAnnouncedCallOverFinalEcho keeps a call the stream announced
// from being clobbered or doubled by the terminal payload's copy.
func TestDecoderKeepsAnnouncedCallOverFinalEcho(t *testing.T) {
	got, err := feed(t,
		`data: {"type":"response.output_item.added","output_index":0,"item":{"id":"fc_1","call_id":"call_live","type":"function_call","name":"shell"}}`,
		`data: {"type":"response.function_call_arguments.delta","output_index":0,"delta":"{\"command\":\"ls\"}"}`,
		`data: {"type":"response.completed","response":{"output":[{"id":"fc_1","call_id":"call_final","type":"function_call","name":"OTHER","arguments":"{\"command\":\"rm\"}"}],"usage":{"input_tokens":1,"output_tokens":1}}}`,
	)
	if err != nil {
		t.Fatalf("Process: %v", err)
	}
	for _, want := range []string{
		`"tool_call_id":"call_live"`, `"name":"shell"`, `"arguments":"{\"command\":\"ls\"}"`,
	} {
		if !strings.Contains(got, want) {
			t.Errorf("output missing %q; got:\n%s", want, got)
		}
	}
	if strings.Contains(got, `"call_final"`) || strings.Contains(got, `"rm"`) {
		t.Errorf("the final payload overwrote the announced call; got:\n%s", got)
	}
}

// TestDecoderHandlesMixedItemStream is the whole shape from the production hunt,
// in the order that stream used: a reasoning item, a message item carrying the
// reply whole, a function call announced by output_item events, and a terminal
// response that repeats all three in its output list. Every part must survive:
// the reasoning, the reply text, and the call — and the call must not be
// dropped, because a dropped call ends the turn while the model is mid-task.
func TestDecoderHandlesMixedItemStream(t *testing.T) {
	got, err := feed(t,
		`data: {"type":"response.created","response":{"id":"resp_m","status":"in_progress"}}`,
		`data: {"type":"response.output_item.added","output_index":0,"item":{"id":"rs_1","type":"reasoning","content":[{"text":"recall the rest","type":"reasoning_text"}]}}`,
		`data: {"type":"response.output_item.added","output_index":1,"item":{"id":"msg_1","content":[{"text":"Reading the rest of it now.","type":"output_text"}],"role":"assistant","type":"message"}}`,
		`data: {"type":"response.output_item.added","output_index":2,"item":{"id":"fc_1","call_id":"call_x1","type":"function_call","name":"recall_tool_output","arguments":""}}`,
		`data: {"type":"response.function_call_arguments.delta","output_index":2,"item_id":"fc_1","delta":"{\"call_id\":\"call_1\",\"offset\":0}"}`,
		`data: {"type":"response.function_call_arguments.done","output_index":2,"item_id":"fc_1","arguments":"{\"call_id\":\"call_1\",\"offset\":0}"}`,
		`data: {"type":"response.output_item.done","output_index":1,"item":{"id":"msg_1","content":[{"text":"Reading the rest of it now.","type":"output_text"}],"role":"assistant","type":"message"}}`,
		`data: {"type":"response.output_item.done","output_index":2,"item":{"id":"fc_1","call_id":"call_x1","type":"function_call","name":"recall_tool_output","arguments":"{\"call_id\":\"call_1\",\"offset\":0}"}}`,
		`data: {"type":"response.completed","response":{"output":[{"id":"rs_1","type":"reasoning"},{"id":"msg_1","type":"message"},{"id":"fc_1","call_id":"call_x1","type":"function_call","name":"recall_tool_output","arguments":"{\"call_id\":\"call_1\",\"offset\":0}"}],"usage":{"input_tokens":10,"output_tokens":77}}}`,
	)
	if err != nil {
		t.Fatalf("Process: %v", err)
	}
	for _, want := range []string{
		`"reasoning":"recall the rest"`,
		`"content":"Reading the rest of it now."`,
		`"type":"tool_call"`, `"tool_call_id":"call_x1"`, `"name":"recall_tool_output"`,
		`"arguments":"{\"call_id\":\"call_1\",\"offset\":0}"`,
		`"type":"usage"`, `"output_tokens":77`,
	} {
		if !strings.Contains(got, want) {
			t.Errorf("output missing %q; got:\n%s", want, got)
		}
	}
	// The reply text is committed once, and the call is emitted once.
	if n := strings.Count(got, `"content":"Reading the rest of it now."`); n != 1 {
		t.Errorf("reply committed %d times, want 1; got:\n%s", n, got)
	}
	if n := strings.Count(got, `"type":"tool_call"`); n != 1 {
		t.Errorf("tool call emitted %d times, want 1; got:\n%s", n, got)
	}
}
