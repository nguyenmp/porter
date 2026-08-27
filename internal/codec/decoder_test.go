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
		`data: {"choices":[{"delta":{"content":"Hel"},"finish_reason":null}]}`,
		`data: {"choices":[{"delta":{"content":"lo"},"finish_reason":null}]}`,
		`data: {"choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":20}}`,
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
		`data: {"choices":[{"delta":{"content":"hi"}}]}`,
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
		// DeepSeek-style: reasoning_content interleaved with content.
		`data: {"choices":[{"delta":{"reasoning_content":"think","content":"Answer"},"finish_reason":null}]}`,
		`data: {"choices":[{"delta":{"reasoning_content":"ing"},"finish_reason":null}]}`,
		// OpenAI-style: reasoning alongside a content delta.
		`data: {"choices":[{"delta":{"reasoning":".","content":" is 4"},"finish_reason":"stop"}]}`,
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
		`data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"shell","arguments":""}}]},"finish_reason":null}]}`,
		`data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"echo "}}]},"finish_reason":null}]}`,
		`data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"hi"}}]},"finish_reason":"tool_calls"}]}`,
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
		`data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"shell","arguments":""}}]},"finish_reason":null}]}`,
		`data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"echo "}}]},"finish_reason":null}]}`,
		`data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"hi"}}]},"finish_reason":"tool_calls"}]}`,
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

func TestDecoderToolCallDeltaSkipsEmptyFragment(t *testing.T) {
	got, err := feed(t,
		`data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"shell","arguments":"hi"}}]},"finish_reason":null}]}`,
		// An echo of the same index with no new fields must not emit a delta.
		`data: {"choices":[{"delta":{"tool_calls":[{"index":0}]},"finish_reason":"tool_calls"}]}`,
		`data: [DONE]`,
	)
	if err != nil {
		t.Fatalf("Process: %v", err)
	}
	if !strings.Contains(got, `"type":"tool_call_delta"`) {
		t.Fatalf("expected a live delta; got:\n%s", got)
	}
	if n, want := strings.Count(got, `"type":"tool_call_delta"`), 1; n != want {
		t.Errorf("delta count = %d, want %d (empty fragment must not emit); got:\n%s", n, want, got)
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

// TestDecoderConsumesTrailingUsageChunk reproduces the real OpenAI/OpenAI-
// compatible streaming shape: usage arrives in a separate chunk AFTER the
// finish_reason chunk, with empty choices. The decoder must keep reading past
// finish_reason (not return done there) so the usage is captured before [DONE].
func TestDecoderConsumesTrailingUsageChunk(t *testing.T) {
	got, err := feed(t,
		`data: {"choices":[{"delta":{"content":"Hi"},"finish_reason":null}]}`,
		`data: {"choices":[{"delta":{},"finish_reason":"stop"}]}`,
		`data: {"choices":[],"usage":{"prompt_tokens":10,"completion_tokens":20}}`,
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

// TestDecoderFinishReasonDoesNotStopStream pins the contract that Process only
// reports done at [DONE], not at finish_reason: a usage-only chunk can still
// follow a finish_reason chunk.
func TestDecoderFinishReasonDoesNotStopStream(t *testing.T) {
	dec := NewDecoder(nil)
	done, err := dec.Process(`data: {"choices":[{"delta":{},"finish_reason":"stop"}]}`)
	if err != nil {
		t.Fatalf("Process: %v", err)
	}
	if done {
		t.Fatal("Process must not report done at finish_reason; usage may follow in a separate chunk")
	}
	done, err = dec.Process(`data: {"choices":[],"usage":{"prompt_tokens":1,"completion_tokens":2}}`)
	if err != nil {
		t.Fatalf("Process: %v", err)
	}
	if done {
		t.Fatal("Process must not report done on a usage-only chunk; [DONE] is still to come")
	}
	done, err = dec.Process(`data: [DONE]`)
	if err != nil {
		t.Fatalf("Process: %v", err)
	}
	if !done {
		t.Fatal("Process must report done at [DONE]")
	}
}

// TestDecoderFinalFlushesWithoutDone covers the EOF path: a stream that ends
// (EOF/close) without a [DONE] marker. Final() must flush the terminal events,
// including usage consumed in a trailing chunk, and must be idempotent.
func TestDecoderFinalFlushesWithoutDone(t *testing.T) {
	var buf bytes.Buffer
	dec := NewDecoder(NewEncoder(&buf))
	for _, line := range []string{
		`data: {"choices":[{"delta":{"content":"yo"},"finish_reason":null}]}`,
		`data: {"choices":[{"delta":{},"finish_reason":"stop"}]}`,
		`data: {"choices":[],"usage":{"prompt_tokens":3,"completion_tokens":4}}`,
	} {
		done, err := dec.Process(line)
		if err != nil {
			t.Fatalf("Process: %v", err)
		}
		if done {
			t.Fatalf("Process reported done before [DONE]")
		}
	}
	if !dec.Final() {
		t.Fatal("Final reported not done")
	}
	got := buf.String()
	for _, want := range []string{
		`"type":"message"`, `"content":"yo"`,
		`"type":"usage"`, `"input_tokens":3`, `"output_tokens":4`,
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
// provider reports how many prompt tokens were served from cache
// (prompt_tokens_details.cached_tokens, the OpenAI-compatible shape LiteLLM and
// OpenRouter normalize to). The emitted usage event carries both parts, which
// sum to the total input, plus the unchanged total for wire compatibility.
func TestDecoderSplitsCachedAndUncachedInput(t *testing.T) {
	got, err := feed(t,
		`data: {"choices":[{"delta":{"content":"Hi"},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":20,"prompt_tokens_details":{"cached_tokens":4}}}`,
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

// TestDecoderSplitsCachedInputDeepSeek covers the native DeepSeek shape, where
// the cache-hit count is a top-level prompt_cache_hit_tokens field rather than
// nested under prompt_tokens_details. This is what the current .env model
// (deepseek/deepseek-v4-flash) sends when it bypasses an OpenAI-normalizing
// proxy.
func TestDecoderSplitsCachedInputDeepSeek(t *testing.T) {
	got, err := feed(t,
		`data: {"choices":[{"delta":{"content":"Hi"},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":20,"prompt_cache_hit_tokens":7}}`,
		`data: [DONE]`,
	)
	if err != nil {
		t.Fatalf("Process: %v", err)
	}
	for _, want := range []string{
		`"cached_input_tokens":7`,
		`"uncached_input_tokens":3`,
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
		`data: {"choices":[{"delta":{"content":"Hi"},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":3}}`,
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
