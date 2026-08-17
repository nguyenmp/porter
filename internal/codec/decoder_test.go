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
