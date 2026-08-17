package codec

import (
	"encoding/json"
	"errors"
	"fmt"
	"strings"
)

var errDone = errors.New("stream terminated")

// delta is the per-choice streaming payload field. Providers name the reasoning
// stream differently: OpenAI sends `reasoning`, DeepSeek/Llama send
// `reasoning_content`. Capture both so the codec is agnostic.
type delta struct {
	Content          string `json:"content"`
	Reasoning        string `json:"reasoning"`
	ReasoningContent string `json:"reasoning_content"`
}

// chunk is the shape of a single SSE `data:` payload from a streaming Chat
// Completions response.
type chunk struct {
	Choices []struct {
		Index        int     `json:"index"`
		Delta        delta   `json:"delta"`
		FinishReason *string `json:"finish_reason"`
	}
	Usage *struct {
		PromptTokens     int `json:"prompt_tokens"`
		CompletionTokens int `json:"completion_tokens"`
	} `json:"usage"`
}

// Decoder accumulates a stream of raw SSE `data:` lines into structured Events,
// tracking the running message content, reasoning, and token usage. Reasoning
// is accumulated separately from content and emitted on its own, so it can be
// surfaced for transparency without ever being glued into resubmitted content.
type Decoder struct {
	enc       *Encoder
	full      strings.Builder
	reasoning strings.Builder
	inTokens  int
	outTokens int

	// OnEvent, when set, is invoked for every Event the decoder produces, in
	// addition to writing it to enc. It lets consumers render or react to the
	// stream without re-parsing.
	OnEvent func(Event)
}

// NewDecoder builds a Decoder that emits events onto enc.
func NewDecoder(enc *Encoder) *Decoder {
	return &Decoder{enc: enc}
}

// emit writes an event to enc (if any) and notifies OnEvent (if set).
func (d *Decoder) emit(ev Event) error {
	if d.enc != nil {
		if err := d.enc.Write(ev); err != nil {
			return err
		}
	}
	if d.OnEvent != nil {
		d.OnEvent(ev)
	}
	return nil
}

// Process consumes one raw SSE `data:` line (with the marker still attached, as
// produced by llm.SSELines) and emits any Events it implies. It returns true
// once the stream is complete (`[DONE]` or a terminal finish_reason), at which
// point callers should stop feeding lines.
func (d *Decoder) Process(line string) (bool, error) {
	payload, done, err := payloadOf(line)
	if err != nil {
		return true, fmt.Errorf("parse sse line: %w", err)
	}
	if done {
		return d.emitFinal(), nil
	}

	var c chunk
	if err := json.Unmarshal(payload, &c); err != nil {
		return true, fmt.Errorf("decode chunk: %w", err)
	}

	if c.Usage != nil {
		d.inTokens = c.Usage.PromptTokens
		d.outTokens = c.Usage.CompletionTokens
	}

	finished := false
	for _, choice := range c.Choices {
		if c := choice.Delta.Content; c != "" {
			d.full.WriteString(c)
			if err := d.emit(Event{Type: TypeMessageDelta, Role: "assistant", Delta: c}); err != nil {
				return false, err
			}
		}
		if r := reasoningOf(choice.Delta); r != "" {
			d.reasoning.WriteString(r)
			if err := d.emit(Event{Type: TypeReasoningDelta, Reasoning: r}); err != nil {
				return false, err
			}
		}
		if choice.FinishReason != nil {
			finished = true
		}
	}

	if finished {
		return d.emitFinal(), nil
	}
	return false, nil
}

// emitFinal writes the accumulated message and usage events and reports done.
func (d *Decoder) emitFinal() bool {
	_ = d.emit(Event{Type: TypeMessage, Role: "assistant", Content: d.full.String(), Reasoning: d.reasoning.String()})
	if d.inTokens > 0 || d.outTokens > 0 {
		_ = d.emit(Event{Type: TypeUsage, InputTokens: d.inTokens, OutputTokens: d.outTokens})
	}
	return true
}

// reasoningOf returns the reasoning text a provider emitted in this delta,
// preferring `reasoning` (OpenAI) and falling back to `reasoning_content`
// (DeepSeek/Llama).
func reasoningOf(d delta) string {
	if d.Reasoning != "" {
		return d.Reasoning
	}
	return d.ReasoningContent
}

// payloadOf strips the `data:` SSE marker and following space from a line,
// returning the JSON payload. The bool reports a `[DONE]` marker.
func payloadOf(line string) ([]byte, bool, error) {
	trimmed := strings.TrimSpace(line)
	if !strings.HasPrefix(trimmed, "data:") {
		return nil, false, fmt.Errorf("not a data line: %q", line)
	}
	payload := strings.TrimSpace(strings.TrimPrefix(trimmed, "data:"))
	if payload == "[DONE]" {
		return nil, true, nil
	}
	return []byte(payload), false, nil
}
