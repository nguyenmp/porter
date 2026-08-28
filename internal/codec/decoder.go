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
	Content          string      `json:"content"`
	Reasoning        string      `json:"reasoning"`
	ReasoningContent string      `json:"reasoning_content"`
	ToolCalls        []deltaCall `json:"tool_calls"`
}

// deltaCall is the per-tool-call streaming fragment. A single logical call is
// split across many deltas keyed by Index; Arguments arrives in pieces.
type deltaCall struct {
	Index    int               `json:"index"`
	ID       string            `json:"id"`
	Type     string            `json:"type"`
	Function deltaCallFunction `json:"function"`
}

type deltaCallFunction struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

// ToolCall is the completed form of one tool call, emitted once its arguments
// have fully streamed. Index is the stream index it was assembled from, so
// consumers can match the completed call to the live tool_call_delta blocks.
type ToolCall struct {
	ID        string `json:"id"`
	Type      string `json:"type"`
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
	Index     int    `json:"index"`
}

// chunk is the shape of a single SSE `data:` payload from a streaming Chat
// Completions response.
type chunk struct {
	Choices []struct {
		Index        int     `json:"index"`
		Delta        delta   `json:"delta"`
		FinishReason *string `json:"finish_reason"`
	}
	Usage *usage `json:"usage"`
}

// usage is the token-accounting object a provider sends on the final streamed
// chunk. The cache split travels in two shapes, depending on who normalizes it:
// OpenAI-compatible proxies (LiteLLM, OpenRouter) emit
// prompt_tokens_details.cached_tokens, while DeepSeek's native API uses the
// top-level prompt_cache_hit_tokens. cachedInput() prefers the normalized form
// and falls back to the native one, so porter tolerates both.
type usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	// PromptCacheHitTokens is DeepSeek's native cache-hit count (a top-level
	// field, not nested under prompt_tokens_details).
	PromptCacheHitTokens int `json:"prompt_cache_hit_tokens"`
	// PromptTokensDetails is the OpenAI-compatible cache breakdown.
	PromptTokensDetails *struct {
		CachedTokens int `json:"cached_tokens"`
	} `json:"prompt_tokens_details"`
}

// cachedInput returns how many prompt tokens were served from cache, or 0 when
// the provider did not report a split.
func (u *usage) cachedInput() int {
	if u == nil {
		return 0
	}
	if u.PromptTokensDetails != nil && u.PromptTokensDetails.CachedTokens > 0 {
		return u.PromptTokensDetails.CachedTokens
	}
	return u.PromptCacheHitTokens
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
	cachedIn  int

	// finalized guards the one-time emission of the terminal events
	// (TypeMessage, TypeToolCall, TypeUsage). A stream may end by a [DONE]
	// marker or by EOF; both paths must emit them exactly once, and a
	// finish_reason must not emit them early (the usage chunk is still to
	// come).
	finalized bool

	// finished is set when any chunk carried a terminal finish_reason, so a
	// caller can tell a stream that reached its natural end (even without a
	// [DONE] marker) from one cut off mid-flight. It is false while the model
	// is still generating.
	finished bool

	// toolCalls accumulates streamed tool-call fragments keyed by their stream
	// index, so a call split across deltas reassembles into one.
	toolCalls map[int]*ToolCall

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
// once the stream is complete — that is, at the `[DONE]` marker — at which
// point callers should stop feeding lines. A terminal finish_reason does NOT
// end the stream: OpenAI-compatible APIs send the usage totals in a separate
// chunk after the finish_reason chunk (with empty choices), so Process keeps
// consuming past finish_reason to pick that usage up. Callers that reach EOF
// without a [DONE] marker should call Final() to flush the terminal events.
func (d *Decoder) Process(line string) (bool, error) {
	payload, done, err := payloadOf(line)
	if err != nil {
		return true, fmt.Errorf("parse sse line: %w", err)
	}
	if done {
		return d.finalize(), nil
	}

	var c chunk
	if err := json.Unmarshal(payload, &c); err != nil {
		return true, fmt.Errorf("decode chunk: %w", err)
	}

	if c.Usage != nil {
		d.inTokens = c.Usage.PromptTokens
		d.outTokens = c.Usage.CompletionTokens
		d.cachedIn = c.Usage.cachedInput()
	}

	for _, choice := range c.Choices {
		if choice.FinishReason != nil {
			d.finished = true
		}
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
		for _, call := range choice.Delta.ToolCalls {
			if err := d.accumulateToolCall(call); err != nil {
				return false, err
			}
		}
	}

	return false, nil
}

// Final flushes the accumulated message, tool-call, and usage events for a
// stream that ended without a [DONE] marker (EOF or transport close). It is
// safe to call after [DONE] too: finalize runs exactly once, so callers can
// always invoke Final() after their read loop without double-emitting.
func (d *Decoder) Final() bool {
	return d.finalize()
}

// Finished reports whether the stream reached a terminal finish_reason (the
// provider finished generating), even if no [DONE] marker followed. It is
// false while the model is still generating. The agent uses it to distinguish
// a user stop mid-stream from a stream that completed without a [DONE].
func (d *Decoder) Finished() bool { return d.finished }

// finalize writes the accumulated message, tool-call, and usage events exactly
// once and reports done.
func (d *Decoder) finalize() bool {
	if d.finalized {
		return true
	}
	d.finalized = true
	_ = d.emit(Event{Type: TypeMessage, Role: "assistant", Content: d.full.String(), Reasoning: d.reasoning.String()})
	for _, call := range d.completedToolCalls() {
		_ = d.emit(Event{Type: TypeToolCall, Index: call.Index, ToolCallID: call.ID, Name: call.Name, Arguments: call.Arguments})
	}
	if d.inTokens > 0 || d.outTokens > 0 {
		cached := d.cachedIn
		uncached := d.inTokens - cached
		if uncached < 0 {
			uncached = 0
		}
		_ = d.emit(Event{Type: TypeUsage, InputTokens: d.inTokens, OutputTokens: d.outTokens, CachedInputTokens: cached, UncachedInputTokens: uncached})
	}
	return true
}

// accumulateToolCall folds one streamed tool-call fragment into the running
// set, creating a slot on first sight of an index and appending sliced
// arguments on later deltas. It also emits a live tool_call_delta so consumers
// can render the call as it forms, skipping fragments that carry nothing new
// (some providers echo an index with no fields) so they don't cause spurious
// re-renders.
func (d *Decoder) accumulateToolCall(c deltaCall) error {
	if d.toolCalls == nil {
		d.toolCalls = make(map[int]*ToolCall)
	}
	cur, ok := d.toolCalls[c.Index]
	if !ok {
		cur = &ToolCall{Type: c.Type, Index: c.Index}
		d.toolCalls[c.Index] = cur
	}
	if c.ID != "" {
		cur.ID = c.ID
	}
	if c.Function.Name != "" {
		cur.Name = c.Function.Name
	}
	cur.Arguments += c.Function.Arguments

	if c.ID != "" || c.Function.Name != "" || c.Function.Arguments != "" {
		if err := d.emit(Event{
			Type:       TypeToolCallDelta,
			Index:      c.Index,
			ToolCallID: c.ID,
			Name:       c.Function.Name,
			Arguments:  c.Function.Arguments,
		}); err != nil {
			return err
		}
	}
	return nil
}

// completedToolCalls returns the accumulated tool calls in index order.
func (d *Decoder) completedToolCalls() []ToolCall {
	out := make([]ToolCall, 0, len(d.toolCalls))
	for i := 0; i < len(d.toolCalls); i++ {
		if c, ok := d.toolCalls[i]; ok {
			out = append(out, *c)
		}
	}
	return out
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
