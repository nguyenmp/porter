package codec

import (
	"encoding/json"
	"errors"
	"fmt"
	"strings"
)

var errDone = errors.New("stream terminated")

// streamEvent is one SSE `data:` payload from a streaming Responses reply. The
// event's `type` says what happened; the other fields are set only by the event
// types that carry them, so one flat struct is enough and an event porter does
// not know decodes harmlessly.
//
// The events the decoder reads:
//
//	response.output_text.delta              prose, in pieces (delta)
//	response.reasoning_*_text.delta         chain of thought, in pieces (delta)
//	response.output_item.added              an item starts (item)
//	response.function_call_arguments.delta  pieces of a tool call's arguments
//	response.function_call_arguments.done   the completed arguments
//	response.output_item.done               an item is complete (item)
//	response.completed / .incomplete        the end, carrying usage
//	response.failed / error                 the provider gave up
type streamEvent struct {
	Type        string        `json:"type"`
	Delta       string        `json:"delta"`
	Text        string        `json:"text"`
	Arguments   string        `json:"arguments"`
	OutputIndex int           `json:"output_index"`
	Message     string        `json:"message"`
	Item        *outputItem   `json:"item"`
	Response    *responseBody `json:"response"`
	Error       *apiError     `json:"error"`
}

// outputItem is the item an output_item.* event carries. Only a function call
// has anything porter needs; a message or reasoning item is empty here, because
// its prose arrives as delta events instead.
type outputItem struct {
	Type      string `json:"type"`
	ID        string `json:"id"`
	CallID    string `json:"call_id"`
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

// responseBody is the response object the terminal events carry. porter reads
// its token totals, and its error when the response failed.
type responseBody struct {
	Usage *usage    `json:"usage"`
	Error *apiError `json:"error"`
}

// apiError is a provider-reported failure.
type apiError struct {
	Code    string `json:"code"`
	Message string `json:"message"`
}

// usage is the token-accounting object the completed response carries. The
// Responses API names the counts input_tokens/output_tokens, where Chat
// Completions said prompt_tokens/completion_tokens, and it nests the cache
// split under input_tokens_details.
type usage struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
	// InputTokensDetails is the cache breakdown: how much of the input the
	// provider served from its own cache.
	InputTokensDetails *struct {
		CachedTokens int `json:"cached_tokens"`
	} `json:"input_tokens_details"`
}

// cachedInput returns how many input tokens were served from cache, or 0 when
// the provider did not report a split.
func (u *usage) cachedInput() int {
	if u == nil || u.InputTokensDetails == nil {
		return 0
	}
	return u.InputTokensDetails.CachedTokens
}

// ToolCall is the completed form of one tool call, emitted once its arguments
// have fully streamed. Index is the output index it was assembled from, so
// consumers can match the completed call to the live tool_call_delta blocks.
type ToolCall struct {
	ID        string `json:"id"`
	Type      string `json:"type"`
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
	Index     int    `json:"index"`
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
	// (TypeMessage, TypeToolCall, TypeUsage). A stream can end three ways: the
	// response.completed event, a gateway, or EOF when the
	// transport closes with neither. Every path must emit them exactly once.
	finalized bool

	// finished is set when the stream reached a terminal event, so a caller can
	// tell a stream that ended on its own from one cut off mid-flight. It is
	// false while the model is still generating.
	finished bool

	// toolCalls accumulates a tool call keyed by its output index, so the pieces
	// that arrive across many events reassemble into one.
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
// once the stream is complete, at which point callers should stop feeding
// lines: on the response.completed event, on a failure, or on a [DONE] marker
// from a gateway that still sends one. Callers that reach EOF without any of
// those should call Final() to flush the terminal events.
func (d *Decoder) Process(line string) (bool, error) {
	payload, done, err := payloadOf(line)
	if err != nil {
		return true, fmt.Errorf("parse sse line: %w", err)
	}
	if done {
		return d.finalize(), nil
	}

	var ev streamEvent
	if err := json.Unmarshal(payload, &ev); err != nil {
		return true, fmt.Errorf("decode event: %w", err)
	}

	switch {
	case ev.Type == "response.output_text.delta":
		return false, d.emitText(ev.Delta)
	case isReasoningDelta(ev.Type):
		return false, d.emitReasoning(ev.Delta)
	case ev.Type == "response.output_text.done":
		// A gateway that sent no text deltas may still hand the whole message
		// over here instead.
		if d.full.Len() == 0 && ev.Text != "" {
			d.full.WriteString(ev.Text)
		}
	case ev.Type == "response.output_item.added":
		return false, d.startItem(ev)
	case ev.Type == "response.function_call_arguments.delta":
		return false, d.addArguments(ev.OutputIndex, ev.Delta)
	case ev.Type == "response.function_call_arguments.done":
		return false, d.doneArguments(ev.OutputIndex, ev.Arguments)
	case ev.Type == "response.output_item.done":
		return false, d.finishItem(ev)
	case ev.Type == "response.completed", ev.Type == "response.incomplete":
		d.finished = true
		d.captureUsage(ev)
		return d.finalize(), nil
	case ev.Type == "response.failed", ev.Type == "error":
		d.finished = true
		return true, providerError(ev)
	}
	return false, nil
}

// Final flushes the accumulated message, tool-call, and usage events for a
// stream that ended without a terminal event (EOF or transport close). It is
// safe to call after any ending: finalize runs exactly once, so callers can
// always invoke Final() after their read loop without double-emitting.
func (d *Decoder) Final() bool {
	return d.finalize()
}

// Finished reports whether the stream reached a terminal event (the provider
// finished generating, or gave up), even if the transport then closed without a
// marker. It is false while the model is still generating. The agent uses it to
// distinguish a user stop mid-stream from a stream that completed.
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

// captureUsage records the token totals the completed response reported.
func (d *Decoder) captureUsage(ev streamEvent) {
	if ev.Response == nil || ev.Response.Usage == nil {
		return
	}
	d.inTokens = ev.Response.Usage.InputTokens
	d.outTokens = ev.Response.Usage.OutputTokens
	d.cachedIn = ev.Response.Usage.cachedInput()
}

// emitText accumulates a piece of the model's prose and reports it as a live
// delta.
func (d *Decoder) emitText(s string) error {
	if s == "" {
		return nil
	}
	d.full.WriteString(s)
	return d.emit(Event{Type: TypeMessageDelta, Role: "assistant", Delta: s})
}

// emitReasoning accumulates a piece of the model's chain of thought and reports
// it as a live delta, kept apart from the prose.
func (d *Decoder) emitReasoning(s string) error {
	if s == "" {
		return nil
	}
	d.reasoning.WriteString(s)
	return d.emit(Event{Type: TypeReasoningDelta, Reasoning: s})
}

// slot returns the tool call being assembled at an output index, creating it on
// first sight.
func (d *Decoder) slot(index int) *ToolCall {
	if d.toolCalls == nil {
		d.toolCalls = make(map[int]*ToolCall)
	}
	cur, ok := d.toolCalls[index]
	if !ok {
		cur = &ToolCall{Type: "function", Index: index}
		d.toolCalls[index] = cur
	}
	return cur
}

// startItem records an item that just began, when it is a tool call. The call's
// id and name arrive with the item, before any of its arguments do.
//
// The item's own arguments are shown live here but deliberately not stored:
// every provider that reports them with the item also streams them as separate
// deltas, and storing both would double the argument string. finishItem takes
// the item's copy only when no delta ever carried one.
func (d *Decoder) startItem(ev streamEvent) error {
	if ev.Item == nil || ev.Item.Type != "function_call" {
		return nil
	}
	cur := d.slot(ev.OutputIndex)
	if cur.ID == "" {
		cur.ID = firstNonEmpty(ev.Item.CallID, ev.Item.ID)
	}
	if cur.Name == "" {
		cur.Name = ev.Item.Name
	}
	return d.liveDelta(ev.OutputIndex, ev.Item.CallID, ev.Item.Name, ev.Item.Arguments)
}

// addArguments appends a slice of a tool call's arguments and reports it live.
func (d *Decoder) addArguments(index int, args string) error {
	d.slot(index).Arguments += args
	return d.liveDelta(index, "", "", args)
}

// doneArguments records the completed arguments. Some providers send the whole
// argument string here without ever sending deltas; when that happens the call
// is reported live now, so a consumer still renders it as it forms.
func (d *Decoder) doneArguments(index int, args string) error {
	cur := d.slot(index)
	if cur.Arguments != "" || args == "" {
		return nil
	}
	cur.Arguments = args
	return d.liveDelta(index, "", "", args)
}

// finishItem takes the authoritative fields from a completed item, in case the
// deltas never delivered them.
func (d *Decoder) finishItem(ev streamEvent) error {
	if ev.Item == nil || ev.Item.Type != "function_call" {
		return nil
	}
	cur := d.slot(ev.OutputIndex)
	if cur.ID == "" {
		cur.ID = firstNonEmpty(ev.Item.CallID, ev.Item.ID)
	}
	if cur.Name == "" {
		cur.Name = ev.Item.Name
	}
	if cur.Arguments == "" && ev.Item.Arguments != "" {
		cur.Arguments = ev.Item.Arguments
		return d.liveDelta(ev.OutputIndex, "", "", ev.Item.Arguments)
	}
	return nil
}

// liveDelta emits a tool_call_delta when a fragment carries something new.
// Providers sometimes echo an index with no fields; those must not re-render.
func (d *Decoder) liveDelta(index int, id, name, args string) error {
	if id == "" && name == "" && args == "" {
		return nil
	}
	return d.emit(Event{
		Type:       TypeToolCallDelta,
		Index:      index,
		ToolCallID: id,
		Name:       name,
		Arguments:  args,
	})
}

// completedToolCalls returns the accumulated tool calls in output order.
func (d *Decoder) completedToolCalls() []ToolCall {
	out := make([]ToolCall, 0, len(d.toolCalls))
	for i := 0; i < len(d.toolCalls); i++ {
		if c, ok := d.toolCalls[i]; ok {
			out = append(out, *c)
		}
	}
	return out
}

// isReasoningDelta reports whether an event carries a piece of the model's
// chain of thought. The name varies by model and gateway — the summary stream
// is response.reasoning_summary_text.delta and the raw stream is
// response.reasoning_text.delta — so any reasoning event ending in ".delta" is
// taken as one.
func isReasoningDelta(name string) bool {
	return strings.Contains(name, "reasoning") && strings.HasSuffix(name, ".delta")
}

// providerError turns a failure event into an error. The message may sit on the
// event itself or inside the response it carries.
func providerError(ev streamEvent) error {
	msg := ev.Message
	if msg == "" && ev.Error != nil {
		msg = ev.Error.Message
	}
	if msg == "" && ev.Response != nil && ev.Response.Error != nil {
		msg = ev.Response.Error.Message
	}
	if msg == "" {
		msg = "the provider reported a failure"
	}
	return fmt.Errorf("provider error: %s", msg)
}

// firstNonEmpty returns the first argument that is not empty.
func firstNonEmpty(vals ...string) string {
	for _, v := range vals {
		if v != "" {
			return v
		}
	}
	return ""
}

// payloadOf strips the `data:` SSE marker and following space from a line,
// returning the JSON payload. The bool reports a marker.
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
