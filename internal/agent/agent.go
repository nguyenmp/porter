// Package agent runs the tool-calling loop that powers both the interactive
// REPL and the one-shot CLI. It streams a reply, executes any tool calls the
// model makes by feeding their results back into history, and repeats until the
// model answers in plain text.
package agent

import (
	"context"
	"io"
	"os"
	"strings"

	"porter/internal/codec"
	"porter/internal/llm"
	"porter/internal/tools"
)

// Usage reports token counts for a single assistant turn (which may span
// several round-trips when tools are called).
type Usage struct {
	Input  int
	Output int
}

// TurnResult is what one RunTurn call produced.
type TurnResult struct {
	// Text is the final human-visible assistant reply, empty until the model
	// stops calling tools.
	Text string
	// Token usage summed across the whole turn.
	Usage Usage
	// History is History appended with every message produced this turn
	// (assistant calls + tool results + the final reply).
	History []llm.ChatMessage
}

// RunTurn drives one conversation turn. It reads history and extends it so the
// caller can keep it for the next turn. Every event the loop produces (message
// deltas, reasoning, tool calls and results, usage) goes to emit, so the caller
// can render, persist, or relay it without the loop knowing the destination.
// The loop does no rendering; presentation is the caller's job.
func RunTurn(ctx context.Context, client *llm.Client, history []llm.ChatMessage, js tools.Provider, emit func(codec.Event)) (TurnResult, error) {
	res := TurnResult{History: history}

	for {
		var reply strings.Builder
		var calls []codec.ToolCall
		var usage Usage

		dec := codec.NewDecoder(nil)
		dec.OnEvent = func(ev codec.Event) {
			switch ev.Type {
			case codec.TypeMessageDelta:
				reply.WriteString(ev.Delta)
			case codec.TypeMessage:
				reply.Reset()
				reply.WriteString(ev.Content)
			case codec.TypeToolCall:
				calls = append(calls, codec.ToolCall{ID: ev.ToolCallID, Name: ev.Name, Arguments: ev.Arguments})
			case codec.TypeUsage:
				usage.Input += ev.InputTokens
				usage.Output += ev.OutputTokens
			case codec.TypeReasoningDelta, codec.TypeToolResult:
			default:
				panic("unhandled event type: " + string(ev.Type))
			}
			if emit != nil {
				emit(ev)
			}
		}

		body, err := client.Stream(ctx, res.History, js.Defs())
		if err != nil {
			return res, err
		}
		for line := range llm.SSELines(body) {
			done, err := dec.Process(line)
			if err != nil {
				body.Close()
				return res, err
			}
			if done {
				break
			}
		}
		body.Close()

		res.Usage.Input += usage.Input
		res.Usage.Output += usage.Output

		if len(calls) == 0 {
			res.Text = reply.String()
			res.History = append(res.History, llm.AssistantMessage(res.Text, nil))
			return res, nil
		}

		res.History = append(res.History, llm.AssistantMessage(reply.String(), toLLMCalls(calls)))
		for _, c := range calls {
			result, err := js.Run(c.Name, []byte(c.Arguments))
			if err != nil {
				result = "error: " + err.Error()
			}
			if emit != nil {
				emit(codec.Event{Type: codec.TypeToolResult, ToolCallID: c.ID, Name: c.Name, Result: result})
			}
			res.History = append(res.History, llm.ToolResult(c.ID, result))
		}
	}
}

// toLLMCalls converts the codec-level tool calls into llm messages.
func toLLMCalls(calls []codec.ToolCall) []llm.ToolCall {
	out := make([]llm.ToolCall, 0, len(calls))
	for _, c := range calls {
		out = append(out, llm.ToolCall{
			ID:   c.ID,
			Type: "function",
			Function: llm.ToolFunction{
				Name:      c.Name,
				Arguments: c.Arguments,
			},
		})
	}
	return out
}

// EncodeJSON returns a sink that writes each event as a JSONL line to w.
// Callers that want the raw event stream pass this to RunTurn: the one-shot
// CLI writes to stdout, and a server relays the stream over HTTP.
func EncodeJSON(w io.Writer) func(codec.Event) {
	e := codec.NewEncoder(w)
	return func(ev codec.Event) { _ = e.Write(ev) }
}

// Render returns a sink that prints the human-readable conversation to w. It
// echoes message deltas and dims reasoning and tool-call lines when w is a
// terminal. It emits no structured events; JSONL stays with the caller.
func Render(w io.Writer, dim bool) func(codec.Event) {
	return func(ev codec.Event) {
		switch ev.Type {
		case codec.TypeMessageDelta:
			io.WriteString(w, ev.Delta)
		case codec.TypeReasoningDelta:
			writeDimmed(w, dim, ev.Reasoning)
		case codec.TypeToolCall:
			writeDimmed(w, dim, "\n> "+ev.Name+": "+ev.Arguments+"\n")
		}
	}
}

// IsTerminal reports whether w is an interactive character device.
func IsTerminal(w io.Writer) bool {
	f, ok := w.(*os.File)
	if !ok {
		return false
	}
	st, err := f.Stat()
	if err != nil {
		return false
	}
	return st.Mode()&os.ModeCharDevice != 0
}

// writeDimmed writes s, wrapped in dim escape codes when dim is true.
func writeDimmed(w io.Writer, dim bool, s string) {
	if !dim {
		io.WriteString(w, s)
		return
	}
	io.WriteString(w, "\x1b[2m"+s+"\x1b[0m")
}