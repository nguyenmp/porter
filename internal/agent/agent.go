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

// RunTurn drives one conversation turn. history is read and extended so the
// caller can keep it for the next turn. text receives the human-readable view
// (pass io.Discard when emitting only JSONL), while jsonl receives the
// structured event stream. Rendering dims reasoning only when text is a real
// terminal.
func RunTurn(ctx context.Context, client *llm.Client, history []llm.ChatMessage, js *tools.Dispatcher, text, jsonl io.Writer) (TurnResult, error) {
	res := TurnResult{History: history}
	dim := isTerminal(text)

	for {
		dec, enc := newCodec(jsonl)
		var reply strings.Builder
		var calls []codec.ToolCall
		var usage Usage

		dec.OnEvent = func(ev codec.Event) {
			switch ev.Type {
			case codec.TypeMessageDelta:
				io.WriteString(text, ev.Delta)
				reply.WriteString(ev.Delta)
			case codec.TypeMessage:
				reply.Reset()
				reply.WriteString(ev.Content)
			case codec.TypeReasoningDelta:
				writeDimmed(text, dim, ev.Reasoning)
			case codec.TypeToolCall:
				calls = append(calls, codec.ToolCall{ID: ev.ToolCallID, Name: ev.Name, Arguments: ev.Arguments})
				writeDimmed(text, dim, "\n> "+ev.Name+": "+ev.Arguments+"\n")
			case codec.TypeUsage:
				usage.Input += ev.InputTokens
				usage.Output += ev.OutputTokens
			}
		}

		body, err := client.Stream(ctx, res.History, tools.Defs())
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
			_ = enc.Write(codec.Event{Type: codec.TypeToolResult, ToolCallID: c.ID, Name: c.Name, Result: result})
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

// newCodec wires up the agent's shared streaming plumbing: an encoder that
// writes events to jsonl and a decoder that consumes raw SSE lines.
func newCodec(jsonl io.Writer) (*codec.Decoder, *codec.Encoder) {
	enc := codec.NewEncoder(jsonl)
	return codec.NewDecoder(enc), enc
}

// isTerminal reports whether w is an interactive character device.
func isTerminal(w io.Writer) bool {
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