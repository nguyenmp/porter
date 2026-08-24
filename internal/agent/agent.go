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

	"porter/internal/api"
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
// caller can keep it for the next turn. Everything the loop produces goes to
// emit as an api.Envelope — live LLM events (KindLLM) and the system-side tool
// results it runs (KindToolResult) — so the caller can render, persist, or
// relay it without the loop knowing the destination. Every message the loop
// finalizes — assistant messages carrying tool calls, each tool result, and the
// final plain reply — is also handed to onMessage, so a caller that owns
// conversation state can commit each message as it completes (rather than only
// receiving the assembled history at the end). The loop does no rendering;
// presentation is the caller's job.
func RunTurn(ctx context.Context, client *llm.Client, history []llm.ChatMessage, js tools.Provider, emit func(api.Envelope), onMessage func(llm.ChatMessage)) (TurnResult, error) {
	res := TurnResult{History: history}
	// commit appends a finished message to the turn result and, when set,
	// streams it out so callers can commit each message the moment it's done.
	commit := func(m llm.ChatMessage) {
		res.History = append(res.History, m)
		if onMessage != nil {
			onMessage(m)
		}
	}

	for {
		var reply strings.Builder
		var reasoning string
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
				reasoning = ev.Reasoning
			case codec.TypeToolCall:
				calls = append(calls, codec.ToolCall{ID: ev.ToolCallID, Name: ev.Name, Arguments: ev.Arguments})
			case codec.TypeToolCallDelta:
				// Streamed tool-call fragments are forwarded live to emit
				// below; the assembled TypeToolCall is what lands in the
				// committed message.
			case codec.TypeUsage:
				usage.Input += ev.InputTokens
				usage.Output += ev.OutputTokens
			case codec.TypeReasoningDelta:
			default:
				panic("unhandled event type: " + string(ev.Type))
			}
			if emit != nil {
				emit(api.Envelope{Kind: api.KindLLM, Event: &ev})
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
			commit(llm.AssistantMessage(res.Text, reasoning, nil))
			return res, nil
		}

		commit(llm.AssistantMessage(reply.String(), reasoning, toLLMCalls(calls)))
		for _, c := range calls {
			stream, err := js.Run(ctx, c.Name, []byte(c.Arguments))
			if err != nil {
				// The tool never started; there is nothing to stream, so emit the
				// terminal envelope directly (matching the old single-shot shape).
				result := "error: " + err.Error()
				if emit != nil {
					emit(api.Envelope{Kind: api.KindToolResult, ToolCallID: c.ID, Name: c.Name, Result: result})
				}
				commit(llm.ToolResult(c.ID, result))
				continue
			}
			// Stream the result out as it arrives instead of buffering it all
			// first: long-running tools (tests, builds, tail -f) render live in
			// the UI. Each chunk is broadcast as a KindToolResultDelta; the
			// terminal KindToolResult below carries the assembled full result so
			// subscribers reconcile to one complete record. The committed tool
			// message is unchanged — history still stores the full result.
			var result strings.Builder
			buf := make([]byte, 32*1024)
			for {
				n, rerr := stream.Read(buf)
				if n > 0 {
					chunk := string(buf[:n])
					result.WriteString(chunk)
					if emit != nil {
						emit(api.Envelope{Kind: api.KindToolResultDelta, ToolCallID: c.ID, Name: c.Name, Delta: chunk})
					}
				}
				if rerr != nil {
					if rerr != io.EOF {
						errChunk := "error: " + rerr.Error()
						result.WriteString(errChunk)
						if emit != nil {
							emit(api.Envelope{Kind: api.KindToolResultDelta, ToolCallID: c.ID, Name: c.Name, Delta: errChunk})
						}
					}
					break
				}
			}
			_ = stream.Close()
			if emit != nil {
				emit(api.Envelope{Kind: api.KindToolResult, ToolCallID: c.ID, Name: c.Name, Result: result.String()})
			}
			commit(llm.ToolResult(c.ID, result.String()))
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
