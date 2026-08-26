// Package agent runs the tool-calling loop that powers both the interactive
// REPL and the one-shot CLI. It streams a reply, executes any tool calls the
// model makes by feeding their results back into history, and repeats until the
// model answers in plain text.
package agent

import (
	"context"
	"errors"
	"io"
	"os"
	"strings"
	"time"

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

// ErrToolCancelled reports that a running tool was cancelled (e.g. by a user
// clicking Cancel in the UI). RunTurn returns it after committing the partial
// tool result (marked cancelled) and stopping the turn; the caller should end
// the turn cleanly rather than surfacing it as a failure.
var ErrToolCancelled = errors.New("tool call cancelled")

// RunHooks carries optional callbacks RunTurn invokes as a turn progresses.
type RunHooks struct {
	// OnRunStarted is called with the model's tool-call id and the run's cancel
	// function just before that tool starts. The cancel function aborts the
	// running tool — for a local run it kills the command's process group; for
	// a remote run it stops the wait on the stream and signals the connected
	// execution client — so a caller (e.g. the session, wired to a UI Cancel
	// button) can stop a runaway task. It is safe to call from any goroutine
	// and is a no-op after the run ends.
	OnRunStarted func(callID string, cancel func())
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
// presentation is the caller's job. Optional hooks let the caller observe each
// tool run as it starts and cancel it.
func RunTurn(ctx context.Context, client *llm.Client, history []llm.ChatMessage, js tools.Provider, emit func(api.Envelope), onMessage func(llm.ChatMessage) error, hooks ...RunHooks) (TurnResult, error) {
	var h RunHooks
	if len(hooks) > 0 {
		h = hooks[0]
	}
	res := TurnResult{History: history}
	// commit appends a finished message to the turn result and, when set,
	// streams it out so callers can commit each message the moment it's done.
	// A failing onMessage aborts the whole turn: the caller (the session store)
	// persists each committed message first and treats a persist failure as a
	// server fault, so there is no point continuing to run the turn.
	commit := func(m llm.ChatMessage) error {
		res.History = append(res.History, m)
		if onMessage != nil {
			if err := onMessage(m); err != nil {
				return err
			}
		}
		return nil
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
			if err := commit(llm.AssistantMessage(res.Text, reasoning, nil)); err != nil {
				return res, err
			}
			return res, nil
		}

		if err := commit(llm.AssistantMessage(reply.String(), reasoning, toLLMCalls(calls))); err != nil {
			return res, err
		}
		for _, c := range calls {
			// Each tool runs under its own context so it can be cancelled
			// independently of the turn (a user clicking Cancel in the UI stops
			// one runaway command without tearing down the whole session). The
			// hook fires before the tool starts so the caller can register the
			// cancel and no Cancel click can race ahead of it.
			callCtx, callCancel := context.WithCancel(ctx)
			// Release the per-call context when the turn ends. The defer runs
			// after every callCtx.Err() check below, so those checks see only a
			// user's cancellation (via the hook), never our own cleanup.
			defer callCancel()
			if h.OnRunStarted != nil {
				h.OnRunStarted(c.ID, callCancel)
			}
			stream, err := js.Run(callCtx, c.Name, []byte(c.Arguments))
			if err != nil {
				// The tool never started; there is nothing to stream, so emit the
				// terminal envelope directly (matching the old single-shot shape).
				result := "error: " + err.Error()
				if emit != nil {
					emit(api.Envelope{Kind: api.KindToolResult, ToolCallID: c.ID, Name: c.Name, Arguments: c.Arguments, Result: result})
				}
				if err := commit(llm.ToolResult(c.ID, result)); err != nil {
					return res, err
				}
				if callCtx.Err() != nil {
					// Cancelled while it was starting; don't feed the failure
					// back to the model, just stop the turn.
					return res, ErrToolCancelled
				}
				continue
			}
			// The tool is now running. Broadcast the start with the server's
			// clock before reading any output, so clients can show an honest
			// queued -> running transition with an elapsed timer even for a
			// silent long-running tool.
			startedAt := time.Now().UnixMilli()
			if emit != nil {
				emit(api.Envelope{Kind: api.KindToolStarted, ToolCallID: c.ID, Name: c.Name, Arguments: c.Arguments, StartedAt: startedAt})
			}
			// Stream the result out as it arrives instead of buffering it all
			// first: long-running tools (tests, builds, tail -f) render live in
			// the UI. Each chunk is broadcast as a KindToolResultDelta; the
			// terminal KindToolResult below carries the assembled full result and
			// the start/finish clocks so subscribers reconcile to one complete,
			// server-timed record.
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
			finishedAt := time.Now().UnixMilli()

			// Cancelled while it ran: emit the terminal tool_cancelled envelope,
			// commit the partial result marked cancelled so history is
			// transparent, and stop the turn — the user asked to stop, so don't
			// feed the partial output back to the model and keep spending
			// tokens. The stream already ended because cancellation closed it
			// (the local runner kills its process group; the remote runner
			// closes its pipe), so we get here promptly.
			if callCtx.Err() != nil {
				// A tool killed before producing any output (sleep 30, a silent
				// build) leaves an empty partial result. The committed tool
				// message must still carry content: it is what the next LLM
				// turn receives for this call, and most providers reject a
				// role-"tool" message whose content field is missing. Fall back
				// to an explicit marker so the model (and history) sees the run
				// was aborted rather than a tool that returned nothing.
				partial := result.String()
				if strings.TrimSpace(partial) == "" {
					partial = "(cancelled)"
				}
				if emit != nil {
					emit(api.Envelope{Kind: api.KindToolCancelled, ToolCallID: c.ID, Name: c.Name, Arguments: c.Arguments, StartedAt: startedAt, FinishedAt: finishedAt, Result: partial})
				}
				m := llm.ToolResult(c.ID, partial)
				m.StartedAt = startedAt
				m.FinishedAt = finishedAt
				m.Cancelled = true
				if err := commit(m); err != nil {
					return res, err
				}
				return res, ErrToolCancelled
			}

			if emit != nil {
				emit(api.Envelope{Kind: api.KindToolResult, ToolCallID: c.ID, Name: c.Name, Arguments: c.Arguments, StartedAt: startedAt, FinishedAt: finishedAt, Result: result.String()})
			}
			// The committed tool message carries the server clocks (json:"-" so
			// they never reach the model or the history API), letting /view
			// render timing on reload. Committing in completion order is what
			// keeps history (and the live DOM) ordered by completion time.
			m := llm.ToolResult(c.ID, result.String())
			m.StartedAt = startedAt
			m.FinishedAt = finishedAt
			if err := commit(m); err != nil {
				return res, err
			}
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
