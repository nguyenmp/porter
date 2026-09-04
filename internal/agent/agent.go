// Package agent runs the tool-calling loop that powers both the interactive
// REPL and the one-shot CLI. It streams a reply, executes any tool calls the
// model makes by feeding their results back into history, and repeats until the
// model answers in plain text.
package agent

import (
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"strings"
	"time"

	"porter/internal/api"
	"porter/internal/codec"
	"porter/internal/llm"
	"porter/internal/recall"
	"porter/internal/tools"
)

// Usage reports token counts for a single assistant turn (which may span
// several round-trips when tools are called). Input is carried explicitly as
// its two parts — CachedInput (prompt tokens the provider served from its
// cache) and UncachedInput (the rest, cache misses) — because the two price
// differently; Input() returns the derived total.
type Usage struct {
	CachedInput   int
	UncachedInput int
	Output        int
}

// Input returns the turn's total input tokens (cached + uncached).
func (u Usage) Input() int { return u.CachedInput + u.UncachedInput }

// Query reports the outcome of one model request — a single agent loop
// iteration (one LLM stream). It is the origin of token usage and request
// failures: the caller persists it so per-turn totals and failed-turn errors
// survive a reload, and the per-turn Usage is just the sum over a turn's
// queries.
type Query struct {
	// Idx is the request's zero-based position within its turn (the agent runs
	// requests sequentially, so 0 is the first request of the turn).
	Idx int
	// CachedInput/UncachedInput are this request's prompt-token split (cache
	// hits vs misses); Output is its completion tokens.
	CachedInput   int
	UncachedInput int
	Output        int
	// Stopped is set when the turn was aborted by the user (the Stop button)
	// rather than completing or failing. It is set on the request that was
	// streaming when the stop landed (with its partial usage), or on a request
	// that never ran (zero usage), so a reload can mark the turn stopped.
	Stopped bool
	// Err is set when the request itself failed (e.g. a provider error), which
	// ends the turn immediately.
	Err error
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

// ErrTurnStopped reports that the user stopped the whole turn (the Stop button
// in the UI) before it produced a final reply. RunTurn returns it when its
// context is cancelled while the model is streaming (after committing any
// partial reply, marked interrupted) or while a tool runs (after committing the
// cancelled tool result); the caller should end the turn with a stopped marker
// rather than a failure.
var ErrTurnStopped = errors.New("turn stopped")

// interruptedMarker is appended to a partial assistant reply when the user
// stops the turn mid-stream, so the committed history — and the model on the
// next turn — knows the reply was cut off rather than complete.
const interruptedMarker = "... [interrupted]"

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
	// OnQuery is called once per model request, on both success and failure,
	// with the request's token usage and (on failure) the error that ended the
	// turn. It lets a caller that owns persistence record each request at its
	// origin, so per-turn totals and failed-turn errors can be rebuilt on a
	// reload instead of only living on the live bus. A returned error aborts
	// the turn, mirroring how a failing onMessage aborts it.
	OnQuery func(Query) error
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

	// Repeated identical failing calls within one turn (a model re-issuing the
	// same malformed call, as MCP validation errors invite) are flagged on the
	// second failure and blocked outright from the cap onward, so the loop
	// does not burn round-trips on the same mistake. Keyed by tool name + raw
	// arguments; a success with the same key clears the record, so a call that
	// was actually fixed starts a fresh budget.
	repeatCapAt := 3 // an identical call may fail this many times, then it is blocked
	type failRec struct {
		prev  string // first failure's text, for the "already failed" hint
		count int
	}
	lastFails := map[string]*failRec{}
	blockedText := func(key string) string {
		name := key
		if i := strings.IndexByte(key, 0); i >= 0 {
			name = key[:i]
		}
		n := 1
		if f := lastFails[key]; f != nil {
			n = f.count
		}
		return fmt.Sprintf("[this tool call is blocked: %s has already failed %d times in a row with these exact arguments. Do not repeat it; fix the arguments (see the error and the tool's inputSchema), call a different tool, or stop tool use and reply.]", name, n)
	}
	noteRepeat := func(key, result string) string {
		f, ok := lastFails[key]
		if !ok {
			lastFails[key] = &failRec{prev: result, count: 1}
			return result
		}
		f.count++
		if f.count >= repeatCapAt {
			return result + "\n\n" + blockedText(key)
		}
		return result + "\n\n[this exact tool call already failed earlier in this turn: " + noteTrim(f.prev) + " — change the arguments or the approach]"
	}

	for i := 0; ; i++ {
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
				usage.CachedInput += ev.CachedInputTokens
				usage.UncachedInput += ev.UncachedInputTokens
				usage.Output += ev.OutputTokens
			case codec.TypeReasoningDelta:
			default:
				panic("unhandled event type: " + string(ev.Type))
			}
			if emit != nil {
				emit(api.Envelope{Kind: api.KindLLM, Event: &ev})
			}
		}

		// A Stop that landed between requests (after a tool finished, before the
		// model was called again) must end the turn here rather than start
		// another request.
		if ctx.Err() != nil {
			if qerr := h.reportQuery(Query{Idx: i, Stopped: true}); qerr != nil {
				return res, qerr
			}
			return res, ErrTurnStopped
		}

		// The model's view is the committed history projected for the model:
		// tool results larger than the head+tail budget are trimmed to a head +
		// tail slice (the full output stays in History, the DB, and the UI), and
		// recall_tool_output (recall) windows are kept intact. The projection is pure —
		// History always holds full output, so each request re-projects fresh and
		// never compounds.
		msgs := recall.ProjectModelView(res.History)
		if env := js.Environment(); env != "" {
			msgs = append([]llm.ChatMessage{llm.SystemMessage(env)}, msgs...)
		}
		// recall_tool_output is served by the agent itself (from History), so it is
		// declared alongside the provider's tools on every request.
		defs := append([]llm.Tool{recall.Def()}, js.Defs()...)
		// Wall-clock bounds of this model request: started just before the
		// stream opens, finished once it closes. They are stamped on the
		// assistant message(s) this request commits so the UI can show when
		// generation began and how long it took (and derive a tokens/second
		// rate). The clocks are json:"-" on ChatMessage — they never serialize
		// as fields into the request or the model's context — but the model-view
		// projection (recall) surfaces them to the model as compact bracketed
		// text on the outgoing copy of history, so the model sees when things
		// happened too.
		genStart := time.Now().UnixMilli()
		body, err := client.Stream(ctx, msgs, defs)
		if err != nil {
			// The request failed to start. If the user stopped the turn, this is
			// the stop (the transport was cancelled before the request began),
			// not a provider failure: end the turn cleanly.
			if ctx.Err() != nil {
				if qerr := h.reportQuery(Query{Idx: i, Stopped: true}); qerr != nil {
					return res, qerr
				}
				return res, ErrTurnStopped
			}
			// The request itself failed (e.g. a provider error): report it as a
			// failed query before ending the turn.
			if qerr := h.reportQuery(Query{Idx: i, Err: err}); qerr != nil {
				return res, qerr
			}
			return res, err
		}
		streamDone := false
		for line := range llm.SSELines(body) {
			done, err := dec.Process(line)
			if err != nil {
				// A decode failure on a stopped stream is the stop (the
				// transport was cut mid-line), not a failed request: break out
				// and let the stop path below commit the partial.
				if ctx.Err() != nil {
					break
				}
				body.Close()
				// A request that failed mid-stream is still a failed query:
				// report the partial usage and the error before ending.
				if qerr := h.reportQuery(Query{Idx: i, CachedInput: usage.CachedInput, UncachedInput: usage.UncachedInput, Output: usage.Output, Err: err}); qerr != nil {
					return res, qerr
				}
				return res, err
			}
			if done {
				streamDone = true
				break
			}
		}
		// Flush any terminal events the stream did not deliver as a [DONE]
		// marker (a provider that closes the SSE stream right after
		// finish_reason — the usage chunk may arrive in a separate trailing
		// chunk that we only reach by reading to EOF). Final is idempotent,
		// so this is a no-op when [DONE] already finalized the decoder.
		dec.Final()
		body.Close()
		genEnd := time.Now().UnixMilli()

		res.Usage.CachedInput += usage.CachedInput
		res.Usage.UncachedInput += usage.UncachedInput
		res.Usage.Output += usage.Output

		// The user stopped the turn mid-stream: commit any partial reply
		// (marked interrupted so the model knows it was cut off) and end the
		// turn. A fully assembled tool call is deliberately dropped — the tool
		// never ran, and a stop must not launch it. The stream must not have
		// reached a terminal state: a stop that lands in the same instant a
		// reply completes ([DONE] or a finish_reason with no [DONE]) lets the
		// turn finish normally rather than retroactively marking it stopped.
		if ctx.Err() != nil && !streamDone && !dec.Finished() {
			partial := reply.String()
			if strings.TrimSpace(partial) != "" || strings.TrimSpace(reasoning) != "" {
				text := partial
				if strings.TrimSpace(text) != "" {
					text = strings.TrimRight(text, " \t\n") + "\n\n" + interruptedMarker
				} else {
					text = interruptedMarker
				}
				assistant := llm.AssistantMessage(text, reasoning, nil)
				assistant.StartedAt = genStart
				assistant.FinishedAt = genEnd
				assistant.Output = usage.Output
				if err := commit(assistant); err != nil {
					return res, err
				}
				res.Text = text
			}
			if qerr := h.reportQuery(Query{Idx: i, CachedInput: usage.CachedInput, UncachedInput: usage.UncachedInput, Output: usage.Output, Stopped: true}); qerr != nil {
				return res, qerr
			}
			return res, ErrTurnStopped
		}

		// The request succeeded: report its usage so the caller can persist it
		// at the query's origin. Turns are derived (not stored) as the sum
		// over their queries, so this single record is what makes per-turn
		// totals rebuildable on a reload.
		if qerr := h.reportQuery(Query{Idx: i, CachedInput: usage.CachedInput, UncachedInput: usage.UncachedInput, Output: usage.Output}); qerr != nil {
			return res, qerr
		}

		if len(calls) == 0 {
			res.Text = reply.String()
			assistant := llm.AssistantMessage(res.Text, reasoning, nil)
			assistant.StartedAt = genStart
			assistant.FinishedAt = genEnd
			assistant.Output = usage.Output
			if err := commit(assistant); err != nil {
				return res, err
			}
			return res, nil
		}

		assistant := llm.AssistantMessage(reply.String(), reasoning, toLLMCalls(calls))
		assistant.StartedAt = genStart
		assistant.FinishedAt = genEnd
		assistant.Output = usage.Output
		if err := commit(assistant); err != nil {
			return res, err
		}
		for _, c := range calls {
			// A call already at the repeat cap is never issued again: commit
			// the block as its tool result (the assistant message advertising
			// the call is already committed, so history stays well-formed) and
			// move on instead of executing the same failing call once more.
			blockKey := c.Name + "\x00" + c.Arguments
			if f, ok := lastFails[blockKey]; ok && f.count >= repeatCapAt {
				blocked := blockedText(blockKey)
				if emit != nil {
					emit(api.Envelope{Kind: api.KindToolResult, ToolCallID: c.ID, Name: c.Name, Arguments: c.Arguments, Result: blocked})
				}
				if err := commit(llm.ToolResult(c.ID, blocked)); err != nil {
					return res, err
				}
				continue
			}
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
			// recall_tool_output is served by the agent itself from the turn's history:
			// it needs no execution provider, no cancel hook, and works for any
			// provider (local or remote) even when no client is connected.
			if c.Name == recall.ReadOutputTool {
				window, meta, rerr := recall.ServeWindow(res.History, c.Arguments)
				if rerr != nil {
					// A bad recall_tool_output call is a tool that failed to start: emit
					// the terminal envelope and commit the error, then keep the
					// turn going so the model sees the error and can react.
					result := "error: " + rerr.Error()
					if emit != nil {
						emit(api.Envelope{Kind: api.KindToolResult, ToolCallID: c.ID, Name: c.Name, Arguments: c.Arguments, Result: result})
					}
					if err := commit(llm.ToolResult(c.ID, result)); err != nil {
						return res, err
					}
					continue
				}
				// The model gets the full window in its context; the persisted and
				// broadcast copy is a short placeholder so the window bytes are
				// never duplicated in the DB (they live once, under the source
				// tool result). The window must reach res.History directly, not
				// through commit (which would persist it), so the two halves are
				// written separately here.
				placeholder := recall.Placeholder(meta)
				if emit != nil {
					emit(api.Envelope{Kind: api.KindToolResult, ToolCallID: c.ID, Name: c.Name, Arguments: c.Arguments, Result: placeholder, ToolOutput: meta})
				}
				windowMsg := llm.ToolResult(c.ID, window)
				windowMsg.ToolOutput = meta
				res.History = append(res.History, windowMsg)
				if onMessage != nil {
					placeholderMsg := llm.ToolResult(c.ID, placeholder)
					placeholderMsg.ToolOutput = meta
					if err := onMessage(placeholderMsg); err != nil {
						return res, err
					}
				}
				continue
			}
			if h.OnRunStarted != nil {
				h.OnRunStarted(c.ID, callCancel)
			}
			stream, err := js.Run(callCtx, c.Name, []byte(c.Arguments))
			if err != nil {
				// The tool never started; there is nothing to stream, so emit the
				// terminal envelope directly (matching the old single-shot shape).
				result := noteRepeat(c.Name+"\x00"+c.Arguments, "error: "+err.Error())
				meta := recall.Meta(result)
				if emit != nil {
					emit(api.Envelope{Kind: api.KindToolResult, ToolCallID: c.ID, Name: c.Name, Arguments: c.Arguments, Result: result, ToolOutput: meta})
				}
				m := llm.ToolResult(c.ID, result)
				m.ToolOutput = meta
				if err := commit(m); err != nil {
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
			errFailed := false
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
						errFailed = true
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
					emit(api.Envelope{Kind: api.KindToolCancelled, ToolCallID: c.ID, Name: c.Name, Arguments: c.Arguments, StartedAt: startedAt, FinishedAt: finishedAt, Result: partial, ToolOutput: recall.Meta(partial)})
				}
				m := llm.ToolResult(c.ID, partial)
				m.StartedAt = startedAt
				m.FinishedAt = finishedAt
				m.Cancelled = true
				m.ToolOutput = recall.Meta(partial)
				if err := commit(m); err != nil {
					return res, err
				}
				// A Stop cancels the whole turn's context, which also cancels
				// this run's context: distinguish it from a per-tool Cancel so
				// the turn ends with a stopped marker (and a stopped query is
				// persisted for reload) rather than a plain tool cancellation.
				if ctx.Err() != nil {
					if qerr := h.reportQuery(Query{Idx: i + 1, Stopped: true}); qerr != nil {
						return res, qerr
					}
					return res, ErrTurnStopped
				}
				return res, ErrToolCancelled
			}

			// A repeated identical failure gets the hint appended (a success
			// clears the record), so the model does not loop on the same call.
			final := result.String()
			key := c.Name + "\x00" + c.Arguments
			if errFailed || strings.HasPrefix(final, "error:") {
				final = noteRepeat(key, final)
			} else {
				delete(lastFails, key)
			}

			meta := recall.Meta(final)
			if emit != nil {
				emit(api.Envelope{Kind: api.KindToolResult, ToolCallID: c.ID, Name: c.Name, Arguments: c.Arguments, StartedAt: startedAt, FinishedAt: finishedAt, Result: final, ToolOutput: meta})
			}
			// The committed tool message carries the server metadata (json:"-" so
			// they never reach the model or the history API), letting /view
			// render timing on reload. Committing in completion order is what
			// keeps history (and the live DOM) ordered by completion time.
			m := llm.ToolResult(c.ID, final)
			m.StartedAt = startedAt
			m.FinishedAt = finishedAt
			m.ToolOutput = meta
			if err := commit(m); err != nil {
				return res, err
			}
		}
	}
}

// noteTrim shortens a prior failure's text for embedding in a repeat note so
// a long error does not inflate the model's context.
func noteTrim(s string) string {
	s = strings.Join(strings.Fields(s), " ")
	if len(s) > 140 {
		return s[:140] + "…"
	}
	return s
}

// reportQuery hands one request's outcome to the OnQuery hook (if any),
// returning the hook's error. It is called on every exit from the request
// phase — success and both failure paths — so the caller persists each request
// exactly once, at its origin.
func (h RunHooks) reportQuery(q Query) error {
	if h.OnQuery == nil {
		return nil
	}
	return h.OnQuery(q)
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
