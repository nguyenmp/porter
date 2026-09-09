// Package agent runs the tool-calling loop that powers both the interactive
// REPL and the one-shot CLI. It streams a reply, executes any tool calls the
// model makes by feeding their results back into history, and repeats until the
// model answers in plain text.
package agent

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"porter/internal/api"
	"porter/internal/asset"
	"porter/internal/codec"
	"porter/internal/humanize"
	"porter/internal/llm"
	"porter/internal/recall"
	"porter/internal/spool"
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
	// PublishAsset stores a published asset's bytes on the server and returns
	// the asset's hosted URL (e.g. "/assets/sessions/..."), or an error when
	// the server has no asset store. It backs the publish_asset tool: the
	// agent reads the file through the active provider and hands the bytes
	// here, so storage lives wherever the caller (the session) decides.
	PublishAsset func(filename string, content []byte) (string, error)
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
		// repliedScrub strips a hallucinated "[replied ...]" header from the
		// model's own output: the model-view projection prepends that exact
		// note to assistant messages in the history sent on each request, and
		// a model sometimes imitates it and opens its reply with the same
		// line. The note is never part of stored content (the projection adds
		// it fresh), so a header here is always a copy, never intent. The
		// scrubber handles streamed deltas (so the header never flashes in the
		// live view); the decoder's assembled TypeMessage below is scrubbed
		// wholesale, because it is the authoritative text that lands in the
		// committed message. See recall's replied.go for the exact grammar.
		repliedScrub := recall.NewRepliedScrubber()
		dec.OnEvent = func(ev codec.Event) {
			switch ev.Type {
			case codec.TypeMessageDelta:
				// Route the delta through the scrubber: while the opening is
				// held or swallowed nothing is forwarded (the live view sees
				// no header), and the scrubbed remainder is what both the
				// reply buffer and the live stream accumulate.
				if out := repliedScrub.Feed(ev.Delta); out != "" {
					reply.WriteString(out)
					if emit != nil {
						ev.Delta = out
						emit(api.Envelope{Kind: api.KindLLM, Event: &ev})
					}
				}
				return
			case codec.TypeMessage:
				// The assembled full content (the decoder replays everything it
				// streamed). Scrub it wholesale: the scrubber may still be
				// holding a header split at a delta boundary, and its held
				// bytes are a prefix of this content, so this single scrub is
				// what guarantees the committed message is clean.
				ev.Content = recall.StripRepliedPrefix(ev.Content)
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
		// Every request leads with the standing plain-language writing style (a
		// static system prefix, so provider prompt caching keeps it cheap
		// across sessions and turns), then the execution provider's environment
		// context — where commands run, what's in the working directory, and
		// the skills and CLIs available — so the model knows its surroundings
		// and writes replies, drafts, and code comments plainly the first time.
		prefix := []llm.ChatMessage{llm.SystemMessage(humanize.Directive())}
		if env := js.Environment(); env != "" {
			prefix = append(prefix, llm.SystemMessage(env))
		}
		msgs = append(prefix, msgs...)
		// recall_tool_output and spool_output are served by the agent itself
		// (recall from History; spool from History plus a write on the active
		// provider), so they are declared alongside the provider's tools on
		// every request. AddToolContract then declares the per-call contract —
		// the required porter_action_description and porter_timeout_seconds
		// arguments — on every tool the model sees, in one place, so no tool
		// definition site needs to know about it.
		defs := llm.AddToolContract(append([]llm.Tool{recall.Def(), spool.Def(), asset.Def(), asset.IframeDef()}, js.Defs()...))
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
		var prevCancel, prevStop context.CancelFunc // previous call's contexts, released at the next call's start
		for _, c := range calls {
			// Releasing the previous call's contexts here, at the start of the
			// next iteration, stops its deadline timer the moment the call's
			// processing is done — every path below ends in continue or return,
			// so by the time we reach here the previous run is finished and its
			// cancellation can no longer be mistaken for an outcome. (The
			// defers on each context are the backstop for the return paths.)
			if prevStop != nil {
				prevStop()
				prevStop = nil
			}
			if prevCancel != nil {
				prevCancel()
				prevCancel = nil
			}
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
			// The per-call contract is enforced here, on every call the model
			// makes. A call missing porter_action_description or
			// porter_timeout_seconds never runs: it is rejected as a tool that
			// failed to start, so the model sees the error and retries with the
			// missing fields. Repeated identical rejections hit the repeat cap
			// above like any other failure.
			timeout, cerr := parseCallContract(c.Name, []byte(c.Arguments))
			if cerr != nil {
				result := noteRepeat(c.Name+"\x00"+c.Arguments, "error: "+cerr.Error())
				if emit != nil {
					emit(api.Envelope{Kind: api.KindToolResult, ToolCallID: c.ID, Name: c.Name, Arguments: c.Arguments, Result: result})
				}
				if err := commit(llm.ToolResult(c.ID, result)); err != nil {
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
			prevCancel = callCancel
			// Release the per-call context when the turn ends. The defer runs
			// after every callCtx.Err() check below, so those checks see only a
			// user's cancellation (via the hook), never our own cleanup.
			defer callCancel()
			// The run also carries the deadline the model set in
			// porter_timeout_seconds. When it fires, the tool is stopped exactly
			// like a user cancel — the local runner kills its process group, the
			// remote runner is signalled — but the outcome is a timeout, not a
			// cancel: the agent commits the partial output, marked timed out,
			// and continues the turn, so the model sees that its call ran too
			// long and can retry with a bigger bound or a different approach.
			// classifyRun below tells the two apart: a cancelled ancestor
			// context means the user stopped it; a fired deadline on runCtx
			// alone means a timeout.
			runCtx, runStop := context.WithTimeout(callCtx, timeout)
			prevStop = runStop
			defer runStop()
			// finishTool emits the terminal KindToolResult envelope for this
			// call and commits its message, marking it timed out when the run
			// hit its porter_timeout_seconds deadline. The cancel branches
			// below emit KindToolCancelled themselves — a cancel is not a
			// result — and every other terminal (normal, failed, timed out)
			// funnels through here so the envelope, the committed message, and
			// the flags cannot drift.
			finishTool := func(content string, timedOut bool, startedAt, finishedAt int64) error {
				meta := recall.Meta(content)
				if emit != nil {
					env := api.Envelope{Kind: api.KindToolResult, ToolCallID: c.ID, Name: c.Name, Arguments: c.Arguments, StartedAt: startedAt, FinishedAt: finishedAt, Result: content, ToolOutput: meta}
					if timedOut {
						env.TimedOut = true
					}
					emit(env)
				}
				m := llm.ToolResult(c.ID, content)
				m.StartedAt = startedAt
				m.FinishedAt = finishedAt
				m.ToolOutput = meta
				if timedOut {
					m.TimedOut = true
				}
				return commit(m)
			}
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
			// spool_output is served by the agent like recall_tool_output — it
			// finds the named tool result in the turn's history — but the write
			// itself runs on the active execution provider through a private
			// provider tool, so the file lands on the same filesystem shell and
			// the file tools edit (whatever provider is active, even when it is
			// not the one that produced the output). The bytes are written
			// verbatim: a raw mirror of the committed result, so the file never
			// drifts from history.
			if c.Name == spool.OutputTool {
				callID, path, perr := spool.ParseArgs(c.Arguments)
				if perr != nil {
					// A malformed spool_output call is a tool that failed to
					// start: emit the terminal envelope and commit the error,
					// then keep the turn going (same as recall_tool_output).
					result := "error: " + perr.Error()
					if emit != nil {
						emit(api.Envelope{Kind: api.KindToolResult, ToolCallID: c.ID, Name: c.Name, Arguments: c.Arguments, Result: result})
					}
					if err := commit(llm.ToolResult(c.ID, result)); err != nil {
						return res, err
					}
					continue
				}
				content, serr := spool.Lookup(res.History, callID)
				if serr != nil {
					result := "error: " + serr.Error()
					if emit != nil {
						emit(api.Envelope{Kind: api.KindToolResult, ToolCallID: c.ID, Name: c.Name, Arguments: c.Arguments, Result: result})
					}
					if err := commit(llm.ToolResult(c.ID, result)); err != nil {
						return res, err
					}
					continue
				}
				// Write through the active provider. The streamed confirmation
				// becomes the spool_output result the model sees; it names the
				// resolved absolute path so the model can shell or read it.
				startedAt := time.Now().UnixMilli()
				wstream, werr := js.Run(runCtx, tools.SpoolWriteTool, spool.WritePayload(path, content))
				finishedAt := time.Now().UnixMilli()
				if werr != nil {
					// The write never started (e.g. no execution client is
					// connected), or the deadline fired before it answered.
					// Report it like any tool that failed to start — except a
					// fired deadline is a timed-out result, which the model
					// sees and can react to, not a cancel.
					result := "error: " + werr.Error()
					if classifyRun(ctx, callCtx, runCtx) == runTimedOut {
						result += "\n" + timeoutMarker(timeout)
						if err := finishTool(result, true, 0, 0); err != nil {
							return res, err
						}
						continue
					}
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
						return res, ErrToolCancelled
					}
					continue
				}
				var confirm strings.Builder
				buf := make([]byte, 32*1024)
				for {
					n, rerr := wstream.Read(buf)
					if n > 0 {
						confirm.Write(buf[:n])
					}
					if rerr != nil {
						break
					}
				}
				_ = wstream.Close()
				final := confirm.String()
				switch cause := classifyRun(ctx, callCtx, runCtx); cause {
				case runStopped, runCancelled:
					// Cancelled while the write was in flight: commit the
					// partial (marked cancelled) and end like any cancelled
					// tool.
					if strings.TrimSpace(final) == "" {
						final = "(cancelled)"
					}
					if emit != nil {
						emit(api.Envelope{Kind: api.KindToolCancelled, ToolCallID: c.ID, Name: c.Name, Arguments: c.Arguments, StartedAt: startedAt, FinishedAt: finishedAt, Result: final, ToolOutput: recall.Meta(final)})
					}
					m := llm.ToolResult(c.ID, final)
					m.StartedAt = startedAt
					m.FinishedAt = finishedAt
					m.Cancelled = true
					m.ToolOutput = recall.Meta(final)
					if err := commit(m); err != nil {
						return res, err
					}
					if cause == runStopped {
						if qerr := h.reportQuery(Query{Idx: i + 1, Stopped: true}); qerr != nil {
							return res, qerr
						}
						return res, ErrTurnStopped
					}
					return res, ErrToolCancelled
				case runTimedOut:
					// The deadline fired while the write was in flight: report
					// the partial as a timed-out result and keep the turn going.
					marker := timeoutMarker(timeout)
					if strings.TrimSpace(final) == "" {
						final = marker
					} else {
						final += "\n" + marker
					}
					if err := finishTool(final, true, startedAt, finishedAt); err != nil {
						return res, err
					}
					continue
				}
				if err := finishTool(final, false, startedAt, finishedAt); err != nil {
					return res, err
				}
				continue
			}
			// render_iframe is served by the agent too, but it needs no
			// provider: the call only records which published asset to show
			// and how tall to make it. The committed result is a short
			// caption; the web UI recognizes the tool by name and draws the
			// message as an expanded sandboxed iframe instead of a collapsed
			// tool result, on both the live stream and /view reloads.
			if c.Name == asset.IframeTool {
				startedAt := time.Now().UnixMilli()
				args, ierr := asset.ParseIframeArgs(c.Arguments)
				if ierr != nil {
					result := "error: " + ierr.Error()
					if err := finishTool(result, false, startedAt, time.Now().UnixMilli()); err != nil {
						return res, err
					}
					continue
				}
				height := asset.DefaultIframeHeight
				if args.Height != nil {
					height = *args.Height
				}
				result := fmt.Sprintf("Rendered in a sandboxed iframe: %s (height %dpx).", args.Src, height)
				if err := finishTool(result, false, startedAt, time.Now().UnixMilli()); err != nil {
					return res, err
				}
				continue
			}
			// publish_asset is served by the agent like spool_output — the file
			// read runs on the active execution provider through a private
			// provider tool (_porter_asset_read), so publish_asset works for
			// any provider and the bytes land wherever the caller's asset
			// store points (the session's store, keyed by session id). The
			// model sees only the returned URL: the file bytes ride the
			// private channel and are never committed to history or fed back
			// into context.
			if c.Name == asset.PublishTool {
				path, perr := asset.ParseArgs(c.Arguments)
				if perr != nil {
					// A malformed publish_asset call is a tool that failed to
					// start: emit the terminal envelope and commit the error,
					// then keep the turn going (same as spool_output).
					result := "error: " + perr.Error()
					if emit != nil {
						emit(api.Envelope{Kind: api.KindToolResult, ToolCallID: c.ID, Name: c.Name, Arguments: c.Arguments, Result: result})
					}
					if err := commit(llm.ToolResult(c.ID, result)); err != nil {
						return res, err
					}
					continue
				}
				if h.PublishAsset == nil {
					result := "error: publish_asset: no asset store is configured on this server"
					if emit != nil {
						emit(api.Envelope{Kind: api.KindToolResult, ToolCallID: c.ID, Name: c.Name, Arguments: c.Arguments, Result: result})
					}
					if err := commit(llm.ToolResult(c.ID, result)); err != nil {
						return res, err
					}
					continue
				}
				// Read the file through the active provider. The read is a
				// private tool whose whole output is one JSON payload; it is
				// never streamed to the UI or committed.
				startedAt := time.Now().UnixMilli()
				rstream, rerr := js.Run(runCtx, tools.AssetReadTool, asset.ReadPayload(path))
				finishedAt := time.Now().UnixMilli()
				if rerr != nil {
					// The read never started (e.g. no execution client is
					// connected, the file is missing, or it exceeds the size
					// cap), or the deadline fired before it answered. Report
					// it like any tool that failed to start — except a fired
					// deadline is a timed-out result, which the model sees
					// and can react to, not a cancel.
					result := "error: " + rerr.Error()
					if classifyRun(ctx, callCtx, runCtx) == runTimedOut {
						result += "\n" + timeoutMarker(timeout)
						if err := finishTool(result, true, 0, 0); err != nil {
							return res, err
						}
						continue
					}
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
						return res, ErrToolCancelled
					}
					continue
				}
				var raw strings.Builder
				buf := make([]byte, 32*1024)
				for {
					n, rerr := rstream.Read(buf)
					if n > 0 {
						raw.Write(buf[:n])
					}
					if rerr != nil {
						break
					}
				}
				_ = rstream.Close()
				final := raw.String()
				switch cause := classifyRun(ctx, callCtx, runCtx); cause {
				case runStopped, runCancelled:
					// Cancelled while the read was in flight: commit the
					// partial (marked cancelled) and end like any cancelled
					// tool.
					partial := final
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
					if cause == runStopped {
						if qerr := h.reportQuery(Query{Idx: i + 1, Stopped: true}); qerr != nil {
							return res, qerr
						}
						return res, ErrTurnStopped
					}
					return res, ErrToolCancelled
				case runTimedOut:
					// The deadline fired while the read was in flight: report
					// it as a timed-out result and keep the turn going.
					marker := timeoutMarker(timeout)
					if strings.TrimSpace(final) == "" {
						final = marker
					} else {
						final += "\n" + marker
					}
					if err := finishTool(final, true, startedAt, finishedAt); err != nil {
						return res, err
					}
					continue
				}
				// All paths below funnel through finishTool on failure, so a
				// decode or store error commits a normal (non-timed-out)
				// result the model can read and react to.
				var rd struct {
					Path string `json:"path"`
					Size int    `json:"size"`
					Data string `json:"data"`
				}
				failAsset := func(format string, args ...any) error {
					result := "error: publish_asset: " + fmt.Sprintf(format, args...)
					return finishTool(result, false, startedAt, finishedAt)
				}
				if err := json.Unmarshal([]byte(final), &rd); err != nil {
					if err := failAsset("the file could not be read (malformed provider response: %v)", err); err != nil {
						return res, err
					}
					continue
				}
				if rd.Size < 0 || rd.Size > asset.MaxBytes {
					if err := failAsset("file is %d bytes; the limit is %d bytes (%d MiB)", rd.Size, asset.MaxBytes, asset.MaxBytes/(1<<20)); err != nil {
						return res, err
					}
					continue
				}
				data, err := base64.StdEncoding.DecodeString(rd.Data)
				if err != nil || len(data) != rd.Size {
					if err := failAsset("the file could not be decoded after upload (size mismatch)"); err != nil {
						return res, err
					}
					continue
				}
				filename := asset.SanitizeFilename(filepath.Base(rd.Path))
				url, aerr := h.PublishAsset(filename, data)
				if aerr != nil {
					if err := failAsset("%v", aerr); err != nil {
						return res, err
					}
					continue
				}
				if err := finishTool(url, false, startedAt, finishedAt); err != nil {
					return res, err
				}
				continue
			}
			if h.OnRunStarted != nil {
				h.OnRunStarted(c.ID, callCancel)
			}
			stream, err := js.Run(runCtx, c.Name, []byte(c.Arguments))
			if err != nil {
				// The tool never started; there is nothing to stream, so emit
				// the terminal envelope directly (matching the old single-shot
				// shape) — or, when the deadline fired while it was still
				// starting, a timed-out result the model can react to.
				result := "error: " + err.Error()
				if classifyRun(ctx, callCtx, runCtx) == runTimedOut {
					result += "\n" + timeoutMarker(timeout)
					if err := finishTool(result, true, 0, 0); err != nil {
						return res, err
					}
					continue
				}
				result = noteRepeat(c.Name+"\x00"+c.Arguments, result)
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

			// How the run ended decides what happens next. A user cancel stops
			// the turn: the partial result is committed (marked cancelled) for
			// transparency but never fed back to the model — the user asked to
			// stop, so don't keep spending tokens. A timeout is a tool outcome
			// like any other: the partial is committed (marked timed out, with
			// a marker in its content) and the turn continues, so the model
			// sees its call ran too long and can retry with a bigger bound,
			// narrow the command, or explain. The stream already ended in both
			// cases because the deadline/cancel closed it (the local runner
			// kills its process group; the remote runner closes its pipe), so
			// we get here promptly.
			final := result.String()
			switch cause := classifyRun(ctx, callCtx, runCtx); cause {
			case runStopped, runCancelled:
				// A tool killed before producing any output (sleep 30, a silent
				// build) leaves an empty partial result. The committed tool
				// message must still carry content: it is what the next LLM
				// turn receives for this call, and most providers reject a
				// role-"tool" message whose content field is missing. Fall back
				// to an explicit marker so the model (and history) sees the run
				// was aborted rather than a tool that returned nothing.
				partial := final
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
				if cause == runStopped {
					if qerr := h.reportQuery(Query{Idx: i + 1, Stopped: true}); qerr != nil {
						return res, qerr
					}
					return res, ErrTurnStopped
				}
				return res, ErrToolCancelled
			case runTimedOut:
				// The deadline fired. Mark the committed result timed out and
				// put the marker in its content — this is what the model reads
				// on the next request of this same turn — then continue the
				// turn. A timeout counts as a failure for the repeat guard: an
				// identical call (same command, same bound) that keeps timing
				// out is flagged instead of being retried forever, while a
				// retry with a bigger porter_timeout_seconds is a different
				// call and starts fresh.
				marker := timeoutMarker(timeout)
				if strings.TrimSpace(final) == "" {
					final = marker
				} else {
					final += "\n" + marker
				}
				final = noteRepeat(c.Name+"\x00"+c.Arguments, final)
				if err := finishTool(final, true, startedAt, finishedAt); err != nil {
					return res, err
				}
				continue
			}

			// Normal completion (or a stream error). A repeated identical
			// failure gets the hint appended (a success clears the record), so
			// the model does not loop on the same call.
			key := c.Name + "\x00" + c.Arguments
			if errFailed || strings.HasPrefix(final, "error:") {
				final = noteRepeat(key, final)
			} else {
				delete(lastFails, key)
			}
			// The committed tool message carries the server metadata (json:"-"
			// so they never reach the model or the history API), letting /view
			// render timing on reload. Committing in completion order is what
			// keeps history (and the live DOM) ordered by completion time.
			if err := finishTool(final, false, startedAt, finishedAt); err != nil {
				return res, err
			}
		}
	}
}

// parseCallContract extracts and validates the two required porter contract
// fields from a tool call's arguments: porter_action_description (the semantic
// goal, for the user to read) and porter_timeout_seconds (the run deadline the
// agent enforces). It returns an error naming exactly what is wrong so the
// model can fix the call; the agent rejects the call with that error and lets
// the model retry.
func parseCallContract(name string, args []byte) (time.Duration, error) {
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(args, &raw); err != nil {
		return 0, fmt.Errorf("%s: arguments are not valid JSON (%v)", name, err)
	}
	desc, ok := raw[llm.ArgPorterDescription]
	if !ok {
		return 0, fmt.Errorf("%s is missing its required %s argument: say what this call is for in one short sentence, then call the tool again", name, llm.ArgPorterDescription)
	}
	var d string
	if err := json.Unmarshal(desc, &d); err != nil {
		return 0, fmt.Errorf("%s: %s must be a sentence of text, got %s", name, llm.ArgPorterDescription, desc)
	}
	if strings.TrimSpace(d) == "" {
		return 0, fmt.Errorf("%s: %s is empty — say what this call is for in one short sentence, then call the tool again", name, llm.ArgPorterDescription)
	}
	rawTimeout, ok := raw[llm.ArgPorterTimeout]
	if !ok {
		return 0, fmt.Errorf("%s is missing its required %s argument: pick how many seconds this call may run (a whole number from 1 to %d), then call the tool again", name, llm.ArgPorterTimeout, llm.MaxPorterTimeoutSeconds)
	}
	var n json.Number
	if err := json.Unmarshal(rawTimeout, &n); err != nil {
		return 0, fmt.Errorf("%s: %s must be a number, got %s", name, llm.ArgPorterTimeout, rawTimeout)
	}
	secs, err := strconv.ParseInt(n.String(), 10, 64)
	if err != nil || secs < 1 || secs > llm.MaxPorterTimeoutSeconds {
		return 0, fmt.Errorf("%s: %s must be a whole number of seconds from 1 to %d, got %q — pick again and call the tool", name, llm.ArgPorterTimeout, llm.MaxPorterTimeoutSeconds, n.String())
	}
	return time.Duration(secs) * time.Second, nil
}

// timeoutMarker is the line appended to a tool result whose run hit its
// porter_timeout_seconds deadline. It lives in the committed content so the
// model sees it on the next request of the same turn; the TimedOut flag on the
// message and envelope is what history and the UI render from.
func timeoutMarker(d time.Duration) string {
	return fmt.Sprintf("(timed out after %ds)", int(d/time.Second))
}

// runCause says why a per-call run ended.
type runCause int

const (
	runFinished  runCause = iota // the tool completed (or failed) on its own
	runTimedOut                  // the porter_timeout_seconds deadline fired
	runCancelled                 // the user cancelled this one tool (Cancel)
	runStopped                   // the user stopped the whole turn (Stop)
)

// classifyRun reads the three contexts a run lives under to say why it ended.
// The distinction is what the whole timeout design hangs on: a deadline fires
// on runCtx alone, leaving its ancestors (callCtx, then the turn's ctx) alive;
// a user Cancel fires on callCtx; a Stop fires on the turn's ctx, which also
// cancels callCtx. Checking the ancestors first keeps a stop or cancel that
// happened to land at the same instant as a deadline reading as a stop or
// cancel, never a timeout.
func classifyRun(ctx, callCtx, runCtx context.Context) runCause {
	switch {
	case ctx.Err() != nil:
		return runStopped
	case callCtx.Err() != nil:
		return runCancelled
	case runCtx.Err() != nil:
		return runTimedOut
	}
	return runFinished
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
