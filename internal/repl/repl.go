// Package repl is the interactive terminal shell. When stdin is a TTY, porter
// runs a REPL here instead of the one-shot path. Stdout shows the conversation,
// and the structured JSONL event stream goes to stderr. The REPL is a stateless
// client: it creates a session, renders history from a poll, appends user
// messages, and subscribes to the event bus to stream the reply live. The
// server owns all conversation state.
package repl

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"strings"
	"time"

	"porter/internal/agent"
	"porter/internal/api"
	"porter/internal/client"
	"porter/internal/codec"
	"porter/internal/config"
	"porter/internal/exec"
	"porter/internal/llm"
	"porter/internal/tools"
)

// Run drives a multi-turn conversation. in supplies the user's lines, out shows
// the human-readable view (prompt + streamed reply), and jsonl receives the
// structured event stream (normally stderr). When cfg.LogFile is set, the event
// stream and progress lines go to that file instead, so an interactive
// container stays quiet.
func Run(ctx context.Context, cfg config.ClientConfig, in io.Reader, out, jsonl io.Writer) error {
	// Tie the long-lived execution-provider connection to this Run so it closes
	// when we return; otherwise a held /exec connection blocks server shutdown.
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	if cfg.LogFile != "" {
		f, err := os.OpenFile(cfg.LogFile, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
		if err != nil {
			return fmt.Errorf("open log file: %w", err)
		}
		defer f.Close()
		jsonl = f
	}

	c := client.New(cfg.ServerURL)
	info, err := c.Create(ctx)
	if err != nil {
		return fmt.Errorf("create session: %w", err)
	}

	fmt.Fprintf(out, "session %s\n", info.ID)

	// Discover the environment this client runs in (system, working directory,
	// files, skills) so it can report it to the server as the session's
	// execution provider: the server injects it into the model and exposes
	// load_skill for the discovered skills. Skills live in the repo or user
	// roots (e.g. .agents/skills/*/SKILL.md); the data can go stale if skills
	// are edited after connecting, which is fine here.
	env, err := exec.Discover("")
	if err != nil {
		return fmt.Errorf("discover execution environment: %w", err)
	}
	fmt.Fprintf(jsonl, "execution provider: %s @ %s (%d skills)\n", env.System, env.CWD, len(env.Skills))

	// Act as the session's execution provider: hold the exec connection open and
	// run the shell and load_skill tool calls the agent sends on this host,
	// streaming the output back. Register the context on every (re)connect, so
	// a new provider that swaps in brings its own environment and skills.
	dispatcher := tools.NewDispatcherWithSkills(env.Skills)
	go func() {
		for {
			if err := c.PostExecContext(ctx, info.ID, env); err != nil && ctx.Err() == nil {
				fmt.Fprintf(jsonl, "execution provider: context register failed: %v\n", err)
			}
			if err := c.ServeExec(ctx, info.ID, dispatcher.Run); err != nil {
				if ctx.Err() != nil {
					return
				}
				// connection dropped; retry registration
			}
			select {
			case <-ctx.Done():
				return
			case <-time.After(time.Second):
			}
		}
	}()

	// The exec connection above is what registers the provider: it commits the
	// "execution provider connected" notice and makes the environment context
	// (system, cwd, files, skills) available for injection into every model
	// request. Registration runs in a goroutine, so without a wait a fast first
	// input — piped stdin, a script, an impatient keystroke — could start the
	// first turn before the provider is connected, silently dropping the
	// environment context and the notice from that request. Wait until the
	// connection is live; if it never comes up (e.g. the exec endpoint is
	// unreachable) log it and continue — the session falls back to running
	// tools in the server process, matching a client that never connected.
	if err := waitForExec(ctx, c, info.ID); err != nil {
		fmt.Fprintf(jsonl, "execution provider: %v\n", err)
	}

	// view is the single sink for everything the human-readable terminal shows.
	// It tracks the latest committed seq so the next subscribe resumes exactly
	// where the last one left off, relays live LLM events to both the JSONL
	// stream and the view, records system-side facts (tool results) on the
	// JSONL stream, and — crucially — renders a committed assistant message that
	// missed its live stream (the subscribe connected after the reply already
	// streamed) instead of dropping it.
	view := &liveView{out: out, jsonl: jsonl, dim: agent.IsTerminal(out), seq: info.Seq}
	untilTurnDone := func(env api.Envelope) bool { return env.Kind == api.KindTurnDone }

	for _, m := range info.History {
		renderMessage(out, m)
	}

	r := bufio.NewReader(in)
	for {
		fmt.Fprintln(out)
		fmt.Fprint(out, "> ")
		line, err := r.ReadString('\n')
		if line == "" && errors.Is(err, io.EOF) {
			fmt.Fprintln(out)
			return nil
		}
		if err != nil && !errors.Is(err, io.EOF) {
			return err
		}
		text := strings.TrimSpace(line)
		if text == "" {
			continue
		}
		switch text {
		case "quit", "exit":
			return nil
		}

		if err := c.Append(ctx, info.ID, text); err != nil {
			return fmt.Errorf("append: %w", err)
		}
		for {
			err := c.Subscribe(ctx, info.ID, view.seq, view.emit, untilTurnDone)
			if errors.Is(err, client.ErrResync) {
				h, herr := c.History(ctx, info.ID)
				if herr != nil {
					return herr
				}
				for _, m := range h.History {
					renderMessage(out, m)
				}
				view.seq = h.Seq
				continue
			}
			if err != nil {
				return err
			}
			break
		}
		if fl, ok := out.(interface{ Flush() }); ok {
			fl.Flush()
		}
	}
}

// liveView renders a session's event bus to the human-readable view. Live LLM
// events (message/reasoning/tool-call deltas) are emitted by the server in
// real time only — never replayed — while committed messages and turn markers
// are logged and replayed. That asymmetry makes a subscriber that connects
// after a fast reply has streamed miss every live delta. sawLive lets the view
// notice that case: a committed assistant message is rendered from the
// authoritative committed copy only when its live stream was never seen, so a
// reply is never lost to the race and never rendered twice.
type liveView struct {
	out   io.Writer
	jsonl io.Writer
	dim   bool
	seq   uint64
	// sawLive reports whether the current assistant reply is being streamed
	// live. It resets on each committed user message (the start of a turn).
	sawLive bool
}

// tokenLine renders a turn's token usage line. When the provider reported cache
// hits the input split is shown explicitly: "(X cached + Y miss in, Z out
// tokens)". Otherwise it is the plain "(N in, M out tokens)" line. Mirrors the
// server's tokenLine so the REPL and web UI agree.
func tokenLine(cached, uncached, output int) string {
	if cached > 0 {
		return fmt.Sprintf("(%d cached + %d miss in, %d out tokens)", cached, uncached, output)
	}
	return fmt.Sprintf("(%d in, %d out tokens)", uncached, output)
}

// emit handles one envelope from the session's event bus.
func (v *liveView) emit(env api.Envelope) {
	if env.Seq > v.seq {
		v.seq = env.Seq
	}
	switch env.Kind {
	case api.KindTurnDone:
		// A turn that failed (e.g. the LLM provider returned an error) surfaces
		// its error instead of looking like a finished turn with no reply: print
		// it on the human view and record it on the structured stream.
		if env.Error != "" {
			fmt.Fprintf(v.out, "error: %s\n", env.Error)
			writeJSONL(v.jsonl, env)
			return
		}
		// A turn the user stopped (the Stop button) ends with a stopped marker
		// instead of a token line — or both, when partial usage was recorded.
		if env.Stopped {
			if env.CachedInput > 0 || env.UncachedInput > 0 || env.Output > 0 {
				fmt.Fprintf(v.out, "stopped · %s\n", tokenLine(env.CachedInput, env.UncachedInput, env.Output))
			} else {
				fmt.Fprintln(v.out, "stopped")
			}
			return
		}
		if env.CachedInput > 0 || env.UncachedInput > 0 || env.Output > 0 {
			fmt.Fprintln(v.out, tokenLine(env.CachedInput, env.UncachedInput, env.Output))
		}
	case api.KindToolResult, api.KindToolResultDelta, api.KindToolStarted, api.KindToolCancelled:
		// Streaming deltas, the start marker, and a user cancellation are
		// forwarded as they arrive so the JSONL log shows the result in real
		// time; the terminal KindToolResult still carries the full record (and
		// the committed tool message is the replay source).
		writeJSONL(v.jsonl, env)
	case api.KindLLM:
		if env.Event == nil {
			return
		}
		ev := *env.Event
		switch ev.Type {
		case codec.TypeMessageDelta, codec.TypeReasoningDelta, codec.TypeToolCall, codec.TypeToolCallDelta:
			v.sawLive = true
		}
		agent.EncodeJSON(v.jsonl)(ev)
		agent.Render(v.out, v.dim)(ev)
	case api.KindMessage:
		if env.Message == nil {
			return
		}
		m := *env.Message
		switch m.Role {
		case "user":
			// A new turn begins with its user message; the next assistant reply
			// has not streamed live yet.
			v.sawLive = false
		case "assistant":
			// If the live stream was missed (subscribed late, or the turn was
			// replayed from the bus), the committed message is the only copy —
			// render it now so a reply is never lost to that race.
			if !v.sawLive {
				renderMessage(v.out, m)
				// The live stream was missed, so this committed message is the only
				// record of the reply. Capture it on the structured stream (log) too,
				// mirroring the terminal TypeMessage event the live decoder emits.
				agent.EncodeJSON(v.jsonl)(codec.Event{
					Type:      codec.TypeMessage,
					Role:      "assistant",
					Content:   m.Content,
					Reasoning: m.Reasoning,
				})
			}
		case "system":
			// A committed system message (e.g. an execution-provider change
			// notice) renders dimmed so the user sees the environment change.
			renderMessage(v.out, m)
		}
	}
}

// renderMessage prints one committed conversation message to w, matching the
// live view: content as plain text, reasoning dimmed, and tool calls as
// `> name: args` lines. It is used for the initial/resync history dump and for
// committed assistant messages that arrive with no live stream.
func renderMessage(w io.Writer, m llm.ChatMessage) {
	dim := agent.IsTerminal(w)
	switch m.Role {
	case "user":
		fmt.Fprintf(w, "\n> %s\n", m.Content)
	case "assistant":
		if m.Reasoning != "" {
			writeDimmed(w, dim, m.Reasoning)
		}
		fmt.Fprintln(w, m.Content)
		for _, c := range m.ToolCalls {
			writeDimmed(w, dim, "\n> "+c.Function.Name+": "+c.Function.Arguments+"\n")
		}
	case "system":
		writeDimmed(w, dim, "\n[sys] "+m.Content+"\n")
	}
}

// writeDimmed writes s, wrapped in dim escape codes when dim is true. It mirrors
// agent's internal renderer so committed messages render with the same styling
// as their live-streamed counterparts.
func writeDimmed(w io.Writer, dim bool, s string) {
	if !dim {
		io.WriteString(w, s)
		return
	}
	io.WriteString(w, "\x1b[2m"+s+"\x1b[0m")
}

// writeJSONL writes v as a single NDJSON line to w.
func writeJSONL(w io.Writer, v any) {
	data, err := json.Marshal(v)
	if err != nil {
		return
	}
	_, _ = w.Write(append(data, '\n'))
}

// waitForExec polls the session's exec status until the client's exec
// connection has registered (Connected), so the first turn always sees the
// environment context and the connect notice. It gives up after a short
// timeout so a broken or unreachable exec endpoint degrades to local
// execution in the server process instead of hanging the REPL.
func waitForExec(ctx context.Context, c *client.Client, id string) error {
	const timeout = 10 * time.Second
	deadline := time.Now().Add(timeout)
	var lastErr error
	for {
		st, err := c.ExecStatus(ctx, id)
		if err == nil && st.Connected {
			return nil
		}
		if err != nil {
			lastErr = err
		}
		if ctx.Err() != nil {
			return ctx.Err()
		}
		if time.Now().After(deadline) {
			if lastErr != nil {
				return fmt.Errorf("exec provider did not connect within %s: %w", timeout, lastErr)
			}
			return fmt.Errorf("exec provider did not connect within %s", timeout)
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(50 * time.Millisecond):
		}
	}
}
