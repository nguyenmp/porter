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
	// Act as the session's execution provider: hold the exec connection open and
	// run any shell tool calls the agent sends on this host, streaming the
	// output back. Re-register on reconnect until the session ends.
	dispatcher := tools.NewDispatcher()
	go func() {
		for {
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

// emit handles one envelope from the session's event bus.
func (v *liveView) emit(env api.Envelope) {
	if env.Seq > v.seq {
		v.seq = env.Seq
	}
	switch env.Kind {
	case api.KindTurnDone:
		if env.Input > 0 || env.Output > 0 {
			fmt.Fprintf(v.out, "(%d in, %d out tokens)\n", env.Input, env.Output)
		}
	case api.KindToolResult:
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
