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

	"porter/internal/agent"
	"porter/internal/api"
	"porter/internal/client"
	"porter/internal/config"
	"porter/internal/llm"
)

// Run drives a multi-turn conversation. in supplies the user's lines, out shows
// the human-readable view (prompt + streamed reply), and jsonl receives the
// structured event stream (normally stderr). When cfg.LogFile is set, the event
// stream and progress lines go to that file instead, so an interactive
// container stays quiet.
func Run(ctx context.Context, cfg config.ClientConfig, in io.Reader, out, jsonl io.Writer) error {
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

	// One sink tracks the latest committed seq so the next subscribe resumes
	// exactly where the last one left off, relays live LLM events to both the
	// JSONL stream and the human-readable view, and records system-side facts
	// (tool results) on the JSONL stream.
	seq := info.Seq
	emit := func(env api.Envelope) {
		if env.Seq > seq {
			seq = env.Seq
		}
		switch env.Kind {
		case api.KindTurnDone:
			if env.Input > 0 || env.Output > 0 {
				fmt.Fprintf(out, "(%d in, %d out tokens)\n", env.Input, env.Output)
			}
		case api.KindToolResult:
			writeJSONL(jsonl, env)
		case api.KindLLM:
			if env.Event == nil {
				return
			}
			ev := *env.Event
			agent.EncodeJSON(jsonl)(ev)
			agent.Render(out, agent.IsTerminal(out))(ev)
		}
	}
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
			err := c.Subscribe(ctx, info.ID, seq, emit, untilTurnDone)
			if errors.Is(err, client.ErrResync) {
				h, herr := c.History(ctx, info.ID)
				if herr != nil {
					return herr
				}
				for _, m := range h.History {
					renderMessage(out, m)
				}
				seq = h.Seq
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

// renderMessage prints one committed conversation message to w.
func renderMessage(w io.Writer, m llm.ChatMessage) {
	switch m.Role {
	case "user":
		fmt.Fprintf(w, "\n> %s\n", m.Content)
	case "assistant":
		fmt.Fprintln(w, m.Content)
	}
}

// writeJSONL writes v as a single NDJSON line to w.
func writeJSONL(w io.Writer, v any) {
	data, err := json.Marshal(v)
	if err != nil {
		return
	}
	_, _ = w.Write(append(data, '\n'))
}