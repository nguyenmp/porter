// Package repl is the interactive terminal shell. When stdin is a TTY, porter
// runs a REPL here instead of the one-shot JSONL path. Stdout shows the
// conversation, and the structured JSONL event stream goes to stderr. The REPL
// is a thin client: it keeps the conversation history for the next turn, sends
// the full history to the server, and relays events back.
package repl

import (
	"bufio"
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"strings"

	"porter/internal/agent"
	"porter/internal/codec"
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

	// One sink sends every event to both the JSONL stream and the
	// human-readable view.
	emit := func(ev codec.Event) {
		agent.EncodeJSON(jsonl)(ev)
		agent.Render(out, agent.IsTerminal(out))(ev)
	}

	r := bufio.NewReader(in)
	var history []llm.ChatMessage
	for {
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

		history = append(history, llm.UserMessage(text))
		comp, err := c.Stream(ctx, history, emit)
		if err != nil {
			return err
		}
		history = comp.History
		if comp.Input > 0 || comp.Output > 0 {
			fmt.Fprintf(out, "(%d in, %d out tokens)\n", comp.Input, comp.Output)
		}
	}
}