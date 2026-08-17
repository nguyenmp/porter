// Package repl is the interactive, terminal-driven conversation shell. When
// stdin is a TTY, porter runs a REPL here instead of the one-shot JSONL path:
// stdout shows the conversation, and the structured JSONL event stream goes to
// stderr.
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
	"porter/internal/config"
	"porter/internal/llm"
	"porter/internal/tools"
)

// Run drives a linear, multi-turn conversation. in supplies the user's lines,
// out receives the human-readable view (prompt + streamed reply), and jsonl
// receives the structured event stream (normally stderr). When cfg.LogFile is
// set, the event stream and progress lines go to that file instead, so an
// interactive container stays quiet.
func Run(ctx context.Context, cfg config.Config, in io.Reader, out, jsonl io.Writer) error {
	if cfg.LogFile != "" {
		f, err := os.OpenFile(cfg.LogFile, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
		if err != nil {
			return fmt.Errorf("open log file: %w", err)
		}
		defer f.Close()
		jsonl = f
	}

	r := bufio.NewReader(in)
	client := llm.NewClient(cfg, nil)
	client.Debug = os.Stderr
	if cfg.LogFile != "" {
		client.Debug = jsonl
	}
	js := tools.NewDispatcher()

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
		res, err := agent.RunTurn(ctx, client, history, js, out, jsonl)
		if err != nil {
			return err
		}
		history = res.History
		if res.Usage.Input > 0 || res.Usage.Output > 0 {
			fmt.Fprintf(out, "(%d in, %d out tokens)\n", res.Usage.Input, res.Usage.Output)
		}
	}
}