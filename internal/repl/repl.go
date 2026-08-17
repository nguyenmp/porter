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

	"porter/internal/codec"
	"porter/internal/config"
	"porter/internal/llm"
)

// Run drives a linear, multi-turn conversation. in supplies the user's lines,
// out receives the human-readable view (prompt + streamed reply), and jsonl
// receives the structured event stream (normally stderr).
func Run(ctx context.Context, cfg config.Config, in io.Reader, out, jsonl io.Writer) error {
	r := bufio.NewReader(in)
	client := llm.NewClient(cfg, nil)
	client.Debug = os.Stderr

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

		history = append(history, llm.ChatMessage{Role: "user", Content: text})
		reply, tokens, err := streamTurn(ctx, client, history, out, jsonl)
		if err != nil {
			return err
		}
		if reply != "" {
			history = append(history, llm.ChatMessage{Role: "assistant", Content: reply})
		}
		if tokens != nil {
			fmt.Fprintf(out, "(%d in, %d out tokens)\n", tokens.in, tokens.out)
		}
	}
}

type tokenUsage struct {
	in  int
	out int
}

// streamTurn sends history to the model, streams the reply to out while writing
// each structured event to jsonl, and returns the accumulated assistant text
// and token usage.
func streamTurn(ctx context.Context, client *llm.Client, history []llm.ChatMessage, out, jsonl io.Writer) (string, *tokenUsage, error) {
	body, err := client.Stream(ctx, history)
	if err != nil {
		return "", nil, err
	}
	defer body.Close()

	var reply strings.Builder
	var tokens tokenUsage
	r := newRenderer(out)
	dec := codec.NewDecoder(codec.NewEncoder(jsonl))
	dec.OnEvent = func(ev codec.Event) {
		switch ev.Type {
		case codec.TypeMessageDelta:
			reply.WriteString(ev.Delta)
			r.content(ev.Delta)
		case codec.TypeReasoningDelta:
			r.reasoning(ev.Reasoning)
		case codec.TypeMessage:
			r.endl()
		case codec.TypeUsage:
			tokens.in = ev.InputTokens
			tokens.out = ev.OutputTokens
		}
	}
	for line := range llm.SSELines(body) {
		done, err := dec.Process(line)
		if err != nil {
			return reply.String(), &tokens, err
		}
		if done {
			break
		}
	}
	return reply.String(), &tokens, nil
}

// renderer draws the human-readable reply to out. Reasoning is dimmed; dimming
// is only applied when out is a real terminal, so redirected output stays clean.
type renderer struct {
	out io.Writer
	dim bool
}

func newRenderer(out io.Writer) *renderer {
	return &renderer{out: out, dim: isTerminal(out)}
}

func (r *renderer) content(s string) { io.WriteString(r.out, s) }

func (r *renderer) reasoning(s string) {
	if !r.dim {
		io.WriteString(r.out, s)
		return
	}
	io.WriteString(r.out, "\x1b[2m"+s+"\x1b[0m")
}

func (r *renderer) endl() { io.WriteString(r.out, "\n") }

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
