package main

import (
	"context"
	"fmt"
	"io"
	"os"

	"porter/internal/codec"
	"porter/internal/config"
	"porter/internal/llm"
)

func main() {
	if err := runCLI(os.Args[1:], os.Stdout, os.Stdin); err != nil {
		fmt.Fprintln(os.Stderr, "porter:", err)
		os.Exit(1)
	}
}

// runCLI wires up a single prompt against env-var configuration and streams the
// raw SSE response lines to out. Prompts come from the first argument or stdin.
func runCLI(args []string, stdout io.Writer, stdin io.Reader) error {
	cfg := config.Env()

	var prompt string
	switch {
	case len(args) > 0:
		prompt = args[0]
	case !isTerminal(stdin):
		data, err := io.ReadAll(stdin)
		if err != nil {
			return fmt.Errorf("read prompt from stdin: %w", err)
		}
		prompt = string(data)
	default:
		return fmt.Errorf("no prompt provided: pass it as an argument or pipe it via stdin")
	}
	if prompt == "" {
		return fmt.Errorf("empty prompt")
	}

	return run(context.Background(), cfg, prompt, stdout)
}

// run resolves a prompt against cfg and streams the response to out as
// structured JSONL events. It is split from runCLI so it can be tested with a
// fixed config.
func run(ctx context.Context, cfg config.Config, prompt string, out io.Writer) error {
	if err := cfg.Validate(); err != nil {
		return err
	}
	client := llm.NewClient(cfg, nil)
	body, err := client.Stream(ctx, prompt)
	if err != nil {
		return err
	}
	defer body.Close()

	dec := codec.NewDecoder(codec.NewEncoder(out))
	for line := range llm.SSELines(body) {
		done, err := dec.Process(line)
		if err != nil {
			return err
		}
		if done {
			break
		}
	}
	return nil
}

// isTerminal reports whether r is an interactive character device (a TTY),
// which means no prompt was piped in via stdin.
func isTerminal(r io.Reader) bool {
	f, ok := r.(*os.File)
	if !ok {
		return false
	}
	st, err := f.Stat()
	if err != nil {
		return false
	}
	return st.Mode()&os.ModeCharDevice != 0
}
