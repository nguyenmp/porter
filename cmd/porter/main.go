package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"time"

	"porter/internal/api"
	"porter/internal/client"
	"porter/internal/codec"
	"porter/internal/config"
	"porter/internal/repl"
	"porter/internal/server"
)

func main() {
	if err := runCLI(os.Args[1:], os.Stdout, os.Stdin); err != nil {
		fmt.Fprintln(os.Stderr, "porter:", err)
		os.Exit(1)
	}
}

// runCLI dispatches to the server, the interactive REPL (when stdin is a TTY
// and no prompt is given), or the one-shot JSONL path.
func runCLI(args []string, stdout io.Writer, stdin io.Reader) error {
	if len(args) > 0 && args[0] == "server" {
		cfg := config.Env()
		if err := cfg.Validate(); err != nil {
			return err
		}
		return server.Serve(cfg)
	}

	cfg := config.ClientEnv()

	if len(args) == 0 && isTerminal(stdin) {
		return repl.Run(context.Background(), cfg, stdin, stdout, os.Stderr)
	}

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

// run sends a one-shot prompt to the server and streams the model's events to
// out as structured JSONL. Connect/timing progress goes to stderr.
func run(ctx context.Context, cfg config.ClientConfig, prompt string, out io.Writer) error {
	c := client.New(cfg.ServerURL)
	info, err := c.Create(ctx)
	if err != nil {
		return err
	}
	fmt.Fprintf(os.Stderr, "porter: session %s\n", info.ID)
	start := time.Now()
	if err := c.Append(ctx, info.ID, prompt); err != nil {
		return err
	}
	enc := codec.NewEncoder(out)
	err = c.Subscribe(ctx, info.ID, info.Seq, func(env api.Envelope) {
		switch env.Kind {
		case api.KindLLM:
			if env.Event != nil {
				_ = enc.Write(*env.Event)
			}
		case api.KindToolResult, api.KindToolResultDelta:
			// Live deltas are forwarded as they arrive so the CLI streams tool
			// output in real time; the terminal KindToolResult still carries the
			// full record.
			data, _ := json.Marshal(env)
			_, _ = out.Write(append(data, '\n'))
		}
	}, func(env api.Envelope) bool { return env.Kind == api.KindTurnDone })
	if err != nil {
		return err
	}
	fmt.Fprintf(os.Stderr, "porter: stream complete in %s\n", time.Since(start).Round(time.Millisecond))
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