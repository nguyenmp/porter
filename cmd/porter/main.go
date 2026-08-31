package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"time"

	"porter/internal/api"
	"porter/internal/client"
	"porter/internal/codec"
	"porter/internal/config"
	"porter/internal/hostagent"
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
// and no prompt is given), or the one-shot JSONL path. Flags (e.g. --session)
// must precede the subcommand or prompt.
func runCLI(args []string, stdout io.Writer, stdin io.Reader) error {
	fs := flag.NewFlagSet("porter", flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	session := fs.String("session", "", "resume an existing session id instead of creating a new one")
	if err := fs.Parse(args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			fmt.Fprintln(stdout, "usage: porter [flags] [prompt|server|host]")
			fs.SetOutput(stdout)
			fs.PrintDefaults()
			return nil
		}
		return err
	}
	args = fs.Args()

	if len(args) > 0 && args[0] == "server" {
		cfg := config.Env()
		if err := cfg.Validate(); err != nil {
			return err
		}
		return server.Serve(cfg)
	}

	// `porter host` runs the persistent execution host agent: a long-running
	// process on a machine (e.g. the laptop) that connects to the server once
	// and provisions execution contexts for new sessions on demand. The web
	// UI's "new chat on" picker lists connected hosts; choosing one runs that
	// chat's agent loop on the host's machine.
	if len(args) > 0 && args[0] == "host" {
		cfg := config.ClientEnv()
		return hostagent.Run(context.Background(), cfg)
	}

	cfg := config.ClientEnv()
	if *session != "" {
		cfg.Session = *session
	}

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
// out as structured JSONL. Connect/timing progress goes to stderr. When
// cfg.Session is set it appends to that existing session instead of creating a
// new one.
func run(ctx context.Context, cfg config.ClientConfig, prompt string, out io.Writer) error {
	c := client.New(cfg.ServerURL, client.BasicAuth{Username: cfg.Username, Password: cfg.Password})
	var (
		id  string
		seq uint64
		err error
	)
	if cfg.Session != "" {
		var h api.SessionHistory
		h, err = c.History(ctx, cfg.Session)
		if err != nil {
			return fmt.Errorf("load session %s: %w", cfg.Session, err)
		}
		id, seq = cfg.Session, h.Seq
	} else {
		var info api.SessionInfo
		info, err = c.Create(ctx)
		if err != nil {
			return err
		}
		id, seq = info.ID, info.Seq
	}
	fmt.Fprintf(os.Stderr, "porter: session %s\n", id)
	start := time.Now()
	if err := c.Append(ctx, id, prompt); err != nil {
		return err
	}
	enc := codec.NewEncoder(out)
	var turnErr string
	err = c.Subscribe(ctx, id, seq, func(env api.Envelope) {
		switch env.Kind {
		case api.KindLLM:
			if env.Event != nil {
				_ = enc.Write(*env.Event)
			}
		case api.KindToolResult, api.KindToolResultDelta, api.KindToolStarted:
			// Live deltas and the start marker are forwarded as they arrive so
			// the CLI streams tool output in real time; the terminal
			// KindToolResult still carries the full record.
			data, _ := json.Marshal(env)
			_, _ = out.Write(append(data, '\n'))
		case api.KindTurnDone:
			// A turn that failed (the provider returned an error, e.g. a 400 or
			// a rate limit) is reported to the caller rather than ending the
			// stream looking like success.
			turnErr = env.Error
		}
	}, func(env api.Envelope) bool { return env.Kind == api.KindTurnDone })
	if err != nil {
		return err
	}
	if turnErr != "" {
		return fmt.Errorf("turn failed: %s", turnErr)
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
