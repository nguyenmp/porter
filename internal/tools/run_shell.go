package tools

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os/exec"
	"strings"
)

// runShell runs a shell command and returns a stream of its combined
// stdout/stderr, with a trailing exit-status line appended once it finishes.
func runShell(ctx context.Context, args []byte) (io.ReadCloser, error) {
	var in struct {
		Command string `json:"command"`
	}
	if err := json.Unmarshal(args, &in); err != nil {
		return nil, fmt.Errorf("parse shell arguments: %w", err)
	}
	if strings.TrimSpace(in.Command) == "" {
		return nil, errors.New("shell command is empty")
	}

	cmd := exec.CommandContext(ctx, "sh", "-c", in.Command)
	pr, pw := io.Pipe()
	cmd.Stdout = pw
	cmd.Stderr = pw
	if err := cmd.Start(); err != nil {
		_ = pr.CloseWithError(err)
		_ = pw.Close()
		return nil, fmt.Errorf("start shell command: %w", err)
	}

	go func() {
		code := 0
		if err := cmd.Wait(); err != nil {
			if ee, ok := err.(*exec.ExitError); ok {
				code = ee.ExitCode()
			} else {
				_ = pw.CloseWithError(err)
				return
			}
		}
		_, _ = fmt.Fprintf(pw, "\nexit code: %d\n", code)
		_ = pw.Close()
	}()

	return pr, nil
}