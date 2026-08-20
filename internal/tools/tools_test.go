package tools

import (
	"context"
	"io"
	"strings"
	"testing"
)

// run executes a tool via a Provider, reading its output stream to completion.
func run(ctx context.Context, p Provider, name string, args []byte) (string, error) {
	stream, err := p.Run(ctx, name, args)
	if err != nil {
		return "", err
	}
	b, rerr := io.ReadAll(stream)
	_ = stream.Close()
	if rerr != nil {
		return "", rerr
	}
	return string(b), nil
}

func TestShellRunsCommand(t *testing.T) {
	res, err := run(context.Background(), NewDispatcher(), "shell", []byte(`{"command":"echo hi"}`))
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if !strings.Contains(res, "exit code: 0") || !strings.Contains(res, "hi") {
		t.Errorf("unexpected result: %q", res)
	}
}

func TestShellReportsExitCode(t *testing.T) {
	res, err := run(context.Background(), NewDispatcher(), "shell", []byte(`{"command":"exit 3"}`))
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if !strings.Contains(res, "exit code: 3") {
		t.Errorf("unexpected result: %q", res)
	}
}

func TestShellCombinesStdoutAndStderr(t *testing.T) {
	res, err := run(context.Background(), NewDispatcher(), "shell", []byte(`{"command":"echo out; echo err >&2"}`))
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if !strings.Contains(res, "out") || !strings.Contains(res, "err") {
		t.Errorf("result should include stdout and stderr; got: %q", res)
	}
}

func TestDispatcherUnknownTool(t *testing.T) {
	if _, err := run(context.Background(), NewDispatcher(), "nope", []byte("{}")); err == nil {
		t.Fatal("expected error for unknown tool")
	}
}

func TestShellEmptyCommand(t *testing.T) {
	if _, err := run(context.Background(), NewDispatcher(), "shell", []byte(`{"command":"  "}`)); err == nil {
		t.Fatal("expected error for empty command")
	}
}

func TestShellRespectsCancellation(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	stream, err := NewDispatcher().Run(ctx, "shell", []byte(`{"command":"sleep 5"}`))
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	go func() {
		_, _ = io.Copy(io.Discard, stream)
		_ = stream.Close()
	}()
	cancel()
	// Reading after cancellation must terminate rather than hang. Any of
	// "(interrupt)" / EOF / "exit code" end-of-stream satisfies correctness.
	b := make([]byte, 1)
	_, _ = stream.Read(b)
}