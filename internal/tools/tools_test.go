package tools

import (
	"strings"
	"testing"
)

func TestShellRunsCommand(t *testing.T) {
	res, err := NewDispatcher().Run("shell", []byte(`{"command":"echo hi"}`))
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if !strings.Contains(res, "exit code: 0") || !strings.Contains(res, "hi") {
		t.Errorf("unexpected result: %q", res)
	}
}

func TestShellReportsExitCode(t *testing.T) {
	res, err := NewDispatcher().Run("shell", []byte(`{"command":"exit 3"}`))
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if !strings.Contains(res, "exit code: 3") {
		t.Errorf("unexpected result: %q", res)
	}
}

func TestShellCombinesStdoutAndStderr(t *testing.T) {
	res, err := NewDispatcher().Run("shell", []byte(`{"command":"echo out; echo err >&2"}`))
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if !strings.Contains(res, "out") || !strings.Contains(res, "err") {
		t.Errorf("result should include stdout and stderr; got: %q", res)
	}
}

func TestDispatcherUnknownTool(t *testing.T) {
	if _, err := NewDispatcher().Run("nope", []byte("{}")); err == nil {
		t.Fatal("expected error for unknown tool")
	}
}

func TestShellEmptyCommand(t *testing.T) {
	if _, err := NewDispatcher().Run("shell", []byte(`{"command":"  "}`)); err == nil {
		t.Fatal("expected error for empty command")
	}
}
