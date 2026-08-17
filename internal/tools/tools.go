// Package tools defines the tools an agent may call and the dispatcher that
// executes them. For now there is a single `shell` tool: running a command in
// the working directory, which subsumes file edits and network calls.
package tools

import (
	"encoding/json"
	"fmt"
	"os/exec"
	"strings"

	"porter/internal/llm"
)

// shellDef is the model-facing definition of the shell tool.
func shellDef() llm.Tool {
	return llm.Tool{
		Type: "function",
		Function: llm.Function{
			Name:        "shell",
			Description: "Run a shell command in the working directory and return its output. Use this to inspect or edit files, run programs, and make network calls.",
			Parameters: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"command": map[string]any{
						"type":        "string",
						"description": "The shell command to run, e.g. 'cat main.go' or 'git diff'.",
					},
				},
				"required": []string{"command"},
			},
		},
	}
}

// Defs returns every tool an agent may call.
func Defs() []llm.Tool {
	return []llm.Tool{shellDef()}
}

// Dispatcher maps an assistant tool call to a handler and returns its result.
type Dispatcher struct{}

// NewDispatcher returns a dispatcher that can run the available tools.
func NewDispatcher() *Dispatcher {
	return &Dispatcher{}
}

// Run executes the tool named name with the given raw JSON arguments and
// returns a human/agent-readable result string.
func (d *Dispatcher) Run(name string, args []byte) (string, error) {
	switch name {
	case "shell":
		return runShell(args)
	default:
		return "", fmt.Errorf("unknown tool: %q", name)
	}
}

// runShell parses the command argument and executes it, capturing combined
// stdout/stderr and the exit status.
func runShell(args []byte) (string, error) {
	var in struct {
		Command string `json:"command"`
	}
	if err := json.Unmarshal(args, &in); err != nil {
		return "", fmt.Errorf("parse shell arguments: %w", err)
	}
	if strings.TrimSpace(in.Command) == "" {
		return "", fmt.Errorf("shell command is empty")
	}

	cmd := exec.Command("sh", "-c", in.Command)
	out, err := cmd.CombinedOutput()
	code := 0
	if err != nil {
		if ee, ok := err.(*exec.ExitError); ok {
			code = ee.ExitCode()
		} else {
			return "", fmt.Errorf("run shell command: %w", err)
		}
	}
	var b strings.Builder
	fmt.Fprintf(&b, "exit code: %d\n", code)
	if len(out) > 0 {
		b.Write(out)
		if out[len(out)-1] != '\n' {
			b.WriteByte('\n')
		}
	}
	return b.String(), nil
}
