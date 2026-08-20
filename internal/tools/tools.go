// Package tools defines the tools an agent may call and the providers that
// execute them. For now there is a single `shell` tool: running a command in
// the working directory, which subsumes file edits and network calls. Tool
// execution is a stream so it can happen here, on a connected client, or on a
// remote host without changing the agent loop.
package tools

import (
	"context"
	"fmt"
	"io"

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

// Provider runs the tools an agent uses. It declares the tool schemas the model
// sees and executes the calls the model requests. Run returns a stream of the
// tool's output (combined stdout/stderr plus a trailing exit-status line); the
// agent reads it to completion. The agent depends on this interface, not a
// concrete implementation, so execution can live on the server, a connected
// client, or a remote host without changing the agent.
type Provider interface {
	// Defs returns the tools an agent may call.
	Defs() []llm.Tool

	// Run executes the named tool with raw JSON arguments and returns a stream
	// of its output. It returns an error only if the tool cannot be started;
	// command failures surface in the stream.
	Run(ctx context.Context, name string, args []byte) (io.ReadCloser, error)
}

// Dispatcher runs tools locally, on the process that calls it.
type Dispatcher struct{}

// NewDispatcher returns a dispatcher that can run the available tools.
func NewDispatcher() *Dispatcher {
	return &Dispatcher{}
}

// Defs returns the tool schemas a Dispatcher can execute.
func (d *Dispatcher) Defs() []llm.Tool {
	return Defs()
}

// Run executes the named tool, streaming its output.
func (d *Dispatcher) Run(ctx context.Context, name string, args []byte) (io.ReadCloser, error) {
	switch name {
	case "shell":
		return runShell(ctx, args)
	default:
		return nil, fmt.Errorf("unknown tool: %q", name)
	}
}