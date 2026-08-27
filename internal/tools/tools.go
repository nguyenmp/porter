// Package tools defines the tools an agent may call and the providers that
// execute them. For now there are two: `shell` (running a command in the
// working directory, which subsumes file edits and network calls) and
// `load_skill` (loading a discovered skill's full body). Tool execution is a
// stream so it can happen here, on a connected client, or on a remote host
// without changing the agent loop.
package tools

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"sort"
	"strings"
	"sync"

	"porter/internal/api"
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

// loadSkillDef is the model-facing definition of the load_skill tool. Its
// description carries the metadata for every discovered skill (deduplicated),
// so the model sees what is available without loading anything.
func loadSkillDef(skills []api.Skill) llm.Tool {
	var b strings.Builder
	b.WriteString("Load the full content of a skill. Available skills:\n")
	for _, s := range skills {
		fmt.Fprintf(&b, "- %s: %s\n", s.Name, s.Description)
	}
	return llm.Tool{
		Type: "function",
		Function: llm.Function{
			Name:        "load_skill",
			Description: b.String(),
			Parameters: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"name": map[string]any{
						"type":        "string",
						"description": "The name of the skill to load.",
					},
				},
				"required": []string{"name"},
			},
		},
	}
}

// Defs returns the base tool schemas an agent may call (shell only). It is
// what a provider with no discovered skills declares.
func Defs() []llm.Tool {
	return DefsForSkills(nil)
}

// DefsForSkills returns the tool schemas for a provider that can load the
// given skills: shell, plus load_skill when at least one skill is known.
func DefsForSkills(skills []api.Skill) []llm.Tool {
	out := []llm.Tool{shellDef()}
	if len(skills) > 0 {
		out = append(out, loadSkillDef(skills))
	}
	return out
}

// Provider runs the tools an agent uses. It declares the tool schemas the model
// sees, executes the calls the model requests, and reports the environment it
// runs in so the agent can inject it into the model's context. Run returns a
// stream of the tool's output (combined stdout/stderr plus a trailing
// exit-status line); the agent reads it to completion. The agent depends on
// this interface, not a concrete implementation, so execution can live on the
// server, a connected client, or a remote host without changing the agent.
type Provider interface {
	// Defs returns the tools an agent may call.
	Defs() []llm.Tool

	// Run executes the named tool with raw JSON arguments and returns a stream
	// of its output. It returns an error only if the tool cannot be started;
	// command failures surface in the stream.
	Run(ctx context.Context, name string, args []byte) (io.ReadCloser, error)

	// Environment returns the execution environment context to inject into the
	// model's prompt as a system message (system, working directory, files,
	// skills), or "" when the provider has none to report. It is read fresh per
	// request so a provider swap is reflected immediately.
	Environment() string
}

// Dispatcher runs tools locally, on the process that calls it. It can be given
// the discovered skills so it can serve load_skill calls by reading each
// skill's SKILL.md; with no skills it exposes only shell.
type Dispatcher struct {
	mu     sync.RWMutex
	skills map[string]api.Skill
}

// NewDispatcher returns a dispatcher with no skills (shell only).
func NewDispatcher() *Dispatcher {
	return &Dispatcher{}
}

// NewDispatcherWithSkills returns a dispatcher that can load the given skills
// (and therefore exposes the load_skill tool).
func NewDispatcherWithSkills(skills []api.Skill) *Dispatcher {
	d := NewDispatcher()
	d.SetSkills(skills)
	return d
}

// SetSkills replaces the skills the dispatcher can load.
func (d *Dispatcher) SetSkills(skills []api.Skill) {
	m := make(map[string]api.Skill, len(skills))
	for _, s := range skills {
		m[s.Name] = s
	}
	d.mu.Lock()
	d.skills = m
	d.mu.Unlock()
}

// Skills returns the dispatcher's skills, ordered by name.
func (d *Dispatcher) Skills() []api.Skill {
	d.mu.RLock()
	out := make([]api.Skill, 0, len(d.skills))
	for _, s := range d.skills {
		out = append(out, s)
	}
	d.mu.RUnlock()
	sort.Slice(out, func(i, j int) bool { return out[i].Name < out[j].Name })
	return out
}

// Defs returns the tool schemas a Dispatcher can execute.
func (d *Dispatcher) Defs() []llm.Tool {
	return DefsForSkills(d.Skills())
}

// Environment returns the dispatcher's execution environment context. A local
// dispatcher reports none: it runs on the process's own machine, and discovery
// is the connected client's job.
func (d *Dispatcher) Environment() string {
	return ""
}

// Run executes the named tool, streaming its output.
func (d *Dispatcher) Run(ctx context.Context, name string, args []byte) (io.ReadCloser, error) {
	switch name {
	case "shell":
		return runShell(ctx, args)
	case "load_skill":
		return d.runLoadSkill(args)
	default:
		return nil, fmt.Errorf("unknown tool: %q", name)
	}
}

// runLoadSkill reads a discovered skill's SKILL.md and returns its body as a
// stream (with the conventional trailing exit-status line so the agent's tool
// handling is uniform). It returns an error only when the skill is unknown or
// unreadable.
func (d *Dispatcher) runLoadSkill(args []byte) (io.ReadCloser, error) {
	var in struct {
		Name string `json:"name"`
	}
	if err := json.Unmarshal(args, &in); err != nil {
		return nil, fmt.Errorf("parse load_skill arguments: %w", err)
	}
	d.mu.RLock()
	skill, ok := d.skills[in.Name]
	d.mu.RUnlock()
	if !ok {
		return nil, fmt.Errorf("unknown skill: %q", in.Name)
	}
	data, err := os.ReadFile(skill.Path)
	if err != nil {
		return nil, fmt.Errorf("read skill %q: %w", in.Name, err)
	}
	return &stringStream{strings.NewReader(string(data) + "\nexit code: 0\n")}, nil
}

// stringStream is an io.ReadCloser over an in-memory string.
type stringStream struct {
	*strings.Reader
}

func (s stringStream) Close() error { return nil }
