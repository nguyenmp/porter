package llm

import (
	"bytes"
	"encoding/json"
	"sort"
)

// Porter's per-call tool contract: two required arguments that every tool an
// agent may call carries, injected into each tool's argument schema by
// AddToolContract. They exist so the user can see the semantic goal of each
// call without reading its raw arguments (porter_action_description), and so
// no call can hang the agent forever (porter_timeout_seconds, enforced by the
// agent loop as a deadline on the run).
//
// The names carry a "porter_" prefix to keep them namespaced: CallMCP
// forwards its inner args object verbatim to remote MCP tools, and a remote
// tool can legitimately document its own "description" or "timeout_seconds"
// parameter. The prefix keeps porter's copy apart from any tool's own field.
const (
	// ArgPorterDescription is the required human-readable goal of each call.
	ArgPorterDescription = "porter_action_description"
	// ArgPorterTimeout is the required run deadline, in whole seconds.
	ArgPorterTimeout = "porter_timeout_seconds"

	// MaxPorterTimeoutSeconds caps ArgPorterTimeout. The bound exists so a
	// typo (or a model that picks a huge number) cannot defeat the timeout.
	MaxPorterTimeoutSeconds = 86400
)

// AddToolContract declares the per-call contract on every tool in defs: it
// adds porter_action_description and porter_timeout_seconds to each tool's
// argument properties and marks both required. Remote MCP tools never become
// llm.Tool defs — the model reaches them through CallMCP — so this single
// pass covers every tool the model can call.
//
// The two fields are placed deliberately in the schema: the description leads
// the property list and the required list, and the timeout trails the property
// list. Models tend to write tool-call arguments in the order the schema
// lists its properties, so a call comes out shaped the way it reads — purpose
// first, deadline last — and stays readable in the transcript and the raw
// arguments.
//
// Every Defs site builds its schemas fresh per request, so mutating them in
// place here is safe: shell and the file tools (tools), the hub tools
// FindMCP/CallMCP (mcp), and recall_tool_output/spool_output (agent) all hand
// out new maps each time they are asked.
func AddToolContract(defs []Tool) []Tool {
	for i := range defs {
		params := defs[i].Function.Parameters
		props, _ := params["properties"].(map[string]any)
		if props == nil {
			continue
		}
		if _, exists := props[ArgPorterDescription]; !exists {
			props[ArgPorterDescription] = map[string]any{
				"type":        "string",
				"description": "Required. What this tool call is for, in one short sentence, so the user can follow each step without reading the arguments.",
			}
		}
		if _, exists := props[ArgPorterTimeout]; !exists {
			props[ArgPorterTimeout] = map[string]any{
				"type":        "integer",
				"minimum":     1,
				"maximum":     MaxPorterTimeoutSeconds,
				"description": "Required. How many seconds this call may run before porter stops it and reports a timeout. Match it to the work: a couple of seconds for a quick read, longer for a build, test, or command that waits.",
			}
		}
		params["required"] = contractRequired(params["required"])
		// Swap the plain map for orderedProps so the marshaled schema lists the
		// description first and the timeout last (see orderedProps).
		params["properties"] = orderedProps(props)
	}
	return defs
}

// orderedProps is a tool schema's "properties" object that marshals its keys
// in a fixed order: porter_action_description first, then the tool's own
// properties in sorted order, then porter_timeout_seconds last. A plain map
// would marshal alphabetically, dropping the description somewhere among the
// tool's own arguments. The order of the JSON the model receives matters
// because models tend to mirror it when they write a tool call's arguments;
// leading with the description opens every call with what it is for, and
// trailing with the timeout keeps the deadline out of the way.
type orderedProps map[string]any

// MarshalJSON writes the properties object in the order orderedProps
// documents. Each value marshals as itself, so nested schemas are untouched.
func (o orderedProps) MarshalJSON() ([]byte, error) {
	own := make([]string, 0, len(o))
	for name := range o {
		if name != ArgPorterDescription && name != ArgPorterTimeout {
			own = append(own, name)
		}
	}
	sort.Strings(own)

	var buf bytes.Buffer
	buf.WriteByte('{')
	first := true
	write := func(name string) error {
		value, err := json.Marshal(o[name])
		if err != nil {
			return err
		}
		if !first {
			buf.WriteByte(',')
		}
		first = false
		key, err := json.Marshal(name)
		if err != nil {
			return err
		}
		buf.Write(key)
		buf.WriteByte(':')
		buf.Write(value)
		return nil
	}
	if _, ok := o[ArgPorterDescription]; ok {
		if err := write(ArgPorterDescription); err != nil {
			return nil, err
		}
	}
	for _, name := range own {
		if err := write(name); err != nil {
			return nil, err
		}
	}
	if _, ok := o[ArgPorterTimeout]; ok {
		if err := write(ArgPorterTimeout); err != nil {
			return nil, err
		}
	}
	buf.WriteByte('}')
	return buf.Bytes(), nil
}

// contractRequired returns the schema's required list with the description
// leading and the timeout trailing, keeping the tool's own required names in
// their original order between them. Every schema is hand-built with []string,
// but []any is accepted defensively for a schema that came from JSON.
func contractRequired(required any) []string {
	var own []string
	seen := map[string]bool{}
	add := func(name string) {
		if !seen[name] {
			seen[name] = true
			own = append(own, name)
		}
	}
	switch r := required.(type) {
	case []string:
		for _, name := range r {
			add(name)
		}
	case []any:
		for _, v := range r {
			if name, ok := v.(string); ok {
				add(name)
			}
		}
	}
	out := make([]string, 0, len(own)+2)
	out = append(out, ArgPorterDescription)
	for _, name := range own {
		if name != ArgPorterDescription && name != ArgPorterTimeout {
			out = append(out, name)
		}
	}
	return append(out, ArgPorterTimeout)
}
