package llm

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
		params["required"] = appendRequired(params["required"], ArgPorterDescription, ArgPorterTimeout)
	}
	return defs
}

// appendRequired appends each name that is not already in the schema's
// required list. Every schema is hand-built with []string, but []any is
// accepted defensively for a schema that came from JSON.
func appendRequired(required any, names ...string) []string {
	var out []string
	switch r := required.(type) {
	case []string:
		out = append(out, r...)
	case []any:
		for _, n := range r {
			if s, ok := n.(string); ok {
				out = append(out, s)
			}
		}
	}
	for _, name := range names {
		dup := false
		for _, have := range out {
			if have == name {
				dup = true
				break
			}
		}
		if !dup {
			out = append(out, name)
		}
	}
	return out
}
