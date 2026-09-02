package mcp

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"sort"
	"strings"
	"sync/atomic"

	"porter/internal/api"
	"porter/internal/llm"
	"porter/internal/tools"
)

// The two hub tools are the model's entire MCP surface: any number of MCP
// servers and their tools add exactly these two tools to the model's context.
const (
	FindTool = "FindMCP"
	CallTool = "CallMCP"
)

// rpcID is the JSON-RPC id counter for tools/call requests.
var rpcID atomic.Int64

func nextID() int64 { return rpcID.Add(1) }

const (
	// findSnippetMax truncates each tool description in snippet mode.
	findSnippetMax = 140
	// findFullMax bounds how many tools show their full schema per FindMCP
	// call, so full=true cannot blow up the model's context.
	findFullMax = 10
	// findSnippetMaxTools bounds how many tools are listed in snippet mode.
	findSnippetMaxTools = 60
)

// findDef builds the model-facing FindMCP definition. Its description is
// rebuilt per request from the current registry plus any remote servers
// (reported by the active execution provider), so the model always sees the
// server list (name, description, tool count, load status) without calling
// anything.
func (h *Hub) findDef(extra []api.MCPServer) llm.Tool {
	var b strings.Builder
	b.WriteString("Discover MCP (Model Context Protocol) servers and their tools, then call a tool with CallMCP. ")
	h.mu.RLock()
	if len(h.servers) == 0 && len(extra) == 0 {
		b.WriteString("No MCP servers are configured.")
	} else {
		b.WriteString("Configured servers:\n")
		for _, name := range h.order {
			s := h.servers[name]
			status, errMsg := s.Status()
			switch status {
			case "ok":
				fmt.Fprintf(&b, "- %s (%d tools): %s\n", name, len(s.Tools()), s.Description)
			case "error":
				fmt.Fprintf(&b, "- %s (error: %s): %s\n", name, errMsg, s.Description)
			default:
				fmt.Fprintf(&b, "- %s (pending): %s\n", name, s.Description)
			}
		}
		for _, ms := range extra {
			status := ms.Status
			if status == "" {
				status = "ok"
			}
			switch status {
			case "ok":
				fmt.Fprintf(&b, "- %s (%d tools): %s", ms.Name, len(ms.Tools), ms.Description)
			case "error":
				fmt.Fprintf(&b, "- %s (error: %s): %s", ms.Name, ms.Error, ms.Description)
			default:
				fmt.Fprintf(&b, "- %s (pending): %s", ms.Name, ms.Description)
			}
			if ms.Host != "" {
				fmt.Fprintf(&b, " (hosted on %s)", ms.Host)
			}
			b.WriteString("\n")
		}
		b.WriteString("Call with server_name to list one server's tools (omitting it searches all servers) and an optional query substring to filter tool names and descriptions. Pass full=true for full descriptions and input schemas — use sparingly, it is verbose.")
	}
	h.mu.RUnlock()
	return llm.Tool{
		Type: "function",
		Function: llm.Function{
			Name:        FindTool,
			Description: b.String(),
			Parameters: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"server_name": map[string]any{"type": "string", "description": "Optional: only list tools from this server. Omit to search all servers."},
					"query":       map[string]any{"type": "string", "description": "Optional: substring filter on tool names and descriptions."},
					"full":        map[string]any{"type": "boolean", "description": "Return full descriptions and input schemas instead of one-line snippets. Verbose; use sparingly."},
				},
			},
		},
	}
}

// callDef is the model-facing CallMCP definition.
func callDef() llm.Tool {
	return llm.Tool{
		Type: "function",
		Function: llm.Function{
			Name:        CallTool,
			Description: "Call a tool on an MCP server. The tool must exist on the server; discover it with FindMCP, which also shows its input schema when asked. The result is the server's text response; a failing call is reported as an error in the result.",
			Parameters: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"server_name": map[string]any{"type": "string", "description": "The MCP server to call (see FindMCP for configured servers)."},
					"tool_name":   map[string]any{"type": "string", "description": "The tool to call on that server."},
					"args":        map[string]any{"type": "object", "description": "The tool's arguments, as a JSON object matching its input schema (see FindMCP with full=true). Example: {\"urls\": [\"https://example.com\"]} for exa-web_fetch_exa. Always pass this; if the tool takes no arguments, pass {}."},
				},
				"required": []string{"server_name", "tool_name", "args"},
			},
		},
	}
}

// Defs returns the hub's model-facing tool definitions. The Hub is not a full
// tools.Provider (it has no execution environment); Composite merges it with
// one.
func (h *Hub) Defs() []llm.Tool {
	return []llm.Tool{h.findDef(nil), callDef()}
}

// Run executes a hub tool against the hub's own servers. It returns an error
// only when the call cannot be started — unknown tool, unknown server,
// malformed arguments (mirroring how load_skill rejects an unknown skill).
// Remote failures — HTTP or JSON-RPC errors, timeouts, the server reporting
// isError — surface in the streamed result text, mirroring how the shell tool
// reports command failures, so the model sees the failure and can react.
func (h *Hub) Run(ctx context.Context, name string, args []byte) (io.ReadCloser, error) {
	switch name {
	case FindTool:
		return h.runFind(ctx, args, nil)
	case CallTool:
		return h.runCall(ctx, args)
	default:
		return nil, fmt.Errorf("unknown MCP tool: %q", name)
	}
}

// runFind renders the FindMCP listing for the hub's own servers plus extra
// (host-provided) servers, honoring server_name/query/full across both.
func (h *Hub) runFind(ctx context.Context, args []byte, extra []api.MCPServer) (io.ReadCloser, error) {
	// A server that failed at load (e.g. an OAuth server not yet logged in)
	// gets one lazy retry here, so `porter mcp login` after startup takes
	// effect without a restart.
	h.refreshErrored(ctx)
	var in struct {
		ServerName string `json:"server_name"`
		Query      string `json:"query"`
		Full       bool   `json:"full"`
	}
	if err := json.Unmarshal(args, &in); err != nil {
		return nil, fmt.Errorf("parse FindMCP arguments: %w", err)
	}

	h.mu.RLock()
	servers := make([]*Server, 0, len(h.servers))
	for _, name := range h.order {
		if in.ServerName != "" && name != in.ServerName {
			continue
		}
		servers = append(servers, h.servers[name])
	}
	h.mu.RUnlock()

	remotes := extra[:0:0] // copy so filtering below does not mutate the caller's slice
	for _, ms := range extra {
		if in.ServerName != "" && ms.Name != in.ServerName {
			continue
		}
		remotes = append(remotes, ms)
	}

	if len(servers) == 0 && len(remotes) == 0 {
		if in.ServerName != "" {
			return nil, fmt.Errorf("unknown MCP server %q (see the FindMCP description for configured servers)", in.ServerName)
		}
		return newResultStream("No MCP servers are configured."), nil
	}

	query := strings.ToLower(in.Query)
	var b strings.Builder
	for _, s := range servers {
		status, errMsg := s.Status()
		switch status {
		case "ok":
			fmt.Fprintf(&b, "server %s (%d tools): %s\n", s.Name, len(s.Tools()), s.Description)
		case "error":
			fmt.Fprintf(&b, "server %s (error: %s): %s\n", s.Name, errMsg, s.Description)
		default:
			fmt.Fprintf(&b, "server %s (pending): %s\n", s.Name, s.Description)
		}
		matched := 0
		for _, t := range s.Tools() {
			if query != "" && !strings.Contains(strings.ToLower(t.Name), query) && !strings.Contains(strings.ToLower(t.Description), query) {
				continue
			}
			matched++
			switch {
			case in.Full && matched <= findFullMax:
				fmt.Fprintf(&b, "  %s: %s\n", t.Name, t.Description)
				if len(t.InputSchema) > 0 {
					schema, _ := json.Marshal(t.InputSchema)
					fmt.Fprintf(&b, "    inputSchema: %s\n", schema)
				}
			case !in.Full && matched <= findSnippetMaxTools:
				fmt.Fprintf(&b, "  %s: %s", t.Name, snippet(t.Description))
				if hint := argsHint(t.InputSchema); hint != "" {
					fmt.Fprintf(&b, " [%s]", hint)
				}
				fmt.Fprintf(&b, "\n")
			}
		}
		switch {
		case matched == 0:
			b.WriteString("  (no matching tools)\n")
		case in.Full && matched > findFullMax:
			fmt.Fprintf(&b, "  ... and %d more (narrow the search or query for a specific tool)\n", matched-findFullMax)
		case !in.Full && matched > findSnippetMaxTools:
			fmt.Fprintf(&b, "  ... and %d more (narrow the search)\n", matched-findSnippetMaxTools)
		}
	}
	for _, ms := range remotes {
		status := ms.Status
		if status == "" {
			status = "ok"
		}
		switch status {
		case "ok":
			fmt.Fprintf(&b, "server %s (%d tools): %s", ms.Name, len(ms.Tools), ms.Description)
		case "error":
			fmt.Fprintf(&b, "server %s (error: %s): %s", ms.Name, ms.Error, ms.Description)
		default:
			fmt.Fprintf(&b, "server %s (pending): %s", ms.Name, ms.Description)
		}
		if ms.Host != "" {
			fmt.Fprintf(&b, " (hosted on %s)", ms.Host)
		}
		b.WriteString("\n")
		matched := 0
		for _, t := range ms.Tools {
			if query != "" && !strings.Contains(strings.ToLower(t.Name), query) && !strings.Contains(strings.ToLower(t.Description), query) {
				continue
			}
			matched++
			switch {
			case in.Full && matched <= findFullMax:
				fmt.Fprintf(&b, "  %s: %s\n", t.Name, t.Description)
				if len(t.InputSchema) > 0 {
					schema, _ := json.Marshal(t.InputSchema)
					fmt.Fprintf(&b, "    inputSchema: %s\n", schema)
				}
			case !in.Full && matched <= findSnippetMaxTools:
				fmt.Fprintf(&b, "  %s: %s", t.Name, snippet(t.Description))
				if hint := argsHint(t.InputSchema); hint != "" {
					fmt.Fprintf(&b, " [%s]", hint)
				}
				fmt.Fprintf(&b, "\n")
			}
		}
		switch {
		case matched == 0:
			b.WriteString("  (no matching tools)\n")
		case in.Full && matched > findFullMax:
			fmt.Fprintf(&b, "  ... and %d more (narrow the search or query for a specific tool)\n", matched-findFullMax)
		case !in.Full && matched > findSnippetMaxTools:
			fmt.Fprintf(&b, "  ... and %d more (narrow the search)\n", matched-findSnippetMaxTools)
		}
	}
	return newResultStream(b.String()), nil
}

// snippet flattens a tool description to a short one-line form.
func snippet(desc string) string {
	desc = strings.Join(strings.Fields(desc), " ")
	if r := []rune(desc); len(r) > findSnippetMax {
		return string(r[:findSnippetMax]) + "…"
	}
	return desc
}

// requiredArgs returns the top-level required argument names of a JSON Schema
// input schema: the "required" array of strings (when present). Schemas built
// in tests may carry a []string while decoded server schemas hold []any, so
// both shapes are accepted.
func requiredArgs(schema map[string]any) []string {
	switch req := schema["required"].(type) {
	case []string:
		return req
	case []any:
		out := make([]string, 0, len(req))
		for _, v := range req {
			if s, ok := v.(string); ok {
				out = append(out, s)
			}
		}
		return out
	}
	return nil
}

// missingArgs returns the tool's required argument names that are absent from
// the caller's args (nil args counts as absent for everything).
func missingArgs(schema map[string]any, args map[string]any) []string {
	var missing []string
	for _, name := range requiredArgs(schema) {
		if _, ok := args[name]; !ok {
			missing = append(missing, name)
		}
	}
	return missing
}

// argsHint renders a tool's argument names for one FindMCP snippet line:
// required names first, then optional names (properties the schema does not
// require), so the model sees the full surface without a full-schema lookup.
// Both halves are sorted for determinism; the optional list caps at
// snippetOptionalMax to keep wide schemas compact.
const snippetOptionalMax = 6

func argsHint(schema map[string]any) string {
	props, _ := schema["properties"].(map[string]any)
	if len(props) == 0 {
		return ""
	}
	req := make(map[string]bool, len(props))
	for _, name := range requiredArgs(schema) {
		req[name] = true
	}
	required, optional := make([]string, 0, len(req)), make([]string, 0, len(props)-len(req))
	for name := range props {
		if req[name] {
			required = append(required, name)
		} else {
			optional = append(optional, name)
		}
	}
	sort.Strings(required)
	sort.Strings(optional)
	var parts []string
	if len(required) > 0 {
		parts = append(parts, "requires: "+strings.Join(required, ", "))
	}
	if len(optional) > 0 {
		if n := len(optional) - snippetOptionalMax; n > 0 {
			parts = append(parts, fmt.Sprintf("optional: %s, +%d more", strings.Join(optional[:snippetOptionalMax], ", "), n))
		} else {
			parts = append(parts, "optional: "+strings.Join(optional, ", "))
		}
	}
	return strings.Join(parts, "; ")
}

func (h *Hub) runCall(ctx context.Context, args []byte) (io.ReadCloser, error) {
	h.refreshErrored(ctx)
	var in struct {
		ServerName string          `json:"server_name"`
		ToolName   string          `json:"tool_name"`
		Args       json.RawMessage `json:"args"`
	}
	if err := json.Unmarshal(args, &in); err != nil {
		return nil, fmt.Errorf("parse CallMCP arguments: %w", err)
	}
	if in.ServerName == "" {
		return nil, errors.New("CallMCP: server_name is required")
	}
	if in.ToolName == "" {
		return nil, errors.New("CallMCP: tool_name is required")
	}
	s := h.Server(in.ServerName)
	if s == nil {
		return nil, fmt.Errorf("unknown MCP server %q (see the FindMCP description for configured servers)", in.ServerName)
	}

	// Models sometimes pass args as a JSON string; unwrap it before use.
	raw := bytes.TrimSpace(in.Args)
	// argsPresent records whether the caller sent the "args" member at all
	// ("" and {} both count as present-but-empty). It is judged on the raw
	// value before the string-unwrap below, because args:"" unwraps to an
	// empty buffer that would otherwise read as "omitted". The pre-flight
	// error distinguishes omitted from empty: the two failures need different
	// fixes, and models repeatedly send one when they meant the other.
	argsPresent := len(raw) > 0 && string(raw) != "null"
	if len(raw) > 0 && raw[0] == '"' {
		var s string
		if err := json.Unmarshal(raw, &s); err == nil {
			raw = []byte(s)
		}
	}
	var argMap map[string]any
	if len(raw) > 0 && string(raw) != "null" {
		if err := json.Unmarshal(raw, &argMap); err != nil {
			return nil, fmt.Errorf("CallMCP: args must be a JSON object: %w", err)
		}
	}
	// Pre-flight: fail locally, before the network round-trip, when the args
	// omit a field the tool's schema requires. The server would reject the
	// call with a generic -32602, but pointing at the missing names (and the
	// tool's schema) gives the model an actionable fix for the next attempt.
	var inputSchema map[string]any
	if t, ok := s.Tool(in.ToolName); ok {
		inputSchema = t.InputSchema
		if missing := missingArgs(inputSchema, argMap); len(missing) > 0 {
			msg := fmt.Sprintf("CallMCP %s: missing required arguments: %s (tool inputSchema: %s; use FindMCP full=true to see it)", in.ToolName, strings.Join(missing, ", "), schemaJSON(inputSchema))
			if argsPresent && len(argMap) == 0 {
				msg += "\nnote: \"args\" was sent present but empty — the required fields must go inside \"args\" (the object), not omitted or as an empty string/object."
			}
			return nil, errors.New(msg)
		}
	}
	params := map[string]any{"name": in.ToolName}
	if len(argMap) > 0 {
		params["arguments"] = argMap
	}

	result, _, err := s.call(ctx, h.client, nextID(), "tools/call", params)
	if err != nil {
		// The call reached the server but failed (HTTP/JSON-RPC/timeout):
		// report as result content, like a command failure, so the model can
		// react (retry, pick another tool). A -32602 invalid-params rejection
		// that slipped past pre-flight gets the tool's schema appended, so
		// the model can see exactly what shape the args must take.
		return newResultStream("error: " + describeCallError(err, inputSchema, argMap)), nil
	}
	var cr callResult
	if err := json.Unmarshal(result, &cr); err != nil {
		return nil, fmt.Errorf("decode tools/call result: %w", err)
	}
	text := flattenContent(cr.Content)
	if cr.IsError {
		text = "error: " + text
	}
	if strings.TrimSpace(text) == "" {
		text = "(no content)"
	}
	return newResultStream(text), nil
}

// schemaJSON renders a tool input schema as compact JSON for error messages
// ("" when there is none).
func schemaJSON(schema map[string]any) string {
	if len(schema) == 0 {
		return "{}"
	}
	b, err := json.Marshal(schema)
	if err != nil {
		return "{}"
	}
	return string(b)
}

// describeCallError renders a failed tools/call for the model. For an
// invalid-params rejection (-32602) it appends the tool's input schema and a
// note when the call carried no arguments, so a caller-side contract break is
// unmistakable; every other failure keeps the server's own message.
func describeCallError(err error, inputSchema map[string]any, args map[string]any) string {
	var code int
	if cerr, ok := err.(*callError); ok {
		code = cerr.rpcCode
	}
	if code != -32602 {
		return err.Error()
	}
	var b strings.Builder
	b.WriteString(err.Error())
	if len(args) == 0 {
		b.WriteString("\nnote: the call was sent with no arguments")
	}
	if len(inputSchema) > 0 {
		fmt.Fprintf(&b, "\ntool inputSchema: %s", schemaJSON(inputSchema))
	}
	return b.String()
}

// callResult is the result of tools/call.
type callResult struct {
	Content []callContent `json:"content"`
	IsError bool          `json:"isError"`
}

// callContent is one content block of a tools/call result.
type callContent struct {
	Type     string `json:"type"`
	Text     string `json:"text"`
	Data     string `json:"data"`
	MimeType string `json:"mimeType"`
}

// flattenContent renders a tools/call content block list into the plain text
// the model sees. Text blocks are joined; image and resource blocks cannot be
// passed to the model, so they become short markers.
func flattenContent(content []callContent) string {
	parts := make([]string, 0, len(content))
	for _, c := range content {
		switch c.Type {
		case "text":
			parts = append(parts, c.Text)
		case "image":
			parts = append(parts, fmt.Sprintf("[image content omitted: %d bytes, %s]", len(c.Data), c.MimeType))
		case "resource":
			parts = append(parts, fmt.Sprintf("[resource content omitted: %s]", c.MimeType))
		default:
			parts = append(parts, fmt.Sprintf("[content omitted: unknown type %q]", c.Type))
		}
	}
	return strings.Join(parts, "\n")
}

// resultStream is an in-memory io.ReadCloser over a tool result string.
type resultStream struct{ *strings.Reader }

func newResultStream(s string) io.ReadCloser { return resultStream{strings.NewReader(s)} }

func (resultStream) Close() error { return nil }

// Composite merges a session's execution provider (shell, load_skill) with
// the MCP surface (FindMCP, CallMCP) into a single tools.Provider, so the
// agent loop keeps talking to one provider. Routing by name guarantees the
// hub tools are served here, on the server — never crossing the exec channel
// to a connected execution client — except for servers the client itself
// hosts: a CallMCP for a server in Remote is routed down the exec channel and
// executed by the provider's own local hub, so those credentials never leave
// the provider's machine. Each side's credentials stay on the side that owns
// them.
type Composite struct {
	// Exec is the session's execution provider (local dispatcher or a remote
	// client).
	Exec tools.Provider
	// Hub serves FindMCP and CallMCP for the server's own configured servers.
	// May be nil (an empty hub is assumed).
	Hub *Hub
	// Remote is the MCP server metadata reported by the active execution
	// provider (e.g. a laptop host serving VPN-only servers). FindMCP lists
	// them alongside the hub's servers; CallMCP for one of them is routed to
	// Exec (down the exec channel). Empty when the active provider hosts no
	// MCP servers.
	Remote []api.MCPServer
}

// hub returns the composite's hub, defaulting to an empty one so a Composite
// with Remote servers but no configured hub still exposes the MCP tools.
func (c *Composite) hub() *Hub {
	if c.Hub == nil {
		return New(nil)
	}
	return c.Hub
}

// hasMCP reports whether any MCP servers are exposed: the hub's own or the
// remote provider's.
func (c *Composite) hasMCP() bool {
	return len(c.Remote) > 0 || (c.Hub != nil && len(c.Hub.Names()) > 0)
}

// Defs returns the execution provider's tools plus the MCP tools, when any
// server (hub-owned or remote) is available.
func (c *Composite) Defs() []llm.Tool {
	defs := c.Exec.Defs()
	if c.hasMCP() {
		defs = append(defs, c.hub().findDef(c.Remote), callDef())
	}
	return defs
}

// Run routes the hub tools and everything else to the execution provider.
// FindMCP is always served here (discovery is the server's job); CallMCP is
// served here for hub-owned servers and routed down the exec channel for
// remote-owned ones.
func (c *Composite) Run(ctx context.Context, name string, args []byte) (io.ReadCloser, error) {
	switch name {
	case FindTool, CallTool:
		if !c.hasMCP() {
			return nil, fmt.Errorf("%s: no MCP servers are configured", name)
		}
		if name == FindTool {
			return c.hub().runFind(ctx, args, c.Remote)
		}
		serverName, err := serverNameOf(args)
		if err != nil {
			return nil, err
		}
		if c.Hub != nil && c.Hub.Server(serverName) != nil {
			return c.Hub.Run(ctx, CallTool, args)
		}
		if c.remoteHas(serverName) {
			// The server lives on the active execution provider (e.g. the
			// laptop behind a VPN): route the call down its exec channel,
			// where the provider serves CallMCP from its own local hub. Only
			// the call crosses the channel — never the provider's credentials.
			return c.Exec.Run(ctx, CallTool, args)
		}
		return nil, fmt.Errorf("unknown MCP server %q (see the FindMCP description for configured servers)", serverName)
	default:
		return c.Exec.Run(ctx, name, args)
	}
}

// remoteHas reports whether a server name is one of the remote (provider-
// hosted) servers.
func (c *Composite) remoteHas(name string) bool {
	for _, ms := range c.Remote {
		if ms.Name == name {
			return true
		}
	}
	return false
}

// serverNameOf extracts server_name from a CallMCP argument payload.
func serverNameOf(args []byte) (string, error) {
	var in struct {
		ServerName string `json:"server_name"`
	}
	if err := json.Unmarshal(args, &in); err != nil {
		return "", fmt.Errorf("parse CallMCP arguments: %w", err)
	}
	return in.ServerName, nil
}

// Environment reports the execution provider's environment; the hub has none.
func (c *Composite) Environment() string { return c.Exec.Environment() }
