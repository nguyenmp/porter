package mcp

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strings"
	"sync/atomic"

	"porter/internal/llm"
	"porter/internal/tools"
)

// The two hub tools are the model's entire MCP surface: any number of MCP
// servers and their tools add exactly these two tools to the model's context.
const (
	findTool = "FindMCP"
	callTool = "CallMCP"
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
// rebuilt per request from the current registry so the model always sees the
// server list (name, description, tool count, load status) without calling
// anything.
func (h *Hub) findDef() llm.Tool {
	var b strings.Builder
	b.WriteString("Discover MCP (Model Context Protocol) servers and their tools, then call a tool with CallMCP. ")
	h.mu.RLock()
	if len(h.servers) == 0 {
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
		b.WriteString("Call with server_name to list one server's tools (omitting it searches all servers) and an optional query substring to filter tool names and descriptions. Pass full=true for full descriptions and input schemas — use sparingly, it is verbose.")
	}
	h.mu.RUnlock()
	return llm.Tool{
		Type: "function",
		Function: llm.Function{
			Name:        findTool,
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
			Name:        callTool,
			Description: "Call a tool on an MCP server. The tool must exist on the server; discover it with FindMCP, which also shows its input schema when asked. The result is the server's text response; a failing call is reported as an error in the result.",
			Parameters: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"server_name": map[string]any{"type": "string", "description": "The MCP server to call (see FindMCP for configured servers)."},
					"tool_name":   map[string]any{"type": "string", "description": "The tool to call on that server."},
					"args":        map[string]any{"type": "object", "description": "The tool's arguments, matching its input schema (see FindMCP with full=true)."},
				},
				"required": []string{"server_name", "tool_name"},
			},
		},
	}
}

// Defs returns the hub's model-facing tool definitions. The Hub is not a full
// tools.Provider (it has no execution environment); Composite merges it with
// one.
func (h *Hub) Defs() []llm.Tool {
	return []llm.Tool{h.findDef(), callDef()}
}

// Run executes a hub tool. It returns an error only when the call cannot be
// started — unknown tool, unknown server, malformed arguments (mirroring how
// load_skill rejects an unknown skill). Remote failures — HTTP or JSON-RPC
// errors, timeouts, the server reporting isError — surface in the streamed
// result text, mirroring how the shell tool reports command failures, so the
// model sees the failure and can react.
func (h *Hub) Run(ctx context.Context, name string, args []byte) (io.ReadCloser, error) {
	switch name {
	case findTool:
		return h.runFind(ctx, args)
	case callTool:
		return h.runCall(ctx, args)
	default:
		return nil, fmt.Errorf("unknown MCP tool: %q", name)
	}
}

func (h *Hub) runFind(ctx context.Context, args []byte) (io.ReadCloser, error) {
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
	if len(servers) == 0 {
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
				fmt.Fprintf(&b, "  %s: %s\n", t.Name, snippet(t.Description))
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

func (h *Hub) runCall(ctx context.Context, args []byte) (io.ReadCloser, error) {
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
	params := map[string]any{"name": in.ToolName}
	if len(argMap) > 0 {
		params["arguments"] = argMap
	}

	result, _, err := s.call(ctx, h.client, nextID(), "tools/call", params)
	if err != nil {
		// The call reached the server but failed (HTTP/JSON-RPC/timeout):
		// report as result content, like a command failure, so the model can
		// react (retry, pick another tool).
		return newResultStream("error: " + err.Error()), nil
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
// the MCP hub (FindMCP, CallMCP) into a single tools.Provider, so the agent
// loop keeps talking to one provider. Routing by name guarantees the hub
// tools are served here, on the server, and never cross the exec channel to a
// connected execution client — where the MCP credentials would leak. When the
// hub has no servers, the hub tools are not exposed at all.
type Composite struct {
	// Exec is the session's execution provider (local dispatcher or a remote
	// client).
	Exec tools.Provider
	// Hub serves FindMCP and CallMCP. May be nil.
	Hub *Hub
}

// Defs returns the execution provider's tools plus the hub's, when the hub
// has servers configured.
func (c *Composite) Defs() []llm.Tool {
	defs := c.Exec.Defs()
	if c.Hub != nil && len(c.Hub.Names()) > 0 {
		defs = append(defs, c.Hub.Defs()...)
	}
	return defs
}

// Run routes the hub tools to the hub and everything else to the execution
// provider.
func (c *Composite) Run(ctx context.Context, name string, args []byte) (io.ReadCloser, error) {
	switch name {
	case findTool, callTool:
		if c.Hub == nil || len(c.Hub.Names()) == 0 {
			return nil, fmt.Errorf("%s: no MCP servers are configured", name)
		}
		return c.Hub.Run(ctx, name, args)
	default:
		return c.Exec.Run(ctx, name, args)
	}
}

// Environment reports the execution provider's environment; the hub has none.
func (c *Composite) Environment() string { return c.Exec.Environment() }
