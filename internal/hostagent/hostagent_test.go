package hostagent

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"porter/internal/mcp"
	"porter/internal/tools"
)

// mockMCP is a minimal streamable-HTTP MCP server (mirrors the mcp package's
// test helper) returning one tool.
type mockMCP struct {
	tools []map[string]any
}

func (m *mockMCP) handler() http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req struct {
			ID     any             `json:"id"`
			Method string          `json:"method"`
			Params json.RawMessage `json:"params"`
		}
		_ = json.NewDecoder(r.Body).Decode(&req)
		switch req.Method {
		case "initialize":
			_ = json.NewEncoder(w).Encode(map[string]any{
				"jsonrpc": "2.0", "id": req.ID,
				"result": map[string]any{
					"protocolVersion": "2025-06-18",
					"capabilities":    map[string]any{},
					"serverInfo":      map[string]any{"name": "mock", "version": "1.0"},
				},
			})
		case "tools/list":
			_ = json.NewEncoder(w).Encode(map[string]any{
				"jsonrpc": "2.0", "id": req.ID,
				"result": map[string]any{"tools": m.tools},
			})
		case "tools/call":
			_ = json.NewEncoder(w).Encode(map[string]any{
				"jsonrpc": "2.0", "id": req.ID,
				"result": map[string]any{"content": []map[string]any{
					{"type": "text", "text": "whoami result"},
				}},
			})
		default:
			_ = json.NewEncoder(w).Encode(map[string]any{
				"jsonrpc": "2.0", "id": req.ID, "result": map[string]any{},
			})
		}
	})
}

// TestHubSummaryTagsHost proves hubSummary renders a hub's servers as
// reported metadata with the host id attached, so the porter server can list
// them as "hosted on <host>".
func TestHubSummaryTagsHost(t *testing.T) {
	ts := httptest.NewServer((&mockMCP{tools: []map[string]any{{"name": "echo", "description": "Echo"}}}).handler())
	defer ts.Close()

	dir := t.TempDir()
	cfg := filepath.Join(dir, "porter.mcp.json")
	data, _ := json.Marshal(map[string]any{"servers": []map[string]any{
		{"name": "retool", "description": "Retool", "url": ts.URL},
	}})
	if err := os.WriteFile(cfg, data, 0o644); err != nil {
		t.Fatal(err)
	}
	h, err := mcp.Load(cfg, nil)
	if err != nil {
		t.Fatalf("mcp.Load: %v", err)
	}

	sum := hubSummary(h, "macbook")
	if len(sum) != 1 {
		t.Fatalf("hubSummary = %d servers, want 1", len(sum))
	}
	if sum[0].Name != "retool" || sum[0].Host != "macbook" || sum[0].Status != "ok" {
		t.Errorf("hubSummary server = %+v", sum[0])
	}
	if len(sum[0].Tools) != 1 || sum[0].Tools[0].Name != "echo" {
		t.Errorf("hubSummary tools = %+v", sum[0].Tools)
	}
}

// TestProvisionerServesHostMCPServers proves the host's serve loop answers
// CallMCP from its local hub (executing the call against the host's own MCP
// server) and passes every other tool to the dispatcher. It drives the
// dispatch func directly with a real hub backed by a mock MCP server.
func TestProvisionerServesHostMCPServers(t *testing.T) {
	// The host's MCP server: accepts any bearer token and returns one tool.
	ts := httptest.NewServer((&mockMCP{tools: []map[string]any{{"name": "whoami", "description": "Who am I"}}}).handler())
	defer ts.Close()

	dir := t.TempDir()
	cfg := filepath.Join(dir, "porter.mcp.json")
	data, _ := json.Marshal(map[string]any{"servers": []map[string]any{
		{"name": "retool", "description": "Retool", "url": ts.URL},
	}})
	if err := os.WriteFile(cfg, data, 0o644); err != nil {
		t.Fatal(err)
	}
	h, err := mcp.Load(cfg, nil)
	if err != nil {
		t.Fatalf("mcp.Load: %v", err)
	}

	p := &provisioner{mcp: h}
	disp := newTestDispatcher(t)

	// CallMCP routes to the host's hub, which calls the host-only server.
	out, err := p.serveOne(context.Background(), mcp.CallTool, []byte(`{"server_name":"retool","tool_name":"whoami"}`), disp)
	if err != nil {
		t.Fatalf("CallMCP: %v", err)
	}
	data2, _ := io.ReadAll(out)
	_ = out.Close()
	if !strings.Contains(string(data2), "whoami") {
		t.Errorf("CallMCP result = %q, want the tool result", data2)
	}

	// shell routes to the dispatcher, not the hub.
	out, err = p.serveOne(context.Background(), "shell", []byte(`{"command":"echo hi"}`), disp)
	if err != nil {
		t.Fatalf("shell: %v", err)
	}
	data2, _ = io.ReadAll(out)
	_ = out.Close()
	if !strings.Contains(string(data2), "hi") {
		t.Errorf("shell result = %q", data2)
	}

	// Unknown hub server: the hub rejects it (an error, which the exec
	// client surfaces as "error: ..." content).
	if _, err := p.serveOne(context.Background(), mcp.CallTool, []byte(`{"server_name":"nope","tool_name":"x"}`), disp); err == nil || !strings.Contains(err.Error(), "unknown MCP server") {
		t.Errorf("CallMCP unknown err = %v, want unknown-server error", err)
	}
}

// serveOne is the serve loop's dispatch func factored out for testing: route
// CallMCP to the host's hub, everything else to the dispatcher.
func (p *provisioner) serveOne(ctx context.Context, name string, args []byte, disp *tools.Dispatcher) (io.ReadCloser, error) {
	if name == mcp.CallTool {
		return p.mcp.Run(ctx, mcp.CallTool, args)
	}
	return disp.RunDir(ctx, name, args, "")
}

// newTestDispatcher returns a dispatcher with the repo-root skills (none in a
// temp test repo) so shell works.
func newTestDispatcher(t *testing.T) *tools.Dispatcher {
	t.Helper()
	return tools.NewDispatcher()
}
