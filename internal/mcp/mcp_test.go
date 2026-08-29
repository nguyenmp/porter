package mcp

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"porter/internal/llm"
	"porter/internal/tools"
)

// mockMCP is a configurable MCP server for tests, speaking the streamable
// HTTP transport.
type mockMCP struct {
	mu sync.Mutex

	tools    []map[string]any
	token    string // require this bearer token ("" = no auth check)
	session  string // session id returned on initialize
	checkSID bool   // require Mcp-Session-Id on non-initialize requests
	sses     bool   // respond as text/event-stream instead of application/json
	initErr  *rpcError
	failList bool
	cursor   bool // paginate tools/list into two pages
	callErr  bool // tools/call returns isError

	calls []mockCall
}

type mockCall struct {
	tool string
	args map[string]any
}

func (m *mockMCP) handler() http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req struct {
			ID     any             `json:"id"`
			Method string          `json:"method"`
			Params json.RawMessage `json:"params"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, "bad request", http.StatusBadRequest)
			return
		}
		if m.token != "" && r.Header.Get("Authorization") != "Bearer "+m.token {
			http.Error(w, "unauthorized", http.StatusUnauthorized)
			return
		}
		if req.Method != "initialize" && m.checkSID && r.Header.Get("Mcp-Session-Id") != m.session {
			http.Error(w, "missing session id", http.StatusBadRequest)
			return
		}
		switch req.Method {
		case "initialize":
			if m.initErr != nil {
				m.respond(w, m.initErr, req.ID)
				return
			}
			m.respond(w, map[string]any{
				"protocolVersion": "2025-06-18",
				"sessionId":       m.session,
				"capabilities":    map[string]any{},
				"serverInfo":      map[string]any{"name": "mock", "version": "1.0"},
			}, req.ID)
		case "notifications/initialized":
			w.WriteHeader(http.StatusAccepted)
		case "tools/list":
			if m.failList {
				m.respond(w, &rpcError{Code: -32000, Message: "list failed"}, req.ID)
				return
			}
			if m.cursor {
				// First list call returns page 1 + a cursor; the second (the
				// request carries the cursor param) returns the rest.
				if len(req.Params) > 0 && strings.Contains(string(req.Params), "cursor") {
					m.respond(w, map[string]any{"tools": m.tools[1:]}, req.ID)
				} else {
					m.respond(w, map[string]any{"tools": m.tools[:1], "nextCursor": "page2"}, req.ID)
				}
				return
			}
			m.respond(w, map[string]any{"tools": m.tools}, req.ID)
		case "tools/call":
			var p struct {
				Name      string         `json:"name"`
				Arguments map[string]any `json:"arguments"`
			}
			_ = json.Unmarshal(req.Params, &p)
			m.mu.Lock()
			m.calls = append(m.calls, mockCall{tool: p.Name, args: p.Arguments})
			m.mu.Unlock()
			resp := map[string]any{
				"content": []map[string]any{{"type": "text", "text": "result of " + p.Name}},
				"isError": m.callErr,
			}
			if m.callErr {
				resp["content"] = []map[string]any{{"type": "text", "text": "boom"}}
			}
			m.respond(w, resp, req.ID)
		default:
			http.Error(w, "unknown method "+req.Method, http.StatusBadRequest)
		}
	})
}

func (m *mockMCP) respond(w http.ResponseWriter, v any, id any) {
	resp := rpcResponse{JSONRPC: "2.0", ID: id}
	switch x := v.(type) {
	case *rpcError:
		resp.Error = x
	default:
		b, _ := json.Marshal(v)
		resp.Result = b
	}
	data, _ := json.Marshal(resp)
	if m.sses {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("event: message\ndata: "))
		_, _ = w.Write(data)
		_, _ = w.Write([]byte("\n\n"))
		return
	}
	w.Header().Set("Content-Type", "application/json")
	_, _ = w.Write(data)
}

// writeConfig writes a porter.mcp.json with the given server entries and
// returns its path.
func writeConfig(t *testing.T, servers ...map[string]any) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "porter.mcp.json")
	data, _ := json.Marshal(map[string]any{"servers": servers})
	if err := os.WriteFile(path, data, 0o644); err != nil {
		t.Fatalf("write config: %v", err)
	}
	return path
}

func serverEntry(name, url, token string) map[string]any {
	e := map[string]any{"name": name, "description": "The " + name + " server", "url": url}
	if token != "" {
		e["auth"] = map[string]any{"type": "bearer", "token": token}
	}
	return e
}

func TestLoadNoConfig(t *testing.T) {
	for _, path := range []string{"", filepath.Join(t.TempDir(), "nope.json")} {
		h, err := Load(path, nil)
		if err != nil {
			t.Fatalf("Load(%q): %v", path, err)
		}
		if len(h.Names()) != 0 {
			t.Errorf("Load(%q): %d servers, want 0", path, len(h.Names()))
		}
	}
}

func TestLoadMalformedAndInvalid(t *testing.T) {
	bad := []struct {
		name string
		body string
	}{
		{"malformed", `{nope`},
		{"no name", `{"servers":[{"url":"https://x"}]}`},
		{"no url", `{"servers":[{"name":"x"}]}`},
		{"bad url", `{"servers":[{"name":"x","url":"ftp://x"}]}`},
		{"bad auth", `{"servers":[{"name":"x","url":"https://x","auth":{"type":"oauth"}}]}`},
		{"duplicate", `{"servers":[{"name":"x","url":"https://a"},{"name":"x","url":"https://b"}]}`},
	}
	for _, tc := range bad {
		t.Run(tc.name, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "porter.mcp.json")
			if err := os.WriteFile(path, []byte(tc.body), 0o644); err != nil {
				t.Fatal(err)
			}
			if _, err := Load(path, nil); err == nil {
				t.Error("Load: want error")
			}
		})
	}
}

func TestFetchAndCall(t *testing.T) {
	m := &mockMCP{
		tools: []map[string]any{
			{"name": "search", "description": "Search the web for a query", "inputSchema": map[string]any{
				"type": "object", "properties": map[string]any{"q": map[string]any{"type": "string"}}, "required": []string{"q"},
			}},
			{"name": "fetch_page", "description": "Fetch a web page"},
		},
		token:    "sekret",
		session:  "sess-123",
		checkSID: true,
	}
	ts := httptest.NewServer(m.handler())
	defer ts.Close()

	h, err := Load(writeConfig(t, serverEntry("web", ts.URL, "sekret")), nil)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}

	s := h.Server("web")
	if s == nil {
		t.Fatal("server web not registered")
	}
	if status, _ := s.Status(); status != "ok" {
		t.Fatalf("status = %q, want ok", status)
	}
	if got := len(s.Tools()); got != 2 {
		t.Fatalf("tools = %d, want 2", got)
	}

	// FindMCP: snippet mode with a query.
	out, err := h.Run(context.Background(), findTool, []byte(`{"server_name":"web","query":"search"}`))
	if err != nil {
		t.Fatalf("FindMCP: %v", err)
	}
	data, _ := io.ReadAll(out)
	_ = out.Close()
	if !strings.Contains(string(data), "server web (2 tools)") || !strings.Contains(string(data), "search") {
		t.Errorf("FindMCP snippet output = %q", data)
	}
	if strings.Contains(string(data), "fetch_page") {
		t.Errorf("FindMCP query filter leaked fetch_page: %q", data)
	}

	// FindMCP: full mode includes the input schema.
	out, err = h.Run(context.Background(), findTool, []byte(`{"server_name":"web","full":true}`))
	if err != nil {
		t.Fatalf("FindMCP full: %v", err)
	}
	data, _ = io.ReadAll(out)
	_ = out.Close()
	if !strings.Contains(string(data), "inputSchema") || !strings.Contains(string(data), `"q"`) {
		t.Errorf("FindMCP full output missing schema: %q", data)
	}

	// The FindMCP tool definition's description lists the server.
	defs := h.Defs()
	if len(defs) != 2 || defs[0].Function.Name != findTool || defs[1].Function.Name != callTool {
		t.Fatalf("Defs = %+v", defs)
	}
	if !strings.Contains(defs[0].Function.Description, "web (2 tools)") {
		t.Errorf("FindMCP description missing server list: %q", defs[0].Function.Description)
	}

	// CallMCP: args forwarded, text result returned.
	out, err = h.Run(context.Background(), callTool, []byte(`{"server_name":"web","tool_name":"search","args":{"q":"hello"}}`))
	if err != nil {
		t.Fatalf("CallMCP: %v", err)
	}
	data, _ = io.ReadAll(out)
	_ = out.Close()
	if string(data) != "result of search" {
		t.Errorf("CallMCP result = %q", data)
	}
	if len(m.calls) != 1 || m.calls[0].tool != "search" || m.calls[0].args["q"] != "hello" {
		t.Errorf("mock calls = %+v", m.calls)
	}

	// CallMCP: args passed as a JSON string are unwrapped.
	_, _ = h.Run(context.Background(), callTool, []byte(`{"server_name":"web","tool_name":"search","args":"{\"q\":\"str\"}"}`))
	if len(m.calls) != 2 || m.calls[1].args["q"] != "str" {
		t.Errorf("string args not unwrapped: %+v", m.calls)
	}
}

func TestSSEResponses(t *testing.T) {
	m := &mockMCP{
		tools:   []map[string]any{{"name": "ping", "description": "Ping"}},
		sses:    true,
		session: "sse-sess",
	}
	ts := httptest.NewServer(m.handler())
	defer ts.Close()

	h, err := Load(writeConfig(t, serverEntry("sse", ts.URL, "")), nil)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if status, _ := h.Server("sse").Status(); status != "ok" {
		t.Fatalf("status = %q, want ok", status)
	}
	out, err := h.Run(context.Background(), callTool, []byte(`{"server_name":"sse","tool_name":"ping"}`))
	if err != nil {
		t.Fatalf("CallMCP over SSE: %v", err)
	}
	data, _ := io.ReadAll(out)
	_ = out.Close()
	if string(data) != "result of ping" {
		t.Errorf("SSE CallMCP result = %q", data)
	}
}

func TestPagination(t *testing.T) {
	m := &mockMCP{
		tools:  []map[string]any{{"name": "a", "description": "A"}, {"name": "b", "description": "B"}},
		cursor: true,
	}
	ts := httptest.NewServer(m.handler())
	defer ts.Close()

	h, err := Load(writeConfig(t, serverEntry("pages", ts.URL, "")), nil)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if got := len(h.Server("pages").Tools()); got != 2 {
		t.Errorf("paginated tools = %d, want 2", got)
	}
}

func TestServerError(t *testing.T) {
	m := &mockMCP{initErr: &rpcError{Code: -32000, Message: "refused"}}
	ts := httptest.NewServer(m.handler())
	defer ts.Close()

	h, err := Load(writeConfig(t, serverEntry("down", ts.URL, "")), nil)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	s := h.Server("down")
	if status, msg := s.Status(); status != "error" || !strings.Contains(msg, "refused") {
		t.Errorf("status = %q, %q; want error with 'refused'", status, msg)
	}
	out, err := h.Run(context.Background(), findTool, []byte(`{"server_name":"down"}`))
	if err != nil {
		t.Fatalf("FindMCP: %v", err)
	}
	data, _ := io.ReadAll(out)
	_ = out.Close()
	if !strings.Contains(string(data), "initialize: MCP error -32000: refused") {
		t.Errorf("FindMCP error output = %q", data)
	}
	// The FindMCP tool description also reflects the failure.
	if !strings.Contains(h.Defs()[0].Function.Description, "down (error:") {
		t.Errorf("FindMCP description missing error status: %q", h.Defs()[0].Function.Description)
	}
}

func TestCallIsError(t *testing.T) {
	m := &mockMCP{tools: []map[string]any{{"name": "fail", "description": "Fails"}}, callErr: true}
	ts := httptest.NewServer(m.handler())
	defer ts.Close()

	h, err := Load(writeConfig(t, serverEntry("err", ts.URL, "")), nil)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	out, err := h.Run(context.Background(), callTool, []byte(`{"server_name":"err","tool_name":"fail"}`))
	if err != nil {
		t.Fatalf("CallMCP: %v", err)
	}
	data, _ := io.ReadAll(out)
	_ = out.Close()
	if string(data) != "error: boom" {
		t.Errorf("isError result = %q, want 'error: boom'", data)
	}
}

func TestCallCancellation(t *testing.T) {
	m := &mockMCP{tools: []map[string]any{{"name": "slow", "description": "Slow"}}}
	ts := httptest.NewServer(m.handler())
	defer ts.Close()

	h, err := Load(writeConfig(t, serverEntry("slow", ts.URL, "")), nil)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel() // cancelled before the call, like a user clicking Cancel
	out, err := h.Run(ctx, callTool, []byte(`{"server_name":"slow","tool_name":"slow"}`))
	if err != nil {
		t.Fatalf("CallMCP: %v", err)
	}
	data, _ := io.ReadAll(out)
	_ = out.Close()
	if !strings.HasPrefix(string(data), "error:") {
		t.Errorf("cancelled call result = %q, want error prefix", data)
	}
}

func TestCallErrors(t *testing.T) {
	h := New(nil)
	cases := []struct {
		name string
		args string
	}{
		{"unknown server", `{"server_name":"nope","tool_name":"x"}`},
		{"missing server", `{"tool_name":"x"}`},
		{"missing tool", `{"server_name":"x"}`},
		{"bad args", `{"server_name":"x","tool_name":"y","args":"not json"}`},
	}
	for _, tc := range cases {
		if _, err := h.Run(context.Background(), callTool, []byte(tc.args)); err == nil {
			t.Errorf("%s: want error", tc.name)
		}
	}
	if _, err := h.Run(context.Background(), "nope", nil); err == nil {
		t.Error("unknown hub tool: want error")
	}
}

func TestFlattenContent(t *testing.T) {
	got := flattenContent([]callContent{
		{Type: "text", Text: "hello"},
		{Type: "image", Data: "AAAA", MimeType: "image/png"},
		{Type: "resource", MimeType: "text/plain"},
		{Type: "weird"},
	})
	want := "hello\n[image content omitted: 4 bytes, image/png]\n[resource content omitted: text/plain]\n[content omitted: unknown type \"weird\"]"
	if got != want {
		t.Errorf("flattenContent = %q, want %q", got, want)
	}
}

func TestCompositeRouting(t *testing.T) {
	m := &mockMCP{tools: []map[string]any{{"name": "echo", "description": "Echo"}}}
	ts := httptest.NewServer(m.handler())
	defer ts.Close()

	h, err := Load(writeConfig(t, serverEntry("echo", ts.URL, "")), nil)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	exec := tools.NewDispatcher()
	c := &Composite{Exec: exec, Hub: h}

	// With servers, Defs includes the hub tools after the exec tools.
	names := map[string]bool{}
	for _, d := range c.Defs() {
		names[d.Function.Name] = true
	}
	for _, want := range []string{"shell", findTool, callTool} {
		if !names[want] {
			t.Errorf("Defs missing %q: %v", want, names)
		}
	}

	// FindMCP routes to the hub, not the dispatcher.
	out, err := c.Run(context.Background(), findTool, []byte(`{"server_name":"echo"}`))
	if err != nil {
		t.Fatalf("composite FindMCP: %v", err)
	}
	data, _ := io.ReadAll(out)
	_ = out.Close()
	if !strings.Contains(string(data), "server echo") {
		t.Errorf("composite FindMCP output = %q", data)
	}

	// shell routes to the dispatcher.
	out, err = c.Run(context.Background(), "shell", []byte(`{"command":"echo hi"}`))
	if err != nil {
		t.Fatalf("composite shell: %v", err)
	}
	data, _ = io.ReadAll(out)
	_ = out.Close()
	if !strings.Contains(string(data), "hi") {
		t.Errorf("composite shell output = %q", data)
	}

	// Unknown names still hit the dispatcher's error.
	if _, err := c.Run(context.Background(), "nope", nil); err == nil {
		t.Error("composite unknown tool: want error")
	}

	// No servers: hub tools are not exposed and calls are rejected.
	empty := &Composite{Exec: exec, Hub: New(nil)}
	if len(empty.Defs()) != 1 || empty.Defs()[0].Function.Name != "shell" {
		t.Errorf("empty hub Defs = %+v", empty.Defs())
	}
	if _, err := empty.Run(context.Background(), findTool, nil); err == nil {
		t.Error("empty hub FindMCP: want error")
	}
	if _, err := empty.Run(context.Background(), "shell", []byte(`{"command":"echo hi"}`)); err != nil {
		t.Errorf("empty hub shell should still work: %v", err)
	}
}

// compositeProvider satisfies tools.Provider (compile-time check that
// Composite implements the interface).
var _ tools.Provider = (*Composite)(nil)

var _ = llm.Tool{}
