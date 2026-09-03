package mcp

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"porter/internal/api"
	"porter/internal/llm"
	"porter/internal/tools"
)

// mockMCP is a configurable MCP server for tests, speaking the streamable
// HTTP transport.
type mockMCP struct {
	mu sync.Mutex

	tools      []map[string]any
	token      string // require this bearer token ("" = no auth check)
	session    string // session id returned on initialize
	checkSID   bool   // require Mcp-Session-Id on non-initialize requests
	sses       bool   // respond as text/event-stream instead of application/json
	initErr    *rpcError
	failList   bool
	cursor     bool      // paginate tools/list into two pages
	callErr    bool      // tools/call returns isError
	callRPCErr *rpcError // tools/call returns this JSON-RPC error instead of a result
	// Session-expiry simulation: after expireAfter id-bearing non-initialize
	// requests the server rejects with -32001 "Session not found" until the
	// next initialize mints a new session generation.
	expireAfter int
	reqs        int // id-bearing non-initialize requests in the current session
	sessGen     int // session generation, 1-based; "sess-<n>" when expireAfter > 0
	inits       int // initialize requests seen

	calls []mockCall
}

// sessionID returns the session id the server currently accepts.
func (m *mockMCP) sessionID() string {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.expireAfter > 0 {
		return fmt.Sprintf("sess-%d", m.sessGen)
	}
	return m.session
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
		if req.Method != "initialize" && req.ID != nil && m.expireAfter > 0 {
			m.mu.Lock()
			m.reqs++
			expired := m.reqs > m.expireAfter
			m.mu.Unlock()
			if expired {
				w.WriteHeader(http.StatusNotFound)
				m.respond(w, &rpcError{Code: -32001, Message: "Session not found"}, req.ID)
				return
			}
		}
		if req.Method != "initialize" && m.checkSID && r.Header.Get("Mcp-Session-Id") != m.sessionID() {
			http.Error(w, "missing session id", http.StatusBadRequest)
			return
		}
		switch req.Method {
		case "initialize":
			if m.initErr != nil {
				m.respond(w, m.initErr, req.ID)
				return
			}
			m.mu.Lock()
			m.inits++
			if m.expireAfter > 0 {
				m.sessGen++
				m.reqs = 0
			}
			m.mu.Unlock()
			m.respond(w, map[string]any{
				"protocolVersion": "2025-06-18",
				"sessionId":       m.sessionID(),
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
			if m.callRPCErr != nil {
				m.respond(w, m.callRPCErr, req.ID)
				return
			}
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
		{"bad auth", `{"servers":[{"name":"x","url":"https://x","auth":{"type":"apikey"}}]}`},
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
				"type": "object", "properties": map[string]any{"q": map[string]any{"type": "string"}, "limit": map[string]any{"type": "integer"}}, "required": []string{"q"},
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

	// FindMCP: snippet mode with a query. The schema's required field shows
	// up inline so the model knows arguments are mandatory without full mode.
	out, err := h.Run(context.Background(), FindTool, []byte(`{"server_name":"web","query":"search"}`))
	if err != nil {
		t.Fatalf("FindMCP: %v", err)
	}
	data, _ := io.ReadAll(out)
	_ = out.Close()
	if !strings.Contains(string(data), "server web (2 tools)") || !strings.Contains(string(data), "search") {
		t.Errorf("FindMCP snippet output = %q", data)
	}
	if !strings.Contains(string(data), "[requires: q; optional: limit]") {
		t.Errorf("FindMCP snippet should list required and optional args inline: %q", data)
	}
	if strings.Contains(string(data), "fetch_page") {
		t.Errorf("FindMCP query filter leaked fetch_page: %q", data)
	}

	// FindMCP: full mode includes the input schema.
	out, err = h.Run(context.Background(), FindTool, []byte(`{"server_name":"web","full":true}`))
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
	if len(defs) != 2 || defs[0].Function.Name != FindTool || defs[1].Function.Name != CallTool {
		t.Fatalf("Defs = %+v", defs)
	}
	if !strings.Contains(defs[0].Function.Description, "web (2 tools)") {
		t.Errorf("FindMCP description missing server list: %q", defs[0].Function.Description)
	}

	// CallMCP: args forwarded, text result returned.
	out, err = h.Run(context.Background(), CallTool, []byte(`{"server_name":"web","tool_name":"search","args":{"q":"hello"}}`))
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
	_, _ = h.Run(context.Background(), CallTool, []byte(`{"server_name":"web","tool_name":"search","args":"{\"q\":\"str\"}"}`))
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
	out, err := h.Run(context.Background(), CallTool, []byte(`{"server_name":"sse","tool_name":"ping"}`))
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
	out, err := h.Run(context.Background(), FindTool, []byte(`{"server_name":"down"}`))
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
	out, err := h.Run(context.Background(), CallTool, []byte(`{"server_name":"err","tool_name":"fail"}`))
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
	out, err := h.Run(ctx, CallTool, []byte(`{"server_name":"slow","tool_name":"slow"}`))
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
		if _, err := h.Run(context.Background(), CallTool, []byte(tc.args)); err == nil {
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
	for _, want := range []string{"shell", FindTool, CallTool} {
		if !names[want] {
			t.Errorf("Defs missing %q: %v", want, names)
		}
	}

	// FindMCP routes to the hub, not the dispatcher.
	out, err := c.Run(context.Background(), FindTool, []byte(`{"server_name":"echo"}`))
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

	// No servers: hub tools are not exposed (the base tool set remains) and
	// hub calls are rejected.
	empty := &Composite{Exec: exec, Hub: New(nil)}
	if names := toolNames(empty.Defs()); !equalStrings(names, []string{"shell", "read_with_line_numbers", "line_replace", "string_replace"}) {
		t.Errorf("empty hub Defs = %v, want the base tool set", names)
	}
	if _, err := empty.Run(context.Background(), FindTool, nil); err == nil {
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

func fileStat(path string) (fi os.FileInfo, err error) { return os.Stat(path) }

// recordingProvider is a tools.Provider that records every tool name it runs
// and serves CallMCP with a fixed result, delegating everything else to a
// plain Dispatcher.
type recordingProvider struct {
	tools.Dispatcher
	mu    sync.Mutex
	calls []string
}

func (r *recordingProvider) Run(ctx context.Context, name string, args []byte) (io.ReadCloser, error) {
	r.mu.Lock()
	r.calls = append(r.calls, name)
	r.mu.Unlock()
	if name == CallTool {
		return newResultStream("remote-result"), nil
	}
	return r.Dispatcher.Run(ctx, name, args)
}

func (r *recordingProvider) callsOf() []string {
	r.mu.Lock()
	defer r.mu.Unlock()
	return append([]string(nil), r.calls...)
}

// TestCompositeRemoteRouting covers a Composite whose provider hosts MCP
// servers (e.g. a laptop behind a VPN): FindMCP lists them with their host,
// CallMCP routes down the exec channel (never the server hub), and other
// tools still work. It also exercises the nil-hub path, where remote servers
// alone expose the MCP tools.
func TestCompositeRemoteRouting(t *testing.T) {
	remote := []api.MCPServer{{
		Name:        "retool",
		Description: "Retool MCP",
		Host:        "macbook",
		Tools:       []api.MCPTool{{Name: "whoami", Description: "Who am I"}},
	}}
	rec := &recordingProvider{}
	c := &Composite{Exec: rec, Remote: remote} // Hub nil

	// Remote servers alone expose the MCP tools.
	names := map[string]bool{}
	for _, d := range c.Defs() {
		names[d.Function.Name] = true
	}
	for _, want := range []string{"shell", FindTool, CallTool} {
		if !names[want] {
			t.Errorf("Defs missing %q: %v", want, names)
		}
	}

	// FindMCP lists the remote server with its host tag.
	out, err := c.Run(context.Background(), FindTool, []byte(`{"server_name":"retool"}`))
	if err != nil {
		t.Fatalf("FindMCP: %v", err)
	}
	data, _ := io.ReadAll(out)
	_ = out.Close()
	if !strings.Contains(string(data), "server retool (1 tools)") ||
		!strings.Contains(string(data), "(hosted on macbook)") {
		t.Errorf("FindMCP remote output = %q", data)
	}

	// CallMCP for the remote server routes down the exec channel.
	out, err = c.Run(context.Background(), CallTool, []byte(`{"server_name":"retool","tool_name":"whoami"}`))
	if err != nil {
		t.Fatalf("CallMCP remote: %v", err)
	}
	data, _ = io.ReadAll(out)
	_ = out.Close()
	if !strings.Contains(string(data), "remote-result") {
		t.Errorf("CallMCP remote result = %q", data)
	}
	if calls := rec.callsOf(); len(calls) != 1 || calls[0] != CallTool {
		t.Errorf("exec provider calls = %v, want [CallMCP]", calls)
	}

	// shell still routes to the dispatcher (not the MCP path).
	out, err = c.Run(context.Background(), "shell", []byte(`{"command":"echo hi"}`))
	if err != nil {
		t.Fatalf("shell: %v", err)
	}
	data, _ = io.ReadAll(out)
	_ = out.Close()
	if !strings.Contains(string(data), "hi") {
		t.Errorf("shell result = %q", data)
	}

	// Unknown remote server is an error, not a silent local miss.
	if _, err := c.Run(context.Background(), CallTool, []byte(`{"server_name":"nope","tool_name":"x"}`)); err == nil {
		t.Error("CallMCP unknown remote: want error")
	}
}

// TestCompositeRoutesHubServerLocally proves a server owned by the hub is
// served on the server even when the provider also hosts servers: CallMCP for
// it must not cross the exec channel.
func TestCompositeRoutesHubServerLocally(t *testing.T) {
	m := &mockMCP{tools: []map[string]any{{"name": "echo", "description": "Echo"}}}
	ts := httptest.NewServer(m.handler())
	defer ts.Close()
	h, err := Load(writeConfig(t, serverEntry("echo", ts.URL, "")), nil)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	remote := []api.MCPServer{{Name: "retool", Description: "Retool", Host: "macbook"}}
	rec := &recordingProvider{}
	c := &Composite{Exec: rec, Hub: h, Remote: remote}

	out, err := c.Run(context.Background(), CallTool, []byte(`{"server_name":"echo","tool_name":"echo"}`))
	if err != nil {
		t.Fatalf("CallMCP hub server: %v", err)
	}
	data, _ := io.ReadAll(out)
	_ = out.Close()
	if !strings.Contains(string(data), "echo") {
		t.Errorf("hub server result = %q", data)
	}
	if calls := rec.callsOf(); len(calls) != 0 {
		t.Errorf("hub-owned CallMCP crossed the exec channel: %v", calls)
	}
}

// TestSummary renders a hub's servers as reported metadata (used by the host
// to post its MCP servers with the environment).
func TestSummary(t *testing.T) {
	m := &mockMCP{tools: []map[string]any{{"name": "echo", "description": "Echo"}}}
	ts := httptest.NewServer(m.handler())
	defer ts.Close()
	h, err := Load(writeConfig(t, serverEntry("echo", ts.URL, "")), nil)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	sum := h.Summary()
	if len(sum) != 1 {
		t.Fatalf("Summary = %d servers, want 1", len(sum))
	}
	s := sum[0]
	if s.Name != "echo" || s.Status != "ok" || len(s.Tools) != 1 || s.Tools[0].Name != "echo" {
		t.Errorf("Summary server = %+v", s)
	}
}

// TestSessionExpiryRecovery simulates a server whose streamable-HTTP session
// lapses server-side (Retool's sessions live minutes): the first call works,
// the second is rejected with -32001 "Session not found", and porter must
// transparently re-initialize and retry without the caller noticing.
func TestSessionExpiryRecovery(t *testing.T) {
	// The session serves two id-bearing requests (tools/list at load plus one
	// call), then lapses; each initialize mints the next generation.
	m := &mockMCP{
		tools:       []map[string]any{{"name": "ping", "description": "Ping"}},
		checkSID:    true,
		expireAfter: 2,
	}
	ts := httptest.NewServer(m.handler())
	defer ts.Close()

	h, err := Load(writeConfig(t, serverEntry("sess", ts.URL, "")), nil)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if status, _ := h.Server("sess").Status(); status != "ok" {
		t.Fatalf("status = %q, want ok", status)
	}

	call := func() string {
		t.Helper()
		out, err := h.Run(context.Background(), CallTool, []byte(`{"server_name":"sess","tool_name":"ping"}`))
		if err != nil {
			t.Fatalf("CallMCP: %v", err)
		}
		data, _ := io.ReadAll(out)
		_ = out.Close()
		return string(data)
	}

	// First call rides the original session (sess-1).
	if got := call(); got != "result of ping" {
		t.Errorf("first call = %q, want result of ping", got)
	}
	// Second call hits the lapsed session and must recover transparently.
	if got := call(); got != "result of ping" {
		t.Errorf("call after expiry = %q, want result of ping", got)
	}
	// Recovery re-initialized to a new session generation.
	if m.sessionID() != "sess-2" {
		t.Errorf("session after recovery = %q, want sess-2", m.sessionID())
	}
	if m.inits != 2 {
		t.Errorf("initialize count = %d, want 2 (load + recovery)", m.inits)
	}
	if len(m.calls) != 2 {
		t.Errorf("tools/call count = %d, want 2", len(m.calls))
	}
}

// TestNoSessionRecoveryForOtherErrors proves recovery only fires for -32001
// "Session not found": an unrelated JSON-RPC error on a call is reported as-is
// without re-initializing.
func TestNoSessionRecoveryForOtherErrors(t *testing.T) {
	m := &mockMCP{
		tools:      []map[string]any{{"name": "ping", "description": "Ping"}},
		session:    "sess-x",
		checkSID:   true,
		callRPCErr: &rpcError{Code: -32000, Message: "boom"},
	}
	ts := httptest.NewServer(m.handler())
	defer ts.Close()

	h, err := Load(writeConfig(t, serverEntry("sess", ts.URL, "")), nil)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	out, err := h.Run(context.Background(), CallTool, []byte(`{"server_name":"sess","tool_name":"ping"}`))
	if err != nil {
		t.Fatalf("CallMCP: %v", err)
	}
	data, _ := io.ReadAll(out)
	_ = out.Close()
	if !strings.Contains(string(data), "MCP error -32000: boom") {
		t.Errorf("result = %q, want MCP error -32000: boom", data)
	}
	if m.inits != 1 {
		t.Errorf("initialize count = %d, want 1 (no recovery for unrelated errors)", m.inits)
	}
}

func TestRPCErrorData(t *testing.T) {
	e := &rpcError{Code: -32602, Message: "Input validation error", Data: json.RawMessage(`[{"code":"invalid_type","path":[]}]`)}
	want := `MCP error -32602: Input validation error (data: [{"code":"invalid_type","path":[]}])`
	if got := e.Error(); got != want {
		t.Errorf("Error() = %q, want %q", got, want)
	}
	// A server that sends no data keeps the terse form.
	plain := &rpcError{Code: -32000, Message: "boom"}
	if got, want := plain.Error(), "MCP error -32000: boom"; got != want {
		t.Errorf("plain Error() = %q, want %q", got, want)
	}
}

func TestCallMCPPreflightMissingArgs(t *testing.T) {
	m := &mockMCP{
		tools: []map[string]any{
			{"name": "search", "description": "Search the web for a query", "inputSchema": map[string]any{
				"type": "object", "properties": map[string]any{"q": map[string]any{"type": "string"}, "limit": map[string]any{"type": "integer"}}, "required": []string{"q"},
			}},
		},
	}
	ts := httptest.NewServer(m.handler())
	defer ts.Close()

	h, err := Load(writeConfig(t, serverEntry("web", ts.URL, "")), nil)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}

	// Call with no args at all: rejected locally before the network, naming
	// the missing required field and pointing at the schema.
	out, err := h.Run(context.Background(), CallTool, []byte(`{"server_name":"web","tool_name":"search"}`))
	if err == nil {
		data, _ := io.ReadAll(out)
		_ = out.Close()
		t.Fatalf("CallMCP no-args: want error, got result %q", data)
	}
	if !strings.Contains(err.Error(), `missing required arguments: q`) || !strings.Contains(err.Error(), `inputSchema`) {
		t.Errorf("no-args error = %v, want a pointer at missing q and the schema", err)
	}
	if len(m.calls) != 0 {
		t.Errorf("pre-flight must not reach the server, got %d remote calls", len(m.calls))
	}

	// Missing only one of several required fields is caught too.
	out, err = h.Run(context.Background(), CallTool, []byte(`{"server_name":"web","tool_name":"search","args":{"not_q":1}}`))
	if err == nil || !strings.Contains(err.Error(), "q") {
		t.Errorf("wrong-key args: want error naming q, got %v", err)
	}

	// An args member sent empty ("" or {}) is distinguished from omitted, so
	// the model can see which of its behaviors was wrong and fix it: the
	// fix for "I sent nothing" differs from the fix for "I sent {}".
	out, err = h.Run(context.Background(), CallTool, []byte(`{"server_name":"web","tool_name":"search","args":""}`))
	if err == nil || !strings.Contains(err.Error(), "present but empty") {
		t.Errorf("empty-string args: want the present-but-empty note, got %v", err)
	}
	out, err = h.Run(context.Background(), CallTool, []byte(`{"server_name":"web","tool_name":"search","args":{}}`))
	if err == nil || !strings.Contains(err.Error(), "present but empty") {
		t.Errorf("empty-object args: want the present-but-empty note, got %v", err)
	}

	// Correct args still go through.
	out, err = h.Run(context.Background(), CallTool, []byte(`{"server_name":"web","tool_name":"search","args":{"q":"hi"}}`))
	if err != nil {
		t.Fatalf("CallMCP with args: %v", err)
	}
	data, _ := io.ReadAll(out)
	_ = out.Close()
	if string(data) != "result of search" {
		t.Errorf("result = %q", data)
	}
	if len(m.calls) != 1 {
		t.Errorf("remote calls = %d, want 1", len(m.calls))
	}
}

func TestCallMCPValidationErrorDecoration(t *testing.T) {
	m := &mockMCP{
		tools: []map[string]any{
			{"name": "search", "description": "Search", "inputSchema": map[string]any{
				"type": "object", "properties": map[string]any{"q": map[string]any{"type": "string"}},
			}},
		},
		callRPCErr: &rpcError{Code: -32602, Message: "Input validation error", Data: json.RawMessage(`[{"code":"invalid_type","received":"undefined","path":[]}]`)},
	}
	ts := httptest.NewServer(m.handler())
	defer ts.Close()

	h, err := Load(writeConfig(t, serverEntry("web", ts.URL, "")), nil)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}

	// A -32602 rejection from the server gets the "no arguments" note and the
	// tool's schema appended, so the model can correct the call.
	out, err := h.Run(context.Background(), CallTool, []byte(`{"server_name":"web","tool_name":"search"}`))
	if err != nil {
		t.Fatalf("CallMCP: %v", err)
	}
	data, _ := io.ReadAll(out)
	_ = out.Close()
	text := string(data)
	for _, want := range []string{
		"MCP error -32602: Input validation error",
		"received",
		"note: the call was sent with no arguments",
		"inputSchema",
		`"q"`,
	} {
		if !strings.Contains(text, want) {
			t.Errorf("decorated result missing %q:\n%s", want, text)
		}
	}
}

func TestArgsHint(t *testing.T) {
	// Required and optional names are both shown, sorted, in that order.
	got := argsHint(map[string]any{
		"type": "object",
		"properties": map[string]any{
			"b": map[string]any{"type": "string"},
			"a": map[string]any{"type": "string"},
			"c": map[string]any{"type": "integer"},
		},
		"required": []string{"b"},
	})
	if want := "requires: b; optional: a, c"; got != want {
		t.Errorf("argsHint = %q, want %q", got, want)
	}

	// Optional list is capped, with the overflow counted.
	props := map[string]any{"req": map[string]any{"type": "string"}}
	for i := 0; i < snippetOptionalMax+2; i++ {
		props[fmt.Sprintf("p%d", i)] = map[string]any{"type": "string"}
	}
	got = argsHint(map[string]any{"type": "object", "properties": props, "required": []string{"req"}})
	if want := "requires: req; optional: p0, p1, p2, p3, p4, p5, +2 more"; got != want {
		t.Errorf("argsHint capped = %q, want %q", got, want)
	}

	// No schema or no properties: no hint.
	if got := argsHint(nil); got != "" {
		t.Errorf("argsHint(nil) = %q, want empty", got)
	}
	if got := argsHint(map[string]any{"type": "object", "properties": map[string]any{}}); got != "" {
		t.Errorf("argsHint(no props) = %q, want empty", got)
	}
}

func TestCallDefRequiresArgs(t *testing.T) {
	def := callDef()
	params := def.Function.Parameters
	required, _ := params["required"].([]string)
	found := false
	for _, r := range required {
		if r == "args" {
			found = true
		}
	}
	if !found {
		t.Errorf("callDef required = %v, want args required so the model cannot omit it", required)
	}
	desc, _ := params["properties"].(map[string]any)["args"].(map[string]any)["description"].(string)
	if !strings.Contains(desc, "Always pass this") {
		t.Errorf("args description should make the requirement explicit: %q", desc)
	}
}

// toolNames returns the names of tool defs, in order.
func toolNames(defs []llm.Tool) []string {
	out := make([]string, len(defs))
	for i, d := range defs {
		out[i] = d.Function.Name
	}
	return out
}

// equalStrings reports whether two string slices match in order.
func equalStrings(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
