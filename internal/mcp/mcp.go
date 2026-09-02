// Package mcp connects porter to Model Context Protocol (MCP) servers over
// the streamable HTTP transport. MCP tools reach the model through exactly
// two hub tools — FindMCP (discover servers and their tools) and CallMCP
// (call one tool) — so any number of MCP servers adds a constant two tools to
// the model's context instead of one per server.
//
// The hub runs wherever its servers' credentials live: server-configured
// servers are served by the server process (hub calls never cross the exec
// channel, so server-side credentials stay server-side), and a connected
// execution host's servers (e.g. a laptop-only server behind a corporate VPN)
// are listed through FindMCP and executed on the host itself — CallMCP for a
// host-owned server is routed down the exec channel and run by the host's own
// local hub, so those credentials never leave the host. Each side's
// credentials stay on the side that owns them.
//
// Only tools are used; resources and prompts are ignored. Connections are
// stateless — one POST per JSON-RPC request, no persistent stream is held —
// so notifications/list_changed and ping are never received, and the only
// per-server state kept is the session id streamable-HTTP servers assign on
// initialize.
package mcp

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"net/http"
	"net/url"
	"os"
	"strings"
	"sync"
	"time"

	"porter/internal/api"
)

// ProtocolVersion is the MCP protocol version porter proposes on initialize.
// The server responds with the version it supports; porter accepts whatever it
// returns and echoes it on subsequent requests.
const ProtocolVersion = "2025-06-18"

// DefaultTimeout bounds every JSON-RPC request to a server (per request, not
// per connection — connections are one POST each). A server can override it
// with timeout_seconds in the config file.
const DefaultTimeout = 30 * time.Second

// Tool is one tool an MCP server exposes. InputSchema is the tool's JSON
// input schema, shown to the model in full only when FindMCP is asked for it.
type Tool struct {
	Name        string
	Description string
	InputSchema map[string]any
}

// Auth carries the credentials porter sends to a server. Type "bearer" sends
// a static token; type "oauth" uses the OAuth 2.0 authorization-code flow
// (see oauth.go): tokens live in the user's ~/.porter/mcp/tokens.json, are
// refreshed automatically, and never appear in the config file. Scope is the
// OAuth scope to request ("" = the server's defaults).
type Auth struct {
	Type  string
	Token string
	Scope string
}

// Server is one configured MCP server: its static config plus the runtime
// state discovered when the hub loads (status and tools). The static fields
// are immutable after construction; the state fields are guarded by mu.
type Server struct {
	mu sync.RWMutex

	Name        string
	Description string
	URL         string
	Auth        Auth
	Timeout     time.Duration
	// tokens is the shared OAuth token store for OAuth-authed servers; nil
	// for bearer servers (or when no store is available). Set by the Hub.
	tokens *TokenStore

	status    string // "ok", "error", or "pending"
	err       string
	tools     []Tool
	sessionID string // streamable-HTTP session id from initialize
	protocol  string // negotiated protocol version
}

// Status returns the server's load status ("ok", "error", or "pending") and,
// on error, the failure detail.
func (s *Server) Status() (string, string) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.status, s.err
}

// Tools returns the tools discovered on the server (empty when it failed or
// has none). The slice is immutable once set, so it is safe to keep.
func (s *Server) Tools() []Tool {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.tools
}

// Tool returns the discovered tool with the given name, or ok=false.
func (s *Server) Tool(name string) (Tool, bool) {
	for _, t := range s.Tools() {
		if t.Name == name {
			return t, true
		}
	}
	return Tool{}, false
}

func (s *Server) setStatus(status, err string) {
	s.mu.Lock()
	s.status, s.err = status, err
	s.mu.Unlock()
}

func (s *Server) setTools(tools []Tool) {
	s.mu.Lock()
	s.tools = tools
	s.mu.Unlock()
}

// config is the on-disk shape of porter.mcp.json.
type config struct {
	Servers []serverConfig `json:"servers"`
}

type serverConfig struct {
	Name           string     `json:"name"`
	Description    string     `json:"description"`
	URL            string     `json:"url"`
	Auth           authConfig `json:"auth"`
	TimeoutSeconds int        `json:"timeout_seconds,omitempty"`
}

type authConfig struct {
	Type  string `json:"type"`
	Token string `json:"token"`
	Scope string `json:"scope,omitempty"`
}

// Hub is the in-memory registry of MCP servers and their tools. It is loaded
// at server startup: every configured server is fetched concurrently, and a
// server that fails to respond is recorded with an error status rather than
// failing startup. The registry is read on every model request, and FindMCP's
// description is rebuilt from it per request, so the model always sees the
// current server list. It is safe for concurrent use.
type Hub struct {
	mu      sync.RWMutex
	servers map[string]*Server
	order   []string // server names in config order
	client  *http.Client
	// tokens is the shared OAuth token store, opened at Load (nil for a Hub
	// built with New). OAuth-authed servers read and refresh their tokens
	// through it.
	tokens *TokenStore
	// refreshing serializes refreshErrored's lazy re-fetch so concurrent
	// FindMCP/CallMCP calls never run two refreshes at once.
	refreshing bool
}

// New returns an empty Hub using client (defaulting to http.DefaultClient).
func New(client *http.Client) *Hub {
	if client == nil {
		client = http.DefaultClient
	}
	return &Hub{servers: map[string]*Server{}, client: client}
}

// Load reads the MCP config file at path and returns a Hub with every
// configured server's tools fetched (concurrently; see Hub.Refresh). A missing
// file — or an empty path — returns an empty Hub: MCP is optional. A malformed
// config file is an error.
func Load(path string, client *http.Client) (*Hub, error) {
	h := New(client)
	if path == "" {
		return h, nil
	}
	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return h, nil
		}
		return nil, fmt.Errorf("read MCP config %s: %w", path, err)
	}
	var cfg config
	if err := json.Unmarshal(data, &cfg); err != nil {
		return nil, fmt.Errorf("parse MCP config %s: %w", path, err)
	}
	seen := make(map[string]bool, len(cfg.Servers))
	for _, sc := range cfg.Servers {
		if seen[sc.Name] {
			return nil, fmt.Errorf("MCP config: duplicate server name %q", sc.Name)
		}
		seen[sc.Name] = true
		s, err := parseServer(sc)
		if err != nil {
			return nil, err
		}
		h.add(s)
	}
	// OAuth-authed servers share one token store in the user's home. A
	// missing file is an empty store; a malformed one is logged and skipped
	// (bearer servers keep working, and the store is only read, never
	// created, on load).
	if t, err := OpenTokenStore(); err != nil {
		log.Printf("mcp: token store: %v (OAuth servers will be unavailable)", err)
	} else {
		h.setTokens(t)
	}
	h.Refresh(context.Background())
	return h, nil
}

// parseServer validates one config entry and builds its Server.
func parseServer(sc serverConfig) (*Server, error) {
	if strings.TrimSpace(sc.Name) == "" {
		return nil, errors.New("MCP config: server name is required")
	}
	if strings.TrimSpace(sc.URL) == "" {
		return nil, fmt.Errorf("MCP config server %q: url is required", sc.Name)
	}
	if u, err := url.Parse(sc.URL); err != nil || (u.Scheme != "http" && u.Scheme != "https") {
		return nil, fmt.Errorf("MCP config server %q: url must be http(s): %q", sc.Name, sc.URL)
	}
	switch sc.Auth.Type {
	case "", "bearer", "oauth":
	default:
		return nil, fmt.Errorf("MCP config server %q: unsupported auth type %q (supported: \"bearer\", \"oauth\")", sc.Name, sc.Auth.Type)
	}
	timeout := DefaultTimeout
	if sc.TimeoutSeconds > 0 {
		timeout = time.Duration(sc.TimeoutSeconds) * time.Second
	}
	return &Server{
		Name:        sc.Name,
		Description: sc.Description,
		URL:         sc.URL,
		Auth:        Auth{Type: sc.Auth.Type, Token: sc.Auth.Token, Scope: sc.Auth.Scope},
		Timeout:     timeout,
		status:      "pending",
	}, nil
}

// add registers a server, keeping config order.
func (h *Hub) add(s *Server) {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.servers[s.Name] = s
	h.order = append(h.order, s.Name)
}

// setTokens attaches the shared OAuth token store to the hub and every
// registered server.
func (h *Hub) setTokens(t *TokenStore) {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.tokens = t
	for _, s := range h.servers {
		s.tokens = t
	}
}

// Summary returns every server's metadata (name, description, load status,
// tools) in config order, for inclusion in an execution context so the
// server-side hub can expose host-provided MCP servers through FindMCP and
// route CallMCP down the host's exec channel.
func (h *Hub) Summary() []api.MCPServer {
	h.mu.RLock()
	defer h.mu.RUnlock()
	out := make([]api.MCPServer, 0, len(h.order))
	for _, name := range h.order {
		s := h.servers[name]
		status, errMsg := s.Status()
		ms := api.MCPServer{Name: s.Name, Description: s.Description, Status: status, Error: errMsg}
		for _, t := range s.Tools() {
			ms.Tools = append(ms.Tools, api.MCPTool{Name: t.Name, Description: t.Description, InputSchema: t.InputSchema})
		}
		out = append(out, ms)
	}
	return out
}

// Refresh re-fetches every server's tools. Servers are fetched concurrently
// and independently, and failures are recorded per-server, so one dead server
// never blocks the rest (or startup). It must not run concurrently with
// itself; today it is called once at load.
func (h *Hub) Refresh(ctx context.Context) {
	h.mu.RLock()
	servers := make([]*Server, 0, len(h.servers))
	for _, name := range h.order {
		servers = append(servers, h.servers[name])
	}
	h.mu.RUnlock()

	var wg sync.WaitGroup
	for _, s := range servers {
		wg.Add(1)
		go func(s *Server) {
			defer wg.Done()
			log.Printf("mcp: connecting to %s (%s)", s.Name, s.URL)
			s.fetch(ctx, h.client)
			if status, msg := s.Status(); status == "ok" {
				log.Printf("mcp: connected to %s (%d tools)", s.Name, len(s.Tools()))
			} else {
				log.Printf("mcp: failed to connect to %s: %s", s.Name, msg)
			}
		}(s)
	}
	wg.Wait()
	if n := len(servers); n > 0 {
		log.Printf("mcp: loaded %d server(s)", n)
	}
}

// refreshErrored re-fetches tools for servers whose last attempt failed (e.g.
// an OAuth server not yet logged in when the hub loaded). It is called lazily
// from FindMCP and CallMCP so a `porter mcp login` after startup takes effect
// without a restart. At most one refresh runs at a time; a refresh already in
// progress makes callers no-ops.
func (h *Hub) refreshErrored(ctx context.Context) {
	h.mu.Lock()
	if h.refreshing {
		h.mu.Unlock()
		return
	}
	h.refreshing = true
	h.mu.Unlock()
	defer func() {
		h.mu.Lock()
		h.refreshing = false
		h.mu.Unlock()
	}()

	h.mu.RLock()
	stale := make([]*Server, 0, len(h.order))
	for _, name := range h.order {
		if status, _ := h.servers[name].Status(); status == "error" {
			stale = append(stale, h.servers[name])
		}
	}
	h.mu.RUnlock()
	var wg sync.WaitGroup
	for _, s := range stale {
		wg.Add(1)
		go func(s *Server) {
			defer wg.Done()
			s.fetch(ctx, h.client)
		}(s)
	}
	wg.Wait()
}

// Server returns the server with the given name, or nil.
func (h *Hub) Server(name string) *Server {
	h.mu.RLock()
	defer h.mu.RUnlock()
	return h.servers[name]
}

// Names returns the configured server names in config order.
func (h *Hub) Names() []string {
	h.mu.RLock()
	defer h.mu.RUnlock()
	return append([]string(nil), h.order...)
}

// fetch discovers the server's tools: the initialize handshake (which assigns
// the streamable-HTTP session id and negotiates the protocol version), a
// best-effort initialized notification, then tools/list following pagination
// cursors. Failures are recorded in the server's status, never returned: a
// dead or non-MCP server must not fail startup or block other servers.
func (s *Server) fetch(ctx context.Context, client *http.Client) {
	if err := s.handshake(ctx, client); err != nil {
		s.setStatus("error", err.Error())
		return
	}
	tools, err := s.listTools(ctx, client)
	if err != nil {
		s.setStatus("error", err.Error())
		return
	}
	s.setTools(tools)
	s.setStatus("ok", "")
}

func (s *Server) handshake(ctx context.Context, client *http.Client) error {
	// A fresh initialize starts a new streamable-HTTP session: clear any
	// cached session id and negotiated protocol first, so the handshake goes
	// out as a clean session start (a stale Mcp-Session-Id header could make
	// the server reject it outright). handshake is also the recovery path
	// when a session lapses, so this keeps re-initializes honest.
	s.mu.Lock()
	s.sessionID = ""
	s.protocol = ""
	s.mu.Unlock()
	result, header, err := s.call(ctx, client, 1, "initialize", map[string]any{
		"protocolVersion": ProtocolVersion,
		"capabilities":    map[string]any{"tools": map[string]any{}},
		"clientInfo":      map[string]any{"name": "porter", "version": "dev"},
	})
	if err != nil {
		return err
	}
	var init struct {
		ProtocolVersion string `json:"protocolVersion"`
		SessionID       string `json:"sessionId"`
	}
	if err := json.Unmarshal(result, &init); err != nil {
		return fmt.Errorf("decode initialize result: %w", err)
	}
	s.mu.Lock()
	// The session id may come back in the initialize result or as a response
	// header; accept either.
	if id := header.Get("Mcp-Session-Id"); id != "" {
		s.sessionID = id
	} else if init.SessionID != "" {
		s.sessionID = init.SessionID
	}
	if init.ProtocolVersion != "" {
		s.protocol = init.ProtocolVersion
	}
	s.mu.Unlock()
	// The initialized notification is best-effort: stateful servers expect it
	// before other requests, but some servers reject notifications outright,
	// and that alone is not worth failing the server over — tools/list will
	// surface a real problem.
	_ = s.notify(ctx, client, "notifications/initialized", map[string]any{})
	return nil
}

// listToolsMaxPages bounds tools/list pagination so a misbehaving server
// cannot loop forever.
const listToolsMaxPages = 100

func (s *Server) listTools(ctx context.Context, client *http.Client) ([]Tool, error) {
	var out []Tool
	var cursor string
	for id := 2; id < 2+listToolsMaxPages; id++ {
		params := map[string]any{}
		if cursor != "" {
			params["cursor"] = cursor
		}
		result, _, err := s.call(ctx, client, id, "tools/list", params)
		if err != nil {
			return nil, err
		}
		var page struct {
			Tools []struct {
				Name        string         `json:"name"`
				Description string         `json:"description"`
				InputSchema map[string]any `json:"inputSchema"`
			} `json:"tools"`
			NextCursor string `json:"nextCursor"`
		}
		if err := json.Unmarshal(result, &page); err != nil {
			return nil, fmt.Errorf("decode tools/list result: %w", err)
		}
		for _, t := range page.Tools {
			out = append(out, Tool{Name: t.Name, Description: t.Description, InputSchema: t.InputSchema})
		}
		if page.NextCursor == "" {
			return out, nil
		}
		cursor = page.NextCursor
	}
	return nil, fmt.Errorf("tools/list: server returned more than %d pages", listToolsMaxPages)
}
