package mcp

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"
)

// mockOAuth is a minimal OAuth authorization server implementing the pieces
// porter uses: RFC 8414 discovery, RFC 7591 dynamic registration, the token
// endpoint (authorization_code + refresh_token), and revocation. It keeps no
// state beyond counting refreshes.
type mockOAuth struct {
	mu         sync.Mutex
	refreshes  int
	authCode   string   // code the token endpoint accepts
	accessTok  string   // token issued on code exchange
	refreshTok string   // refresh token issued
	asScopes   []string // scopes_supported advertised in RFC 8414 metadata (defaults to mcp:read mcp:write)
}

func (m *mockOAuth) handler() http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.URL.Path == "/.well-known/oauth-authorization-server":
			// Endpoints point back at this test server so the flow is fully
			// self-contained (httptest is plain http).
			base := "http://" + r.Host
			scopes := m.asScopes
			if scopes == nil {
				scopes = []string{"mcp:read", "mcp:write"}
			}
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(map[string]any{
				"issuer":                           base,
				"authorization_endpoint":           base + "/authorize",
				"token_endpoint":                   base + "/token",
				"registration_endpoint":            base + "/register",
				"revocation_endpoint":              base + "/revoke",
				"scopes_supported":                 scopes,
				"code_challenge_methods_supported": []string{"S256"},
			})
		case r.URL.Path == "/register":
			var body map[string]any
			_ = json.NewDecoder(r.Body).Decode(&body)
			if body["redirect_uris"] == nil {
				http.Error(w, `{"error":"invalid_client_metadata"}`, http.StatusBadRequest)
				return
			}
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(map[string]any{
				"client_id":                  "mock-client",
				"token_endpoint_auth_method": "none",
				"redirect_uris":              body["redirect_uris"],
			})
		case r.URL.Path == "/token":
			if err := r.ParseForm(); err != nil {
				http.Error(w, "bad form", http.StatusBadRequest)
				return
			}
			if r.PostForm.Get("client_id") == "" {
				http.Error(w, `{"error":"invalid_client"}`, http.StatusUnauthorized)
				return
			}
			m.mu.Lock()
			defer m.mu.Unlock()
			switch r.PostForm.Get("grant_type") {
			case "authorization_code":
				if r.PostForm.Get("code") == "" || r.PostForm.Get("code_verifier") == "" || r.PostForm.Get("redirect_uri") == "" {
					http.Error(w, `{"error":"invalid_grant"}`, http.StatusBadRequest)
					return
				}
				w.Header().Set("Content-Type", "application/json")
				_ = json.NewEncoder(w).Encode(map[string]any{
					"access_token":  m.accessTok,
					"token_type":    "Bearer",
					"expires_in":    3600,
					"refresh_token": m.refreshTok,
					"scope":         r.PostForm.Get("scope"),
				})
			case "refresh_token":
				if r.PostForm.Get("refresh_token") == "" {
					http.Error(w, `{"error":"invalid_grant"}`, http.StatusBadRequest)
					return
				}
				m.refreshes++
				w.Header().Set("Content-Type", "application/json")
				_ = json.NewEncoder(w).Encode(map[string]any{
					"access_token":  fmt.Sprintf("fresh-%d", m.refreshes),
					"token_type":    "Bearer",
					"expires_in":    3600,
					"refresh_token": fmt.Sprintf("ref-%d", m.refreshes),
				})
			default:
				http.Error(w, `{"error":"unsupported_grant_type"}`, http.StatusBadRequest)
			}
		case r.URL.Path == "/revoke":
			w.WriteHeader(http.StatusOK)
		default:
			http.NotFound(w, r)
		}
	})
}

func TestTokenStoreRoundTrip(t *testing.T) {
	path := filepath.Join(t.TempDir(), "tokens.json")
	st, err := OpenTokenStoreAt(path)
	if err != nil {
		t.Fatalf("OpenTokenStoreAt: %v", err)
	}
	if got := st.Get("https://s"); got != nil {
		t.Fatalf("empty store Get = %+v", got)
	}
	entry := &TokenEntry{
		Server: "https://s", ClientID: "c", AccessToken: "a",
		RefreshToken: "r", TokenURL: "https://as/token", ExpiresAt: 123,
	}
	if err := st.Put(entry); err != nil {
		t.Fatalf("Put: %v", err)
	}
	if got := st.Get("https://s"); got == nil || got.AccessToken != "a" || got.ClientID != "c" {
		t.Fatalf("Get after Put = %+v", got)
	}
	// A second store over the same file sees the persisted entry.
	st2, err := OpenTokenStoreAt(path)
	if err != nil {
		t.Fatalf("reopen: %v", err)
	}
	if got := st2.Get("https://s"); got == nil || got.RefreshToken != "r" {
		t.Fatalf("reopened Get = %+v", got)
	}
	if err := st.Delete("https://s"); err != nil {
		t.Fatalf("Delete: %v", err)
	}
	if got := st.Get("https://s"); got != nil {
		t.Fatalf("Get after Delete = %+v", got)
	}
	// A fresh open sees the deletion persisted.
	st3, err := OpenTokenStoreAt(path)
	if err != nil {
		t.Fatalf("reopen after delete: %v", err)
	}
	if got := st3.Get("https://s"); got != nil {
		t.Fatalf("reopened Get after Delete = %+v", got)
	}
	// Idempotent delete of an unknown server.
	if err := st.Delete("https://nope"); err != nil {
		t.Fatalf("Delete unknown: %v", err)
	}
	// The file is written 0600.
	fi, err := fileStat(path)
	if err != nil {
		t.Fatal(err)
	}
	if fi.Mode().Perm() != 0o600 {
		t.Errorf("token file mode = %v, want 0600", fi.Mode().Perm())
	}
}

func TestLoginFlow(t *testing.T) {
	as := &mockOAuth{authCode: "code-1", accessTok: "tok-1", refreshTok: "ref-1"}
	asSrv := httptest.NewServer(as.handler())
	defer asSrv.Close()

	store, err := OpenTokenStoreAt(filepath.Join(t.TempDir(), "tokens.json"))
	if err != nil {
		t.Fatal(err)
	}
	var stdout strings.Builder
	var opened string
	openBrowser := func(u string) error {
		opened = u
		// Simulate the browser completing consent: parse the authorize URL,
		// then hit the loopback redirect with the authorization code and the
		// state echoed back (the AS's redirect would carry both).
		au, err := url.Parse(u)
		if err != nil {
			return err
		}
		redirect := au.Query().Get("redirect_uri")
		resp, err := http.Get(redirect + "?code=code-1&state=" + url.QueryEscape(au.Query().Get("state")))
		if err != nil {
			return err
		}
		resp.Body.Close()
		return nil
	}

	// The MCP server URL is only used for discovery and as the token key; the
	// mock AS serves discovery from any host, so use the AS server itself.
	if err := login(context.Background(), http.DefaultClient, asSrv.URL, "retool", "mcp:read", &stdout, openBrowser, store); err != nil {
		t.Fatalf("login: %v", err)
	}
	if !strings.Contains(opened, "response_type=code") ||
		!strings.Contains(opened, "code_challenge_method=S256") ||
		!strings.Contains(opened, "code_challenge=") ||
		!strings.Contains(opened, "scope=mcp%3Aread") ||
		!strings.Contains(opened, "client_id=mock-client") {
		t.Errorf("authorize URL missing expected params: %s", opened)
	}
	// The authorize URL must carry a CSRF state with enough entropy for
	// servers like Greptile that reject states shorter than 8 characters.
	au, err := url.Parse(opened)
	if err != nil {
		t.Fatalf("parse authorize URL: %v", err)
	}
	if st := au.Query().Get("state"); len(st) < 8 {
		t.Errorf("authorize URL state = %q, want at least 8 characters", st)
	}
	e := store.Get(asSrv.URL)
	if e == nil {
		t.Fatal("no token stored after login")
	}
	if e.AccessToken != "tok-1" || e.RefreshToken != "ref-1" || e.ClientID != "mock-client" {
		t.Errorf("stored entry = %+v", e)
	}
	if e.TokenURL != asSrv.URL+"/token" {
		t.Errorf("token URL = %q", e.TokenURL)
	}
	if e.ExpiresAt < time.Now().Unix() {
		t.Errorf("expires_at not set in the future: %d", e.ExpiresAt)
	}

	// logout removes the entry and hits revocation.
	if err := logout(context.Background(), http.DefaultClient, asSrv.URL, store); err != nil {
		t.Fatalf("logout: %v", err)
	}
	if store.Get(asSrv.URL) != nil {
		t.Error("token still stored after logout")
	}
}

// TestProtectedResourceCandidates proves RFC 9728 discovery tries the root
// well-known URI first, then RFC 9728 §3.1's path-inserted form.
func TestProtectedResourceCandidates(t *testing.T) {
	got := protectedResourceCandidates("https://api.example.com/mcp")
	want := []string{
		"https://api.example.com/.well-known/oauth-protected-resource",
		"https://api.example.com/.well-known/oauth-protected-resource/mcp",
	}
	if !slices.Equal(got, want) {
		t.Errorf("candidates for /mcp = %v, want %v", got, want)
	}
	got = protectedResourceCandidates("https://api.example.com")
	want = []string{"https://api.example.com/.well-known/oauth-protected-resource"}
	if !slices.Equal(got, want) {
		t.Errorf("candidates for root = %v, want %v", got, want)
	}
}

// TestLoginViaProtectedResourceMetadata proves porter follows the MCP spec's
// OAuth discovery order:when the MCP resource server (like Greptile's
// api.greptile.com/mcp) publishes only RFC 9728 protected-resource metadata
// naming an authorization server on a different origin (auth.greptile.com),
// RFC 8414 discovery must happen against that advertised origin, not the
// resource's own (which 404s).
func TestLoginViaProtectedResourceMetadata(t *testing.T) {
	as := &mockOAuth{authCode: "code-2", accessTok: "tok-2", refreshTok: "ref-2"}
	asSrv := httptest.NewServer(as.handler())
	defer asSrv.Close()

	// Resource server: publishes ONLY RFC 9728 metadata, mirroring the
	// Greptile split between api.greptile.com and auth.greptile.com.No RFC 8414
	// document exists at this origin at all.

	var mcpSrv *httptest.Server
	mcpSrv = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/.well-known/oauth-protected-resource" ||
			r.URL.Path == "/.well-known/oauth-protected-resource/mcp" {
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(map[string]any{
				"resource":                 mcpSrv.URL + "/mcp",
				"authorization_servers":    []string{asSrv.URL},
				"scopes_supported":         []string{"read", "write"},
				"bearer_methods_supported": []string{"header"},
			})
			return
		}
		http.NotFound(w, r)
	}))
	defer mcpSrv.Close()

	store, err := OpenTokenStoreAt(filepath.Join(t.TempDir(), "tokens.json"))
	if err != nil {
		t.Fatal(err)
	}
	var stdout strings.Builder
	var opened string
	openBrowser := func(u string) error {
		opened = u
		au, err := url.Parse(u)
		if err != nil {
			return err
		}
		// Simulate the browser completing consent (the AS has no /authorize in
		// this mock; the flow only needs the loopback callback), echoing the
		// state back as the AS's redirect would.
		resp, err := http.Get(au.Query().Get("redirect_uri") + "?code=code-2&state=" + url.QueryEscape(au.Query().Get("state")))
		if err != nil {
			return err
		}
		resp.Body.Close()
		return nil
	}

	mcpURL := mcpSrv.URL + "/mcp" // the resource carries a path, like /mcp

	if err := login(context.Background(), http.DefaultClient, mcpURL, "greptile", "", &stdout, openBrowser, store); err != nil {
		t.Fatalf("login: %v", err)
	}
	e := store.Get(mcpURL)
	if e == nil {
		t.Fatal("no token stored after login")
	}
	if e.AccessToken != "tok-2" || e.RefreshToken != "ref-2" || e.ClientID != "mock-client" {
		t.Errorf("stored entry = %+v", e)
	}
	if e.TokenURL != asSrv.URL+"/token" {
		t.Errorf("token URL = %q, want the authorization server's %q (shows RFC 8414 came from the advertised AS, not the resource)", e.TokenURL, asSrv.URL+"/token")
	}
	// With no scope configured, the login must request the scopes the resource
	// itself advertises (RFC 9728 protected-resource metadata), so the token
	// is not issued empty. The mock advertises "read write".
	if !strings.Contains(opened, "scope=read+write") {
		t.Errorf("authorize URL missing resource-advertised scope: %s", opened)
	}
	if e.Scope != "read write" {
		t.Errorf("stored scope = %q, want resource-advertised %q", e.Scope, "read write")
	}
	if !strings.Contains(opened, "client_id=mock-client") {
		t.Errorf("authorize URL missing registered client: %s", opened)
	}
}

// TestLoginRejectsStateMismatch proves the loopback callback is rejected when
// the state echoed back doesn't match the one sent in the authorize URL (the
// CSRF protection state exists for): a callback with a missing or wrong state
// must never yield a token.
func TestLoginRejectsStateMismatch(t *testing.T) {
	as := &mockOAuth{authCode: "code-3", accessTok: "tok-3", refreshTok: "ref-3"}
	asSrv := httptest.NewServer(as.handler())
	defer asSrv.Close()

	store, err := OpenTokenStoreAt(filepath.Join(t.TempDir(), "tokens.json"))
	if err != nil {
		t.Fatal(err)
	}
	var stdout strings.Builder
	openBrowser := func(u string) error {
		au, err := url.Parse(u)
		if err != nil {
			return err
		}
		// A hostile or buggy redirect: the state is dropped (or wrong).
		resp, err := http.Get(au.Query().Get("redirect_uri") + "?code=code-3")
		if err != nil {
			return err
		}
		resp.Body.Close()
		return nil
	}

	err = login(context.Background(), http.DefaultClient, asSrv.URL, "retool", "", &stdout, openBrowser, store)
	if err == nil {
		t.Fatal("login succeeded despite a state mismatch in the callback")
	}
	if !strings.Contains(err.Error(), "state mismatch") {
		t.Errorf("login error = %q, want a state mismatch error", err)
	}
	if store.Get(asSrv.URL) != nil {
		t.Error("token stored despite state mismatch")
	}
}

// oauthMCPServer returns a mock MCP server that requires the given bearer
// token (used to prove which access token the hub sent).
func oauthMCPServer(t *testing.T, token string) *httptest.Server {
	t.Helper()
	m := &mockMCP{tools: []map[string]any{{"name": "whoami", "description": "Who am I"}}, token: token}
	return httptest.NewServer(m.handler())
}

// oauthHub builds a Hub with one OAuth-authed server and an injected token
// store, putting the given entry in it.
func oauthHub(t *testing.T, mcpURL, token string, entry *TokenEntry) (*Hub, *TokenStore) {
	t.Helper()
	cfg := writeConfig(t, map[string]any{
		"name": "retool", "description": "Retool", "url": mcpURL,
		"auth": map[string]any{"type": "oauth", "scope": "mcp:read"},
	})
	h, err := Load(cfg, nil)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	store, err := OpenTokenStoreAt(filepath.Join(t.TempDir(), "tokens.json"))
	if err != nil {
		t.Fatal(err)
	}
	if entry != nil {
		if err := store.Put(entry); err != nil {
			t.Fatal(err)
		}
	}
	h.setTokens(store)
	return h, store
}

// TestOAuthRefreshOnExpiry proves an expired stored token is refreshed (and
// persisted) before the MCP call, and the fresh token is what's sent.
func TestOAuthRefreshOnExpiry(t *testing.T) {
	as := &mockOAuth{accessTok: "tok-1", refreshTok: "ref-1"}
	asSrv := httptest.NewServer(as.handler())
	defer asSrv.Close()

	// The MCP server accepts only the token the refresh issues ("fresh-1").
	mcpSrv := oauthMCPServer(t, "fresh-1")
	defer mcpSrv.Close()

	entry := &TokenEntry{
		Server: mcpSrv.URL, ClientID: "mock-client",
		AccessToken: "stale", RefreshToken: "ref-1",
		TokenURL:  asSrv.URL + "/token",
		ExpiresAt: time.Now().Unix() - 60, // expired
	}
	h, store := oauthHub(t, mcpSrv.URL, "", entry)

	out, err := h.Run(context.Background(), CallTool, []byte(`{"server_name":"retool","tool_name":"whoami"}`))
	if err != nil {
		t.Fatalf("CallMCP: %v", err)
	}
	data, _ := io.ReadAll(out)
	_ = out.Close()
	if !strings.Contains(string(data), "whoami") {
		t.Errorf("call result = %q, want the tool result", data)
	}
	if got := store.Get(mcpSrv.URL).AccessToken; got != "fresh-1" {
		t.Errorf("stored access token after refresh = %q, want fresh-1", got)
	}
}

// TestOAuth401Retry proves a rejected token triggers a refresh and one retry.
func TestOAuth401Retry(t *testing.T) {
	as := &mockOAuth{accessTok: "tok-1", refreshTok: "ref-1"}
	asSrv := httptest.NewServer(as.handler())
	defer asSrv.Close()

	// The MCP server rejects the stored token; only the refreshed one works.
	mcpSrv := oauthMCPServer(t, "fresh-1")
	defer mcpSrv.Close()

	entry := &TokenEntry{
		Server: mcpSrv.URL, ClientID: "mock-client",
		AccessToken: "revoked", RefreshToken: "ref-1",
		TokenURL:  asSrv.URL + "/token",
		ExpiresAt: time.Now().Unix() + 3600, // not expired; the server rejects it anyway
	}
	h, store := oauthHub(t, mcpSrv.URL, "", entry)

	out, err := h.Run(context.Background(), CallTool, []byte(`{"server_name":"retool","tool_name":"whoami"}`))
	if err != nil {
		t.Fatalf("CallMCP: %v", err)
	}
	data, _ := io.ReadAll(out)
	_ = out.Close()
	if !strings.Contains(string(data), "whoami") {
		t.Errorf("call result = %q, want the tool result after retry", data)
	}
	if got := store.Get(mcpSrv.URL).AccessToken; got != "fresh-1" {
		t.Errorf("stored access token after 401 retry = %q, want fresh-1", got)
	}
}

// TestOAuthNotLoggedIn proves a missing token surfaces a clear login hint in
// the streamed result (remote failures are reported as content, like shell
// command failures), not a hang or a tokenless request.
func TestOAuthNotLoggedIn(t *testing.T) {
	mcpSrv := oauthMCPServer(t, "")
	defer mcpSrv.Close()
	h, _ := oauthHub(t, mcpSrv.URL, "", nil)
	out, err := h.Run(context.Background(), CallTool, []byte(`{"server_name":"retool","tool_name":"whoami"}`))
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	data, _ := io.ReadAll(out)
	_ = out.Close()
	if !strings.Contains(string(data), "porter mcp login") {
		t.Fatalf("result = %q, want a login hint", data)
	}
}

// TestWithOfflineAccess pins down when login appends the offline_access scope:
// only when the authorization server advertises it, and never twice.
func TestWithOfflineAccess(t *testing.T) {
	tests := []struct {
		name      string
		scope     string
		supported []string
		want      string
	}{
		{
			name:      "appends when advertised and missing",
			scope:     "read write",
			supported: []string{"offline_access", "offline", "openid", "read"},
			want:      "read write offline_access",
		},
		{
			name:      "already present is left alone",
			scope:     "read write offline_access",
			supported: []string{"offline_access"},
			want:      "read write offline_access",
		},
		{
			name:      "not advertised is not requested",
			scope:     "read write",
			supported: []string{"openid", "read"},
			want:      "read write",
		},
		{
			name:      "empty scope stays empty",
			scope:     "",
			supported: []string{"offline_access"},
			want:      "",
		},
		{
			name:      "no supported scopes leaves scope alone",
			scope:     "mcp:read mcp:write",
			supported: nil,
			want:      "mcp:read mcp:write",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := withOfflineAccess(tt.scope, tt.supported); got != tt.want {
				t.Errorf("withOfflineAccess(%q, %v) = %q, want %q", tt.scope, tt.supported, got, tt.want)
			}
		})
	}
}

// TestLoginRequestsOfflineAccessWhenAdvertised mirrors the Greptile shape end
// to end: the authorization server advertises offline_access (and, like
// Greptile's, omits write from its own list), the MCP resource advertises
// read write on a different origin, and no scope is configured. porter must
// request the resource scopes plus offline_access so the AS issues a refresh
// token — without offline_access, Greptile returns no refresh token and the
// stored token can never be renewed.
func TestLoginRequestsOfflineAccessWhenAdvertised(t *testing.T) {
	as := &mockOAuth{
		authCode:   "code-4",
		accessTok:  "tok-4",
		refreshTok: "ref-4",
		asScopes:   []string{"offline_access", "offline", "openid", "read"},
	}
	asSrv := httptest.NewServer(as.handler())
	defer asSrv.Close()

	var mcpSrv *httptest.Server
	mcpSrv = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/.well-known/oauth-protected-resource" ||
			r.URL.Path == "/.well-known/oauth-protected-resource/mcp" {
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(map[string]any{
				"resource":                 mcpSrv.URL + "/mcp",
				"authorization_servers":    []string{asSrv.URL},
				"scopes_supported":         []string{"read", "write"},
				"bearer_methods_supported": []string{"header"},
			})
			return
		}
		http.NotFound(w, r)
	}))
	defer mcpSrv.Close()

	store, err := OpenTokenStoreAt(filepath.Join(t.TempDir(), "tokens.json"))
	if err != nil {
		t.Fatal(err)
	}
	var stdout strings.Builder
	var opened string
	openBrowser := func(u string) error {
		opened = u
		au, err := url.Parse(u)
		if err != nil {
			return err
		}
		// Simulate the browser completing consent, echoing state back.
		resp, err := http.Get(au.Query().Get("redirect_uri") + "?code=code-4&state=" + url.QueryEscape(au.Query().Get("state")))
		if err != nil {
			return err
		}
		resp.Body.Close()
		return nil
	}

	mcpURL := mcpSrv.URL + "/mcp"
	if err := login(context.Background(), http.DefaultClient, mcpURL, "greptile", "", &stdout, openBrowser, store); err != nil {
		t.Fatalf("login: %v", err)
	}
	if !strings.Contains(opened, "scope=read+write+offline_access") {
		t.Errorf("authorize URL missing offline_access alongside resource scopes: %s", opened)
	}
	e := store.Get(mcpURL)
	if e == nil {
		t.Fatal("no token stored after login")
	}
	if e.Scope != "read write offline_access" {
		t.Errorf("stored scope = %q, want %q", e.Scope, "read write offline_access")
	}
	if e.RefreshToken != "ref-4" {
		t.Errorf("refresh token = %q, want %q (the point of offline_access)", e.RefreshToken, "ref-4")
	}
}
