package mcp

import (
	"bytes"
	"context"
	"crypto/rand"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"sync"
	"time"
)

// This file implements the OAuth 2.0 client side of the MCP auth spec:
// dynamic client registration (RFC 7591), the authorization-code flow with
// PKCE (RFC 7636) over an ephemeral loopback redirect, and refresh-token
// rotation. Tokens are stored per server in ~/.porter/mcp/tokens.json
// (0600); the interactive browser flow runs only in `porter mcp login`, so
// the execution host daemon never opens a browser — it reads and refreshes
// stored tokens.

// oauthDir is the directory under the user's home holding OAuth state.
// Client registrations are not persisted: clients are registered per login
// (cheap, public, no secret) and the client_id rides with the stored token
// for refresh.
const oauthDir = ".porter/mcp"

// tokensPath returns ~/.porter/mcp/tokens.json.
func tokensPath() (string, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(home, oauthDir, "tokens.json"), nil
}

// TokenEntry is one server's stored OAuth state: the access token, the
// refresh token, and what refresh needs — the public client id and the token
// endpoint (discovered at login). ExpiresAt is the access token's expiry in
// epoch seconds, 0 when unknown.
type TokenEntry struct {
	Server       string `json:"server"` // the MCP server URL this entry is for
	ClientID     string `json:"client_id"`
	AccessToken  string `json:"access_token"`
	RefreshToken string `json:"refresh_token"`
	TokenURL     string `json:"token_endpoint"`
	ExpiresAt    int64  `json:"expires_at"`
	Scope        string `json:"scope,omitempty"`
}

// TokenStore reads and writes the token file. Reads are side-effect free (a
// missing file is an empty store); writes create the file and directory as
// needed with 0600/0700 permissions. Safe for concurrent use.
type TokenStore struct {
	mu     sync.Mutex
	path   string
	tokens map[string]*TokenEntry
}

// OpenTokenStore loads ~/.porter/mcp/tokens.json. A missing file is an empty
// store with no error; a malformed file is an error so callers can log it and
// fall back to an empty store rather than failing startup.
func OpenTokenStore() (*TokenStore, error) {
	path, err := tokensPath()
	if err != nil {
		return nil, err
	}
	return OpenTokenStoreAt(path)
}

// OpenTokenStoreAt loads the token store at an explicit path (for tests).
func OpenTokenStoreAt(path string) (*TokenStore, error) {
	t := &TokenStore{path: path, tokens: map[string]*TokenEntry{}}
	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return t, nil
		}
		return nil, err
	}
	var file struct {
		Servers []*TokenEntry `json:"servers"`
	}
	if err := json.Unmarshal(data, &file); err != nil {
		return nil, fmt.Errorf("parse token store %s: %w", path, err)
	}
	for _, e := range file.Servers {
		if e != nil && e.Server != "" {
			t.tokens[e.Server] = e
		}
	}
	return t, nil
}

// Get returns the stored entry for a server, or nil.
func (t *TokenStore) Get(server string) *TokenEntry {
	t.mu.Lock()
	defer t.mu.Unlock()
	return t.tokens[server]
}

// Put stores (or replaces) the entry for its server, persisting the file.
func (t *TokenStore) Put(e *TokenEntry) error {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.tokens[e.Server] = e
	return t.saveLocked()
}

// Delete removes the entry for a server, persisting the file. It is
// idempotent: an unknown server is a no-op success.
func (t *TokenStore) Delete(server string) error {
	t.mu.Lock()
	defer t.mu.Unlock()
	if _, ok := t.tokens[server]; !ok {
		return nil
	}
	delete(t.tokens, server)
	return t.saveLocked()
}

// saveLocked writes the store deterministically (servers sorted by URL).
func (t *TokenStore) saveLocked() error {
	servers := make([]string, 0, len(t.tokens))
	for s := range t.tokens {
		servers = append(servers, s)
	}
	sort.Strings(servers)
	file := struct {
		Servers []*TokenEntry `json:"servers"`
	}{Servers: make([]*TokenEntry, 0, len(servers))}
	for _, s := range servers {
		file.Servers = append(file.Servers, t.tokens[s])
	}
	data, err := json.MarshalIndent(file, "", "  ")
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(t.path), 0o700); err != nil {
		return err
	}
	return os.WriteFile(t.path, data, 0o600)
}

// wellKnown is the RFC 8414 authorization-server metadata document. Only the
// fields porter uses are decoded.
type wellKnown struct {
	Issuer                string   `json:"issuer"`
	AuthorizationEndpoint string   `json:"authorization_endpoint"`
	TokenEndpoint         string   `json:"token_endpoint"`
	DeviceAuthEndpoint    string   `json:"device_authorization_endpoint"`
	RegistrationEndpoint  string   `json:"registration_endpoint"`
	RevocationEndpoint    string   `json:"revocation_endpoint"`
	ScopesSupported       []string `json:"scopes_supported"`
	CodeChallengeMethods  []string `json:"code_challenge_methods_supported"`
}

// protectedResource is the RFC 9728 protected-resource metadata document.
type protectedResource struct {
	Resource               string   `json:"resource"`
	AuthorizationServers   []string `json:"authorization_servers"`
	ScopesSupported        []string `json:"scopes_supported"`
	BearerMethodsSupported []string `json:"bearer_methods_supported"`
}

// protectedResourceCandidates enumerates the well-known URIs for RFC 9728
// discovery against serverURL:the root form, then RFC 9728 §3.1's
// path-inserted form when the MCP URL carries a path (e.g. for
// https://host/mcp the path form is https://host/.well-known/oauth-protected-resource/mcp).
func protectedResourceCandidates(serverURL string) []string {
	u, err := url.Parse(serverURL)
	if err != nil {
		return nil
	}
	root := *u
	root.Path = "/.well-known/oauth-protected-resource"
	root.RawQuery = ""
	root.Fragment = ""
	candidates := []string{root.String()}
	if p := strings.Trim(u.Path, "/"); p != "" {
		pathForm := *u
		pathForm.Path = "/.well-known/oauth-protected-resource/" + p
		pathForm.RawQuery = ""
		pathForm.Fragment = ""
		candidates = append(candidates, pathForm.String())
	}
	return candidates
}

// discoverProtectedResource fetches the RFC 9728 protected-resource metadata
// for the MCP server, or returns nil without error when the server publishes none
// (pre-RFC 9728 servers). It tries the root well-known URI first, then
// RFC 9728 §3.1's path-inserted form.The resource's authorization_servers
// names the authorization server to discover next (the MCP spec's discovery
// chain: RFC 9728, then RFC 8414 against the advertised server).
func discoverProtectedResource(ctx context.Context, client *http.Client, serverURL string) (*protectedResource, error) {
	for _, u := range protectedResourceCandidates(serverURL) {
		req, err := http.NewRequestWithContext(ctx, http.MethodGet, u, nil)
		if err != nil {
			return nil, err
		}
		resp, err := client.Do(req)
		if err != nil {
			return nil, err
		}
		raw, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
		resp.Body.Close()
		if resp.StatusCode != http.StatusOK {
			continue
		}
		var m protectedResource
		if err := json.Unmarshal(raw, &m); err != nil {
			continue // not an RFC 9728 document; try the next form
		}
		if len(m.AuthorizationServers) > 0 || m.Resource != "" {
			return &m, nil
		}
	}
	return nil, nil // no protected-resource metadata published
}

// discoverWellKnown fetches the RFC 8414 authorization-server metadata from
// the given server's origin. The well-known path is relative to the server's
// origin: https://<host>/.well-known/oauth-authorization-server.
func discoverWellKnown(ctx context.Context, client *http.Client, server string) (*wellKnown, error) {
	u, err := url.Parse(server)
	if err != nil {
		return nil, fmt.Errorf("parse server URL: %w", err)
	}
	u.Path = "/.well-known/oauth-authorization-server"
	u.RawQuery = ""
	u.Fragment = ""
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, u.String(), nil)
	if err != nil {
		return nil, err
	}
	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("discover OAuth metadata: unexpected status %s", resp.Status)
	}
	var meta wellKnown
	if err := json.NewDecoder(resp.Body).Decode(&meta); err != nil {
		return nil, fmt.Errorf("decode OAuth metadata: %w", err)
	}
	if meta.AuthorizationEndpoint == "" || meta.TokenEndpoint == "" {
		return nil, errors.New("OAuth metadata missing authorization or token endpoint")
	}
	return &meta, nil
}

// discover fetches the OAuth authorization-server metadata for an MCP server,
// following the MCP spec's discovery order: RFC 9728 protected-resource
// metadata from the resource (whose authorization_servers names the true
// authorization server), falling back to RFC 8414 discovery against the
// resource's own origin for servers that don't publish protected-resource metadata
// (including the common single-origin layout where both documents live at the
// same host). This matters for servers like Greptile, where the MCP resource
// (api.greptile.com) and the authorization server (auth.greptile.com) live on
// different origins and only the RFC 9728 document exists at the resource's
// origin.
func discover(ctx context.Context, client *http.Client, serverURL string) (*wellKnown, error) {
	if prm, err := discoverProtectedResource(ctx, client, serverURL); err != nil {
		return nil, err
	} else if prm != nil {
		for _, asURL := range prm.AuthorizationServers {
			if meta, err := discoverWellKnown(ctx, client, asURL); err == nil {
				return meta, nil
			}
		}
	}
	// Legacy fallback: pre-RFC 9728 servers and single-origin servers serve
	// RFC 8414 directly against the resource's origin.This is where porter
	// always looked before.
	return discoverWellKnown(ctx, client, serverURL)
}

// registerClient dynamically registers a public OAuth client (RFC 7591) with
// the authorization server, requesting a loopback redirect for the
// authorization-code flow. Servers like Retool's accept registration without
// an initial access token and return a public client (token_endpoint_auth_method
// "none", no secret).
func registerClient(ctx context.Context, client *http.Client, meta *wellKnown, name, redirectURI string) (string, error) {
	if meta.RegistrationEndpoint == "" {
		return "", errors.New("OAuth server does not support dynamic client registration")
	}
	body := map[string]any{
		"client_name":                name,
		"redirect_uris":              []string{redirectURI},
		"token_endpoint_auth_method": "none",
		"grant_types":                []string{"authorization_code", "refresh_token"},
		"response_types":             []string{"code"},
	}
	data, _ := json.Marshal(body)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, meta.RegistrationEndpoint, bytes.NewReader(data))
	if err != nil {
		return "", err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := client.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return "", fmt.Errorf("client registration failed: %s: %s", resp.Status, strings.TrimSpace(string(raw)))
	}
	var reg struct {
		ClientID string `json:"client_id"`
	}
	if err := json.Unmarshal(raw, &reg); err != nil {
		return "", fmt.Errorf("decode client registration: %w", err)
	}
	if reg.ClientID == "" {
		return "", errors.New("client registration returned no client_id")
	}
	return reg.ClientID, nil
}

// pkcePair generates a random code verifier and its S256 challenge.
func pkcePair() (verifier, challenge string, err error) {
	buf := make([]byte, 64)
	if _, err := rand.Read(buf); err != nil {
		return "", "", err
	}
	verifier = base64.RawURLEncoding.EncodeToString(buf)
	sum := sha256.Sum256([]byte(verifier))
	challenge = base64.RawURLEncoding.EncodeToString(sum[:])
	return verifier, challenge, nil
}

// stateValue generates a random OAuth state value for CSRF protection
// (RFC 6749 §10.12). 16 random bytes base64url-encoded is 22 characters —
// comfortably past the 8-character minimum servers like Greptile enforce.
func stateValue() (string, error) {
	buf := make([]byte, 16)
	if _, err := rand.Read(buf); err != nil {
		return "", err
	}
	return base64.RawURLEncoding.EncodeToString(buf), nil
}

// buildAuthorizeURL assembles the authorization endpoint URL with PKCE and a
// CSRF state value. state must be echoed back on the redirect; waitForCode
// rejects callbacks that don't.
func buildAuthorizeURL(endpoint, clientID, redirect, challenge, scope, state string) (string, error) {
	u, err := url.Parse(endpoint)
	if err != nil {
		return "", err
	}
	q := u.Query()
	q.Set("client_id", clientID)
	q.Set("response_type", "code")
	q.Set("redirect_uri", redirect)
	q.Set("code_challenge", challenge)
	q.Set("code_challenge_method", "S256")
	q.Set("state", state)
	if scope != "" {
		q.Set("scope", scope)
	}
	u.RawQuery = q.Encode()
	return u.String(), nil
}

// tokenResponse is the token endpoint's response for both the code exchange
// and refresh.
type tokenResponse struct {
	AccessToken  string `json:"access_token"`
	TokenType    string `json:"token_type"`
	ExpiresIn    int64  `json:"expires_in"`
	RefreshToken string `json:"refresh_token"`
	Scope        string `json:"scope"`
	Error        string `json:"error"`
	ErrorDesc    string `json:"error_description"`
}

func (t tokenResponse) err() error {
	if t.Error == "" {
		return nil
	}
	if t.ErrorDesc != "" {
		return fmt.Errorf("OAuth %s: %s", t.Error, t.ErrorDesc)
	}
	return fmt.Errorf("OAuth %s", t.Error)
}

// doTokenRequest POSTs form-encoded params to the token endpoint and decodes
// the response. Public clients send client_id in the body (the server
// advertises token_endpoint_auth_method "none", so there is no secret).
func doTokenRequest(ctx context.Context, client *http.Client, tokenURL, clientID string, form url.Values, scope string) (*tokenResponse, error) {
	if scope != "" && form.Get("scope") == "" {
		form.Set("scope", scope)
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, tokenURL, strings.NewReader(form.Encode()))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, fmt.Errorf("token request failed: %s: %s", resp.Status, strings.TrimSpace(string(raw)))
	}
	var tr tokenResponse
	if err := json.Unmarshal(raw, &tr); err != nil {
		return nil, fmt.Errorf("decode token response: %w", err)
	}
	if err := tr.err(); err != nil {
		return nil, err
	}
	if tr.AccessToken == "" {
		return nil, errors.New("token response missing access_token")
	}
	return &tr, nil
}

// exchangeCode trades the authorization code for a token entry.
func exchangeCode(ctx context.Context, client *http.Client, tokenURL, serverURL, clientID, code, redirectURI, verifier, scope string) (*TokenEntry, error) {
	form := url.Values{
		"grant_type":    {"authorization_code"},
		"code":          {code},
		"redirect_uri":  {redirectURI},
		"client_id":     {clientID},
		"code_verifier": {verifier},
	}
	tr, err := doTokenRequest(ctx, client, tokenURL, clientID, form, scope)
	if err != nil {
		return nil, err
	}
	e := &TokenEntry{
		Server:       serverURL,
		ClientID:     clientID,
		AccessToken:  tr.AccessToken,
		RefreshToken: tr.RefreshToken,
		TokenURL:     tokenURL,
		Scope:        tr.Scope,
	}
	if tr.ExpiresIn > 0 {
		e.ExpiresAt = time.Now().Unix() + tr.ExpiresIn
	}
	return e, nil
}

// refreshToken refreshes an entry's access token in place from its refresh
// token, rotating the refresh token when the server issues a new one.
func refreshToken(ctx context.Context, client *http.Client, e *TokenEntry) error {
	form := url.Values{
		"grant_type":    {"refresh_token"},
		"refresh_token": {e.RefreshToken},
		"client_id":     {e.ClientID},
	}
	tr, err := doTokenRequest(ctx, client, e.TokenURL, e.ClientID, form, e.Scope)
	if err != nil {
		return err
	}
	e.AccessToken = tr.AccessToken
	if tr.RefreshToken != "" {
		e.RefreshToken = tr.RefreshToken
	}
	if tr.ExpiresIn > 0 {
		e.ExpiresAt = time.Now().Unix() + tr.ExpiresIn
	}
	if tr.Scope != "" {
		e.Scope = tr.Scope
	}
	return nil
}

// Login runs the interactive OAuth authorization-code flow against the MCP
// server at serverURL: discover metadata, dynamically register a public
// client, open the authorization URL in a browser with an ephemeral loopback
// redirect and PKCE, exchange the code, and store the tokens. scope may be ""
// to use the server's defaults. It blocks until the user authorizes (bounded
// by loginTimeout). openBrowser, when nil, uses the platform's opener.
func Login(ctx context.Context, client *http.Client, serverURL, name, scope string, stdout io.Writer, openBrowser func(string) error) error {
	store, err := OpenTokenStore()
	if err != nil {
		return err
	}
	return login(ctx, client, serverURL, name, scope, stdout, openBrowser, store)
}

// login is Login with an explicit token store (tests inject a temp one).
func login(ctx context.Context, client *http.Client, serverURL, name, scope string, stdout io.Writer, openBrowser func(string) error, store *TokenStore) error {
	meta, err := discover(ctx, client, serverURL)
	if err != nil {
		return err
	}
	// Ephemeral loopback redirect: bind port 0 so the OS picks a free port
	// and the redirect URI always matches this run's registration.
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return fmt.Errorf("open loopback listener: %w", err)
	}
	defer ln.Close()
	redirect := fmt.Sprintf("http://127.0.0.1:%d/callback", ln.Addr().(*net.TCPAddr).Port)

	clientID, err := registerClient(ctx, client, meta, "porter", redirect)
	if err != nil {
		return err
	}
	verifier, challenge, err := pkcePair()
	if err != nil {
		return err
	}
	state, err := stateValue()
	if err != nil {
		return err
	}
	authURL, err := buildAuthorizeURL(meta.AuthorizationEndpoint, clientID, redirect, challenge, scope, state)
	if err != nil {
		return err
	}

	// Start accepting the callback before opening the browser: a browser (or
	// an automated opener) may hit the loopback redirect the moment the URL is
	// opened, and the listener must already be accepting or the request blocks
	// forever. The goroutine resolves through a channel so the flow still
	// blocks (bounded by waitForCode's timeout) on the user's consent.
	type codeResult struct {
		code string
		err  error
	}
	codeCh := make(chan codeResult, 1)
	go func() {
		code, err := waitForCode(ctx, ln, state)
		codeCh <- codeResult{code: code, err: err}
	}()

	if openBrowser == nil {
		openBrowser = openURL
	}
	fmt.Fprintf(stdout, "Open this URL to authorize porter to access %s:\n\n%s\n\n", serverURL, authURL)
	if err := openBrowser(authURL); err != nil {
		fmt.Fprintf(stdout, "(could not open a browser automatically: %v — open the URL manually)\n", err)
	}

	res := <-codeCh
	if res.err != nil {
		return res.err
	}
	entry, err := exchangeCode(ctx, client, meta.TokenEndpoint, serverURL, clientID, res.code, redirect, verifier, scope)
	if err != nil {
		return err
	}
	if err := store.Put(entry); err != nil {
		return err
	}
	fmt.Fprintf(stdout, "Logged in to %s.\n", serverURL)
	return nil
}

// loginTimeout bounds how long porter waits for the user to complete the
// browser authorization.
const loginTimeout = 5 * time.Minute

// waitForCode accepts one HTTP request on the loopback listener and returns
// the authorization code from its query string (or an error for an OAuth
// error response, a state mismatch, a malformed request, or a timeout).
// state is the CSRF value sent in the authorize URL; a callback that doesn't
// echo it back is rejected before the code is accepted.
func waitForCode(ctx context.Context, ln net.Listener, state string) (string, error) {
	ctx, cancel := context.WithTimeout(ctx, loginTimeout)
	defer cancel()
	type res struct {
		code string
		err  error
	}
	ch := make(chan res, 1)
	go func() {
		conn, err := ln.Accept()
		if err != nil {
			ch <- res{err: err}
			return
		}
		defer conn.Close()
		_ = conn.SetReadDeadline(time.Now().Add(10 * time.Second))
		var buf []byte
		tmp := make([]byte, 1024)
		for {
			n, err := conn.Read(tmp)
			if n > 0 {
				buf = append(buf, tmp[:n]...)
				if i := bytes.Index(buf, []byte("\r\n\r\n")); i >= 0 {
					buf = buf[:i]
					break
				}
			}
			if err != nil {
				ch <- res{err: err}
				return
			}
		}
		// Request line: GET /callback?code=... HTTP/1.1
		parts := strings.SplitN(string(buf), " ", 3)
		if len(parts) < 2 {
			ch <- res{err: errors.New("bad callback request")}
			return
		}
		u, err := url.Parse(parts[1])
		if err != nil {
			ch <- res{err: err}
			return
		}
		if e := u.Query().Get("error"); e != "" {
			desc := u.Query().Get("error_description")
			if desc != "" {
				ch <- res{err: fmt.Errorf("authorization failed: %s: %s", e, desc)}
			} else {
				ch <- res{err: fmt.Errorf("authorization failed: %s", e)}
			}
			return
		}
		if got := u.Query().Get("state"); got != state {
			ch <- res{err: errors.New("authorization response state mismatch (possible CSRF)")}
			return
		}
		code := u.Query().Get("code")
		if code == "" {
			ch <- res{err: errors.New("authorization response missing code")}
			return
		}
		body := "<html><body><p>Authorization complete. You can close this window and return to porter.</p></body></html>"
		fmt.Fprintf(conn, "HTTP/1.1 200 OK\r\nContent-Type: text/html\r\nContent-Length: %d\r\nConnection: close\r\n\r\n%s", len(body), body)
		ch <- res{code: code}
	}()
	select {
	case r := <-ch:
		return r.code, r.err
	case <-ctx.Done():
		return "", fmt.Errorf("timed out waiting for authorization (open the printed URL within %s)", loginTimeout)
	}
}

// openURL opens url in the user's browser, best-effort.
func openURL(u string) error {
	var cmd *exec.Cmd
	switch runtime.GOOS {
	case "darwin":
		cmd = exec.Command("open", u)
	case "linux":
		cmd = exec.Command("xdg-open", u)
	case "windows":
		cmd = exec.Command("rundll32", "url.dll,FileProtocolHandler", u)
	default:
		return fmt.Errorf("no browser opener for %s", runtime.GOOS)
	}
	return cmd.Start()
}

// Logout revokes the stored token at the server (best-effort) and removes it
// from the store. It is idempotent: a server with no stored token is a
// no-op success.
func Logout(ctx context.Context, client *http.Client, serverURL string) error {
	store, err := OpenTokenStore()
	if err != nil {
		return err
	}
	return logout(ctx, client, serverURL, store)
}

// logout is Logout with an explicit token store (tests inject a temp one).
func logout(ctx context.Context, client *http.Client, serverURL string, store *TokenStore) error {
	e := store.Get(serverURL)
	if e == nil {
		return nil
	}
	if meta, err := discover(ctx, client, serverURL); err == nil && meta.RevocationEndpoint != "" && e.AccessToken != "" {
		form := url.Values{"token": {e.AccessToken}, "client_id": {e.ClientID}}
		req, err := http.NewRequestWithContext(ctx, http.MethodPost, meta.RevocationEndpoint, strings.NewReader(form.Encode()))
		if err == nil {
			req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
			if resp, err := client.Do(req); err == nil {
				_ = resp.Body.Close()
			}
		}
	}
	return store.Delete(serverURL)
}

// ensureToken makes sure the server has a fresh access token before a
// request: for bearer servers it is a no-op; for OAuth servers it loads the
// stored token and refreshes it when expired. It runs under s.mu so
// concurrent requests refresh once (the agent runs tools serially, but a
// fetch and a call can overlap).
func (s *Server) ensureToken(ctx context.Context, client *http.Client) error {
	if s.Auth.Type != "oauth" {
		return nil
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	e := s.tokenEntry()
	if e == nil || e.AccessToken == "" {
		return fmt.Errorf("OAuth: not logged in to %s (run `porter mcp login %s`)", s.URL, s.Name)
	}
	if e.ExpiresAt > 0 && time.Now().Unix() >= e.ExpiresAt-30 {
		if err := s.refreshTokenLocked(ctx, client, e); err != nil {
			return fmt.Errorf("refresh OAuth token for %s: %w", s.Name, err)
		}
	}
	return nil
}

// forceRefresh discards the stored token and refreshes from the refresh
// token, clearing the entry when the refresh token is rejected too. It is
// the 401 path: the access token was revoked or rejected server-side.
func (s *Server) forceRefresh(ctx context.Context, client *http.Client) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	e := s.tokenEntry()
	if e == nil || e.RefreshToken == "" {
		if s.tokens != nil {
			_ = s.tokens.Delete(s.URL)
		}
		return fmt.Errorf("OAuth: token rejected for %s and no refresh token (run `porter mcp login %s`)", s.Name, s.Name)
	}
	if err := s.refreshTokenLocked(ctx, client, e); err != nil {
		if s.tokens != nil {
			_ = s.tokens.Delete(s.URL)
		}
		return fmt.Errorf("OAuth: token rejected for %s and refresh failed: %w", s.Name, err)
	}
	return nil
}

// refreshTokenLocked refreshes an entry and persists it. Callers hold s.mu.
func (s *Server) refreshTokenLocked(ctx context.Context, client *http.Client, e *TokenEntry) error {
	if err := refreshToken(ctx, client, e); err != nil {
		return err
	}
	if s.tokens != nil {
		return s.tokens.Put(e)
	}
	return nil
}

// tokenEntry returns the server's stored token entry, or nil.
func (s *Server) tokenEntry() *TokenEntry {
	if s.tokens == nil {
		return nil
	}
	return s.tokens.Get(s.URL)
}
