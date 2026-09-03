// Package client is the stateless HTTP client a porter front-end (REPL or
// one-shot) uses to reach the server. It owns no conversation state: it creates
// a session, appends user messages, polls history, and subscribes to the event
// bus to render live. The server decides when turns run.
package client

import (
	"bufio"
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"sync"

	"porter/internal/api"
)

// ErrResync indicates a subscription fell too far behind the server's ring
// buffer. The caller must refetch history and resubscribe with the new seq.
var ErrResync = errors.New("resync required: refetch history and resubscribe")

// DuplicateHostError reports that the server rejected this host's
// registration because another host agent is already connected with the same
// host id (the server answers 409 Conflict with a message naming the other
// process). The host agent surfaces the server's message and exits instead
// of retrying forever, so a second `make host` fails loudly.
type DuplicateHostError struct {
	Message string
}

func (e *DuplicateHostError) Error() string { return e.Message }

// BasicAuth carries the credentials a client sends to a password-protected
// porter server (see the server's PORTER_AUTH_USERNAME/PORTER_AUTH_PASSWORD).
// When Username is empty the client sends no Authorization header.
type BasicAuth struct {
	Username string
	Password string
}

// Client talks to a porter server.
type Client struct {
	base string
	http *http.Client
}

// New returns a Client for the given server base URL. Pass BasicAuth to have
// every request (including long-lived exec and SSE streams) carry the header.
func New(base string, auth ...BasicAuth) *Client {
	c := &Client{base: base, http: http.DefaultClient}
	if len(auth) > 0 && auth[0].Username != "" {
		token := "Basic " + base64.StdEncoding.EncodeToString([]byte(auth[0].Username+":"+auth[0].Password))
		c.http = &http.Client{Transport: authTransport{token: token}}
	}
	return c
}

// authTransport injects the Authorization header into every request, so the
// exec provider connection and SSE streams authenticate just like the REST
// calls do.
type authTransport struct {
	token string
}

func (t authTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	req = req.Clone(req.Context())
	req.Header.Set("Authorization", t.token)
	return http.DefaultTransport.RoundTrip(req)
}

// Create makes a new session on the server process ("local" execution) and
// returns its id, initial (empty) history, and the seq to subscribe from.
func (c *Client) Create(ctx context.Context) (api.SessionInfo, error) {
	return c.CreateHost(ctx, api.CreateRequest{})
}

// CreateHost makes a new session whose execution provider is provisioned on
// the named execution host (persistent agent, e.g. a laptop) before the
// server returns — so the first message runs there. req.Host "" or "local"
// uses the server process (Create). A provisioning failure is not fatal: the
// returned SessionInfo carries a Warning and the session falls back to local
// execution until the host's provider connects.
func (c *Client) CreateHost(ctx context.Context, req api.CreateRequest) (api.SessionInfo, error) {
	var info api.SessionInfo
	if err := c.doJSON(ctx, http.MethodPost, c.base+api.SessionsPath, jsonMarshal(req), &info); err != nil {
		return api.SessionInfo{}, err
	}
	return info, nil
}

// jsonMarshal encodes v, returning a nil slice (instead of "null") when v is
// the zero value, so a body-less POST stays body-less.
func jsonMarshal(v any) []byte {
	b, err := json.Marshal(v)
	if err != nil {
		return nil
	}
	if string(b) == "{}" || string(b) == "null" {
		return nil
	}
	return b
}

// History returns the session's authoritative committed history and its seq.
func (c *Client) History(ctx context.Context, id string) (api.SessionHistory, error) {
	var h api.SessionHistory
	if err := c.doJSON(ctx, http.MethodGet, c.path(api.SessionHistoryPath, id), nil, &h); err != nil {
		return api.SessionHistory{}, err
	}
	return h, nil
}

// Runs returns the session's in-flight tool runs and the server's clock.
func (c *Client) Runs(ctx context.Context, id string) (api.RunsResponse, error) {
	var out api.RunsResponse
	err := c.doJSON(ctx, http.MethodGet, c.path(api.SessionRunsPath, id), nil, &out)
	return out, err
}

// Append queues a user message for the session's turn scheduler. It is
// non-blocking with respect to the running turn; the server decides when it
// runs.
func (c *Client) Append(ctx context.Context, id, content string) error {
	body, err := json.Marshal(api.AppendRequest{Content: content})
	if err != nil {
		return fmt.Errorf("marshal append: %w", err)
	}
	return c.doJSON(ctx, http.MethodPost, c.path(api.SessionMessagesPath, id), body, nil)
}

// Cancel stops an in-flight tool run: the server kills the running command
// (locally via its process group, remotely by signalling the execution client)
// and ends the turn. It returns an error when the run is unknown, e.g. it
// already finished.
func (c *Client) Cancel(ctx context.Context, id, callID string) error {
	return c.doJSON(ctx, http.MethodPost, c.path(api.SessionCancelPath, id, callID), nil, nil)
}

// Stop aborts the session's currently running turn: the server cancels the
// model stream (committing any partial reply, marked interrupted), stops any
// running tool, and ends the turn with a stopped marker. It returns an error
// when no turn is running (idle, or already finished).
func (c *Client) Stop(ctx context.Context, id string) error {
	return c.doJSON(ctx, http.MethodPost, c.path(api.SessionStopPath, id), nil, nil)
}

// Archive marks the session archived, folding it out of the active sidebar
// list into the Archived folder. The chat itself is unaffected.
func (c *Client) Archive(ctx context.Context, id string) error {
	return c.doJSON(ctx, http.MethodPost, c.path(api.SessionArchivePath, id), nil, nil)
}

// Unarchive restores an archived session to the active list.
func (c *Client) Unarchive(ctx context.Context, id string) error {
	return c.doJSON(ctx, http.MethodPost, c.path(api.SessionUnarchivePath, id), nil, nil)
}

// Rename sets (or clears) a session's custom display name. An empty name
// clears it back to the first-message preview.
func (c *Client) Rename(ctx context.Context, id, name string) error {
	body, err := json.Marshal(api.RenameRequest{Name: name})
	if err != nil {
		return fmt.Errorf("marshal rename request: %w", err)
	}
	return c.doJSON(ctx, http.MethodPost, c.path(api.SessionRenamePath, id), body, nil)
}

// Subscribe streams the session's event bus as NDJSON, calling onEvent for
// every envelope until the connection ends, until until(env) reports true, or
// until a resync is required (returned as ErrResync). onEvent may be nil.
func (c *Client) Subscribe(ctx context.Context, id string, since uint64, onEvent func(api.Envelope), until func(api.Envelope) bool) error {
	u := c.path(api.SessionEventsPath, id) + "?since=" + strconv.FormatUint(since, 10)
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, u, nil)
	if err != nil {
		return fmt.Errorf("build subscribe: %w", err)
	}
	resp, err := c.http.Do(req)
	if err != nil {
		return fmt.Errorf("subscribe: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return c.statusError(resp)
	}

	sc := bufio.NewScanner(resp.Body)
	for sc.Scan() {
		var env api.Envelope
		if err := json.Unmarshal(sc.Bytes(), &env); err != nil {
			return fmt.Errorf("decode envelope: %w", err)
		}
		if env.Kind == api.KindResync {
			return ErrResync
		}
		if onEvent != nil {
			onEvent(env)
		}
		if until != nil && until(env) {
			return nil
		}
	}
	if err := sc.Err(); err != nil {
		return err
	}
	return nil
}

// ExecConn is the identity a client registers under when it connects as a
// session's execution provider: ID is the stable id the selector addresses
// (e.g. the hostname), Name is the human-readable label shown in the picker,
// and Kind is "remote" (cloud providers later). A zero ExecConn registers
// under a server-generated id, for legacy clients.
type ExecConn struct {
	ID   string
	Name string
	Kind string
}

// ServeExec registers this client as the session's execution provider and
// blocks, reading exec requests from the server and running each via dispatch.
// It streams the output back to the server. Each request runs in its own
// goroutine so the read loop stays free to receive a Cancel=true request while
// a command is running (the agent runs tools sequentially, so at most one
// command is in flight); cancelling aborts the running command's context,
// which for the local dispatcher kills its process group. It returns when the
// connection ends (or ctx is cancelled); callers should retry to re-register.
// An optional ExecConn names the provider in the session's registry (the web
// picker); callers that pass none are registered under a server-generated id.
func (c *Client) ServeExec(ctx context.Context, id string, dispatch func(ctx context.Context, name string, args []byte) (io.ReadCloser, error), conn ...ExecConn) error {
	u := c.path(api.SessionExecPath, id)
	if len(conn) > 0 {
		q := url.Values{}
		if conn[0].ID != "" {
			q.Set("id", conn[0].ID)
		}
		if conn[0].Name != "" {
			q.Set("name", conn[0].Name)
		}
		if conn[0].Kind != "" {
			q.Set("kind", conn[0].Kind)
		}
		if len(q) > 0 {
			u += "?" + q.Encode()
		}
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, u, nil)
	if err != nil {
		return fmt.Errorf("build exec: %w", err)
	}
	resp, err := c.http.Do(req)
	if err != nil {
		return fmt.Errorf("exec: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return c.statusError(resp)
	}

	// running tracks the cancel func of each command started on this
	// connection, keyed by the server's call id.
	var mu sync.Mutex
	running := make(map[string]context.CancelFunc)

	sc := bufio.NewScanner(resp.Body)
	for sc.Scan() {
		var er api.ExecRequest
		if err := json.Unmarshal(sc.Bytes(), &er); err != nil {
			return fmt.Errorf("decode exec request: %w", err)
		}
		if er.Cancel {
			// The user cancelled a run on the server. Stop every command we're
			// running — the agent runs tools one at a time, so this is the one
			// the cancel targets.
			mu.Lock()
			for _, cancel := range running {
				cancel()
			}
			mu.Unlock()
			continue
		}
		callCtx, cancel := context.WithCancel(ctx)
		mu.Lock()
		running[er.CallID] = cancel
		mu.Unlock()
		go func(callID string, name string, args string) {
			defer func() {
				mu.Lock()
				delete(running, callID)
				mu.Unlock()
				cancel()
			}()
			stream, err := dispatch(callCtx, name, []byte(args))
			if err != nil {
				// Can't start the tool; report it as the result so the agent
				// sees an error rather than hanging.
				_ = c.postExecResult(ctx, id, callID, strings.NewReader("error: "+err.Error()))
				return
			}
			_ = c.postExecResult(ctx, id, callID, stream)
			_ = stream.Close()
		}(er.CallID, er.Name, er.Arguments)
	}
	if err := sc.Err(); err != nil {
		return err
	}
	return nil
}

// postExecResult streams a tool's output back to the server for a call id.
func (c *Client) postExecResult(ctx context.Context, id, callID string, body io.Reader) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.path(api.SessionExecResultPath, id, callID), body)
	if err != nil {
		return err
	}
	resp, err := c.http.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return c.statusError(resp)
	}
	return nil
}

// doJSON performs a request, decoding a JSON response body into out when out is
// non-nil.
func (c *Client) doJSON(ctx context.Context, method, url string, body []byte, out any) error {
	req, err := http.NewRequestWithContext(ctx, method, url, bytes.NewReader(body))
	if err != nil {
		return fmt.Errorf("build request: %w", err)
	}
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	resp, err := c.http.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return c.statusError(resp)
	}
	if out != nil {
		if err := json.NewDecoder(resp.Body).Decode(out); err != nil {
			return fmt.Errorf("decode response: %w", err)
		}
	}
	return nil
}

// path substitutes the {id} and {call_id} params into an api route spec.
func (c *Client) path(spec, id string, callID ...string) string {
	if id != "" {
		spec = strings.Replace(spec, "{id}", url.PathEscape(id), 1)
	}
	if len(callID) > 0 && callID[0] != "" {
		spec = strings.Replace(spec, "{call_id}", url.PathEscape(callID[0]), 1)
	}
	return c.base + spec
}

// statusError reads a short error body so we can include it in the error.
func (c *Client) statusError(resp *http.Response) error {
	buf := new(bytes.Buffer)
	_, _ = buf.ReadFrom(http.MaxBytesReader(nil, resp.Body, 4096))
	msg := fmt.Sprintf("server %s: %s", resp.Status, strings.TrimSpace(buf.String()))
	if resp.StatusCode == http.StatusConflict {
		return &DuplicateHostError{Message: msg}
	}
	return fmt.Errorf("%s", msg)
}

// PostExecContext registers the caller's environment context with the session,
// so the server can inject it into the model and expose load_skill for the
// reported skills. It is called when the caller connects as the session's
// execution provider.
func (c *Client) PostExecContext(ctx context.Context, id string, execCtx api.ExecContext) error {
	body, err := json.Marshal(execCtx)
	if err != nil {
		return fmt.Errorf("marshal exec context: %w", err)
	}
	return c.doJSON(ctx, http.MethodPost, c.path(api.SessionExecContextPath, id), body, nil)
}

// ExecStatus returns the session's current execution provider status.
func (c *Client) ExecStatus(ctx context.Context, id string) (api.ExecStatus, error) {
	var out api.ExecStatus
	err := c.doJSON(ctx, http.MethodGet, c.path(api.SessionExecStatusPath, id), nil, &out)
	return out, err
}

// Hosts returns every registered execution host (persistent agents that can
// provision execution contexts), for the web UI's "new chat on" picker.
func (c *Client) Hosts(ctx context.Context) (api.HostsResponse, error) {
	var out api.HostsResponse
	err := c.doJSON(ctx, http.MethodGet, c.base+api.HostsPath, nil, &out)
	return out, err
}

// PostHostContext registers the host's base environment context (system,
// default working directory, files, skills) with the server, so the "new
// chat on" picker can show where the host runs. It is called when the host
// connects.
func (c *Client) PostHostContext(ctx context.Context, hostID string, execCtx api.ExecContext) error {
	body, err := json.Marshal(execCtx)
	if err != nil {
		return fmt.Errorf("marshal host context: %w", err)
	}
	u := strings.Replace(api.HostContextPath, "{host_id}", url.PathEscape(hostID), 1)
	return c.doJSON(ctx, http.MethodPost, c.base+u, body, nil)
}

// PostHostProviderError reports that provisioning a provider failed (e.g. the
// requested working directory does not exist), so the waiting session-create
// request gets the error instead of timing out.
func (c *Client) PostHostProviderError(ctx context.Context, hostID, providerID, msg string) error {
	body, err := json.Marshal(api.ProviderErrorRequest{Error: msg})
	if err != nil {
		return fmt.Errorf("marshal provider error: %w", err)
	}
	u := strings.Replace(api.HostProviderErrorPath, "{host_id}", url.PathEscape(hostID), 1)
	u = strings.Replace(u, "{provider_id}", url.PathEscape(providerID), 1)
	return c.doJSON(ctx, http.MethodPost, c.base+u, body, nil)
}

// ServeHost registers this client as an execution host and blocks, reading
// provision requests from the server. Each request is passed to provision;
// the host is expected to create the requested execution environment and
// register itself as that session's execution provider (see ServeExec). It
// returns when the connection ends (or ctx is cancelled); callers should
// retry to re-register. onConnect, when non-nil, is called exactly once once
// the exec connection is established (the server accepted the registration),
// before any provision request is read, so callers can log readiness.
func (c *Client) ServeHost(ctx context.Context, hostID, instance string, provision func(api.HostRequest) error, onConnect func()) error {
	u := strings.Replace(api.HostExecPath, "{host_id}", url.PathEscape(hostID), 1)
	if instance != "" {
		u += "?instance=" + url.QueryEscape(instance)
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.base+u, nil)
	if err != nil {
		return fmt.Errorf("build host exec: %w", err)
	}
	resp, err := c.http.Do(req)
	if err != nil {
		return fmt.Errorf("host exec: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return c.statusError(resp)
	}
	if onConnect != nil {
		onConnect()
	}

	sc := bufio.NewScanner(resp.Body)
	for sc.Scan() {
		var hr api.HostRequest
		if err := json.Unmarshal(sc.Bytes(), &hr); err != nil {
			return fmt.Errorf("decode host request: %w", err)
		}
		if err := provision(hr); err != nil {
			return fmt.Errorf("provision: %w", err)
		}
	}
	if err := sc.Err(); err != nil {
		return err
	}
	return nil
}
