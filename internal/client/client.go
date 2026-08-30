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

// Create makes a new session and returns its id, initial (empty) history, and
// the seq to subscribe from.
func (c *Client) Create(ctx context.Context) (api.SessionInfo, error) {
	var info api.SessionInfo
	if err := c.doJSON(ctx, http.MethodPost, c.base+api.SessionsPath, nil, &info); err != nil {
		return api.SessionInfo{}, err
	}
	return info, nil
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

// ServeExec registers this client as the session's execution provider and
// blocks, reading exec requests from the server and running each via dispatch.
// It streams the output back to the server. Each request runs in its own
// goroutine so the read loop stays free to receive a Cancel=true request while
// a command is running (the agent runs tools sequentially, so at most one
// command is in flight); cancelling aborts the running command's context,
// which for the local dispatcher kills its process group. It returns when the
// connection ends (or ctx is cancelled); callers should retry to re-register.
func (c *Client) ServeExec(ctx context.Context, id string, dispatch func(ctx context.Context, name string, args []byte) (io.ReadCloser, error)) error {
	u := c.path(api.SessionExecPath, id)
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
	return fmt.Errorf("server %s: %s", resp.Status, strings.TrimSpace(buf.String()))
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
