// Package client is the stateless HTTP client a porter front-end (REPL or
// one-shot) uses to reach the server. It owns no conversation state: it creates
// a session, appends user messages, polls history, and subscribes to the event
// bus to render live. The server decides when turns run.
package client

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/url"
	"strconv"
	"strings"

	"porter/internal/api"
)

// ErrResync indicates a subscription fell too far behind the server's ring
// buffer. The caller must refetch history and resubscribe with the new seq.
var ErrResync = errors.New("resync required: refetch history and resubscribe")

// Client talks to a porter server.
type Client struct {
	base string
	http *http.Client
}

// New returns a Client for the given server base URL.
func New(base string) *Client {
	return &Client{base: base, http: http.DefaultClient}
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

// path substitutes the session id into an api route spec.
func (c *Client) path(spec, id string) string {
	if id != "" {
		spec = strings.Replace(spec, "{id}", url.PathEscape(id), 1)
	}
	return c.base + spec
}

// statusError reads a short error body so we can include it in the error.
func (c *Client) statusError(resp *http.Response) error {
	buf := new(bytes.Buffer)
	_, _ = buf.ReadFrom(http.MaxBytesReader(nil, resp.Body, 4096))
	return fmt.Errorf("server %s: %s", resp.Status, strings.TrimSpace(buf.String()))
}