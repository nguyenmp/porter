package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"porter/internal/config"
)

// chatRequest is the subset of the Chat Completions request body we send.
type chatRequest struct {
	Model    string        `json:"model"`
	Messages []chatMessage `json:"messages"`
	Stream   bool          `json:"stream"`
}

type chatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// Client makes streaming Chat Completions requests.
type Client struct {
	cfg    config.Config
	http   *http.Client
	closer io.Closer

	// Debug, when non-nil, receives progress lines (connect, upload, first
	// byte) on a writer such as stderr. Leave nil to keep output quiet.
	Debug io.Writer
}

// NewClient returns a Client for the given configuration.
func NewClient(cfg config.Config, hc *http.Client) *Client {
	if hc == nil {
		hc = http.DefaultClient
	}
	return &Client{cfg: cfg, http: hc}
}

// debugf writes a formatted progress line to Debug, if set.
func (c *Client) debugf(format string, args ...any) {
	if c.Debug == nil {
		return
	}
	fmt.Fprintf(c.Debug, "porter: "+format+"\n", args...)
}

// After returns the underlying HTTP connection for closing after streaming.
func (c *Client) After() io.Closer {
	return c.closer
}

// Stream starts a streaming Chat Completions request for the given user
// prompt and returns the raw SSE response body. Strip the base URL's trailing
// slash so the endpoint join is predictable.
func (c *Client) Stream(ctx context.Context, prompt string) (io.ReadCloser, error) {
	body, err := json.Marshal(chatRequest{
		Model:    c.cfg.Model,
		Messages: []chatMessage{{Role: "user", Content: prompt}},
		Stream:   true,
	})
	if err != nil {
		return nil, fmt.Errorf("marshal request: %w", err)
	}

	endpoint := strings.TrimSuffix(c.cfg.BaseURL, "/") + "/chat/completions"
	c.debugf("uploading %d byte prompt to %s (model=%s)", len(body), endpoint, c.cfg.Model)

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("build request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	if c.cfg.APIKey != "" {
		req.Header.Set("Authorization", "Bearer "+c.cfg.APIKey)
	}

	c.debugf("connecting to %s...", endpoint)
	start := time.Now()
	resp, err := c.http.Do(req)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", endpoint, err)
	}
	c.debugf("connected in %s: status=%s content-type=%s", time.Since(start).Round(time.Millisecond), resp.Status, resp.Header.Get("Content-Type"))

	if resp.StatusCode != http.StatusOK {
		msg, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		resp.Body.Close()
		return nil, fmt.Errorf("%s: unexpected status %s: %s", endpoint, resp.Status, strings.TrimSpace(string(msg)))
	}
	c.closer = resp.Body
	return resp.Body, nil
}
