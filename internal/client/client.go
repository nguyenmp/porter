// Package client is the thin HTTP client a porter front-end (REPL or one-shot)
// uses to reach the stateless server. It holds no conversation state: it sends
// the full history each turn, relays the server's event stream to a sink, and
// returns the completion so the caller can keep it as its history.
package client

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"

	"porter/internal/api"
	"porter/internal/codec"
	"porter/internal/llm"
)

// Client talks to a porter server.
type Client struct {
	base string
	http *http.Client
}

// New returns a Client for the given server base URL.
func New(base string) *Client {
	return &Client{base: base, http: http.DefaultClient}
}

// Stream sends history to the server, calls onEvent for every event it streams
// back, and returns the api.Completion trailer (history, final text, usage).
// The completion is not passed to onEvent.
func (c *Client) Stream(ctx context.Context, history []llm.ChatMessage, onEvent func(codec.Event)) (api.Completion, error) {
	body, err := json.Marshal(api.StreamRequest{History: history})
	if err != nil {
		return api.Completion{}, fmt.Errorf("marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.base+api.StreamPath, bytes.NewReader(body))
	if err != nil {
		return api.Completion{}, fmt.Errorf("build request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.http.Do(req)
	if err != nil {
		return api.Completion{}, fmt.Errorf("stream: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		msg, _ := readBody(resp)
		return api.Completion{}, fmt.Errorf("server %s: %s", resp.Status, msg)
	}

	var comp api.Completion
	sc := bufio.NewScanner(resp.Body)
	for sc.Scan() {
		line := sc.Bytes()
		var probe struct {
			Completed bool `json:"completed"`
		}
		if err := json.Unmarshal(line, &probe); err != nil {
			return api.Completion{}, fmt.Errorf("decode stream line: %w", err)
		}
		if probe.Completed {
			if err := json.Unmarshal(line, &comp); err != nil {
				return api.Completion{}, fmt.Errorf("decode completion: %w", err)
			}
			break
		}
		var ev codec.Event
		if err := json.Unmarshal(line, &ev); err != nil {
			return api.Completion{}, fmt.Errorf("decode event: %w", err)
		}
		if onEvent != nil {
			onEvent(ev)
		}
	}
	if err := sc.Err(); err != nil {
		return api.Completion{}, err
	}
	return comp, nil
}

// readBody reads a short error body so we can include it in a status error.
func readBody(resp *http.Response) (string, error) {
	buf := new(bytes.Buffer)
	_, err := buf.ReadFrom(http.MaxBytesReader(nil, resp.Body, 4096))
	return buf.String(), err
}