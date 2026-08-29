package mcp

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
)

// rpcRequest is a JSON-RPC 2.0 request. ID is nil for notifications (the id
// field is then omitted).
type rpcRequest struct {
	JSONRPC string `json:"jsonrpc"`
	ID      any    `json:"id,omitempty"`
	Method  string `json:"method"`
	Params  any    `json:"params,omitempty"`
}

// rpcResponse is a JSON-RPC 2.0 response; exactly one of Result and Error is
// set.
type rpcResponse struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      any             `json:"id"`
	Result  json.RawMessage `json:"result,omitempty"`
	Error   *rpcError       `json:"error,omitempty"`
}

// rpcError is a JSON-RPC 2.0 error object.
type rpcError struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
}

func (e *rpcError) Error() string {
	return fmt.Sprintf("MCP error %d: %s", e.Code, e.Message)
}

// newRequest builds the HTTP POST carrying one JSON-RPC request, applying the
// server's auth, session id, and negotiated protocol version headers.
func (s *Server) newRequest(ctx context.Context, id any, method string, params any) (*http.Request, error) {
	body, err := json.Marshal(rpcRequest{JSONRPC: "2.0", ID: id, Method: method, Params: params})
	if err != nil {
		return nil, fmt.Errorf("marshal %s request: %w", method, err)
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, s.URL, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("build %s request: %w", method, err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json, text/event-stream")
	s.applyAuth(req)
	s.mu.RLock()
	if s.sessionID != "" {
		req.Header.Set("Mcp-Session-Id", s.sessionID)
	}
	if s.protocol != "" {
		req.Header.Set("MCP-Protocol-Version", s.protocol)
	}
	s.mu.RUnlock()
	return req, nil
}

// applyAuth sets the Authorization header for a bearer-token server.
func (s *Server) applyAuth(req *http.Request) {
	if s.Auth.Type == "bearer" && s.Auth.Token != "" {
		req.Header.Set("Authorization", "Bearer "+s.Auth.Token)
	}
}

// call sends one id-bearing JSON-RPC request and returns the response's
// result plus the response headers (a server may return the session id in an
// Mcp-Session-Id header on initialize). The request runs under ctx, so
// cancelling ctx (a user clicking Cancel) aborts the HTTP call. An HTTP
// error, a JSON-RPC error, or a timeout are returned as errors.
func (s *Server) call(ctx context.Context, client *http.Client, id any, method string, params any) (json.RawMessage, http.Header, error) {
	timeout := s.Timeout
	if timeout <= 0 {
		timeout = DefaultTimeout
	}
	ctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	req, err := s.newRequest(ctx, id, method, params)
	if err != nil {
		return nil, nil, err
	}
	resp, err := client.Do(req)
	if err != nil {
		return nil, nil, fmt.Errorf("%s: %w", method, err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK && resp.StatusCode != http.StatusAccepted {
		msg, _ := io.ReadAll(io.LimitReader(resp.Body, 2048))
		return nil, nil, fmt.Errorf("%s: unexpected status %s: %s", method, resp.Status, strings.TrimSpace(string(msg)))
	}
	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, nil, fmt.Errorf("read %s response: %w", method, err)
	}
	var rpc rpcResponse
	if ct := resp.Header.Get("Content-Type"); strings.Contains(ct, "text/event-stream") {
		// Streamable HTTP servers may respond to a request with an SSE stream
		// instead of a plain JSON body; the JSON-RPC response rides in the
		// first event's data field.
		ev, err := parseSSEEvent(data)
		if err != nil {
			return nil, nil, fmt.Errorf("parse %s SSE response: %w", method, err)
		}
		if err := json.Unmarshal(ev, &rpc); err != nil {
			return nil, nil, fmt.Errorf("decode %s SSE response: %w", method, err)
		}
	} else {
		if err := json.Unmarshal(data, &rpc); err != nil {
			return nil, nil, fmt.Errorf("decode %s response: %w", method, err)
		}
	}
	if rpc.Error != nil {
		return nil, nil, fmt.Errorf("%s: %w", method, rpc.Error)
	}
	return rpc.Result, resp.Header, nil
}

// notify sends a JSON-RPC notification (no id). The response is drained and
// ignored beyond its status.
func (s *Server) notify(ctx context.Context, client *http.Client, method string, params any) error {
	timeout := s.Timeout
	if timeout <= 0 {
		timeout = DefaultTimeout
	}
	ctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	req, err := s.newRequest(ctx, nil, method, params)
	if err != nil {
		return err
	}
	resp, err := client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	_, _ = io.Copy(io.Discard, resp.Body)
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return fmt.Errorf("%s: unexpected status %s", method, resp.Status)
	}
	return nil
}

// parseSSEEvent extracts the data payload of the first event in an SSE body.
// A JSON-RPC response rides in the data field of a single event; multi-line
// data payloads are joined with newlines per the SSE spec.
func parseSSEEvent(data []byte) ([]byte, error) {
	var payload []byte
	for _, line := range strings.Split(string(data), "\n") {
		line = strings.TrimRight(line, "\r")
		if rest, ok := strings.CutPrefix(line, "data:"); ok {
			if len(payload) > 0 {
				payload = append(payload, '\n')
			}
			payload = append(payload, []byte(strings.TrimPrefix(rest, " "))...)
			continue
		}
		if line == "" && len(payload) > 0 {
			break // end of the first event
		}
	}
	if len(payload) == 0 {
		return nil, errors.New("no data payload in SSE response")
	}
	return payload, nil
}
