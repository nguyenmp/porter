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

// applyAuth sets the Authorization header. Bearer servers use the static
// token from config; OAuth servers use the stored access token (callers must
// ensureToken first so it is fresh).
func (s *Server) applyAuth(req *http.Request) {
	switch s.Auth.Type {
	case "bearer":
		if s.Auth.Token != "" {
			req.Header.Set("Authorization", "Bearer "+s.Auth.Token)
		}
	case "oauth":
		if e := s.tokenEntry(); e != nil && e.AccessToken != "" {
			req.Header.Set("Authorization", "Bearer "+e.AccessToken)
		}
	}
}

// callError distinguishes transport-level failures (an HTTP status) from
// JSON-RPC errors (an error code), so call can apply recovery — a rejected
// token or an expired session — before reporting the failure.
type callError struct {
	httpStatus int    // HTTP status for non-2xx responses; 0 for JSON-RPC errors
	rpcCode    int    // JSON-RPC error code; 0 for HTTP-level failures
	message    string // full error text
}

func (e *callError) Error() string { return e.message }

// sessionNotFound reports whether the failure is a streamable-HTTP "session
// not found" rejection: the server could not match the Mcp-Session-Id we sent
// because the session lapsed server-side.
func (e *callError) sessionNotFound() bool {
	return e.rpcCode == -32001 && strings.Contains(strings.ToLower(e.message), "session")
}

// call sends one id-bearing JSON-RPC request and returns the response's
// result plus the response headers (a server may return the session id in an
// Mcp-Session-Id header on initialize). The request runs under ctx, so
// cancelling ctx (a user clicking Cancel) aborts the HTTP call. An HTTP
// error, a JSON-RPC error, or a timeout are returned as errors.
//
// Two failures are recovered once before being reported:
//
//   - HTTP 401 on an OAuth server: the stored token was rejected (revoked or
//     wrong scope), so it is force-refreshed and the request retried.
//   - JSON-RPC -32001 "Session not found": the streamable-HTTP session the
//     server assigned on initialize lapsed server-side (server-defined, and
//     sometimes minutes), so the initialize handshake re-runs to mint a fresh
//     session and the request is retried.
//
// initialize itself is exempt from both recoveries: it is the recovery, and
// must not recurse.
func (s *Server) call(ctx context.Context, client *http.Client, id any, method string, params any) (json.RawMessage, http.Header, error) {
	timeout := s.Timeout
	if timeout <= 0 {
		timeout = DefaultTimeout
	}
	ctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	// OAuth servers: make sure the stored token is fresh (refreshing when
	// expired) before the request is built.
	if err := s.ensureToken(ctx, client); err != nil {
		return nil, nil, err
	}

	// send performs one round trip and parses the JSON-RPC response, so the
	// recovery paths below can re-run the request after refreshing a token or
	// re-initializing a session.
	send := func() (json.RawMessage, http.Header, *callError) {
		req, err := s.newRequest(ctx, id, method, params)
		if err != nil {
			return nil, nil, &callError{message: fmt.Sprintf("%s: %v", method, err)}
		}
		resp, err := client.Do(req)
		if err != nil {
			return nil, nil, &callError{message: fmt.Sprintf("%s: %v", method, err)}
		}
		defer resp.Body.Close()
		if resp.StatusCode != http.StatusOK && resp.StatusCode != http.StatusAccepted {
			msg, _ := io.ReadAll(io.LimitReader(resp.Body, 2048))
			cerr := &callError{
				httpStatus: resp.StatusCode,
				message:    fmt.Sprintf("%s: unexpected status %s: %s", method, resp.Status, strings.TrimSpace(string(msg))),
			}
			// Some servers report JSON-RPC errors with a non-2xx status (e.g.
			// Retool answers a stale session with 404 + -32001 "Session not
			// found"): parse the body so recovery can recognize them.
			var rpc rpcResponse
			if err := json.Unmarshal(msg, &rpc); err == nil && rpc.Error != nil {
				cerr.rpcCode = rpc.Error.Code
				cerr.message = fmt.Sprintf("%s: %v", method, rpc.Error)
			}
			return nil, nil, cerr
		}
		data, err := io.ReadAll(resp.Body)
		if err != nil {
			return nil, nil, &callError{message: fmt.Sprintf("read %s response: %v", method, err)}
		}
		var rpc rpcResponse
		if ct := resp.Header.Get("Content-Type"); strings.Contains(ct, "text/event-stream") {
			// Streamable HTTP servers may respond to a request with an SSE
			// stream instead of a plain JSON body; the JSON-RPC response rides
			// in the first event's data field.
			ev, err := parseSSEEvent(data)
			if err != nil {
				return nil, nil, &callError{message: fmt.Sprintf("parse %s SSE response: %v", method, err)}
			}
			if err := json.Unmarshal(ev, &rpc); err != nil {
				return nil, nil, &callError{message: fmt.Sprintf("decode %s SSE response: %v", method, err)}
			}
		} else {
			if err := json.Unmarshal(data, &rpc); err != nil {
				return nil, nil, &callError{message: fmt.Sprintf("decode %s response: %v", method, err)}
			}
		}
		if rpc.Error != nil {
			return nil, nil, &callError{
				rpcCode: rpc.Error.Code,
				message: fmt.Sprintf("%s: %v", method, rpc.Error),
			}
		}
		return rpc.Result, resp.Header, nil
	}

	result, header, cerr := send()
	if cerr != nil && method != "initialize" {
		switch {
		case cerr.httpStatus == http.StatusUnauthorized && s.Auth.Type == "oauth":
			// The stored token was rejected (revoked or wrong scope): force a
			// refresh and retry once before reporting the failure.
			if err := s.forceRefresh(ctx, client); err != nil {
				return nil, nil, fmt.Errorf("%s: %w", method, err)
			}
			result, header, cerr = send()
		case cerr.sessionNotFound():
			// The streamable-HTTP session lapsed server-side: re-run the
			// initialize handshake to mint a fresh session and retry once.
			if err := s.handshake(ctx, client); err != nil {
				return nil, nil, fmt.Errorf("%s: session not found, re-initialize failed: %w", method, err)
			}
			result, header, cerr = send()
		}
	}
	if cerr != nil {
		return nil, nil, cerr
	}
	return result, header, nil
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
