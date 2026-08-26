package session

import (
	"context"
	"errors"
	"fmt"
	"io"

	"porter/internal/api"
	"porter/internal/llm"
	"porter/internal/tools"
)

// remoteProvider routes a session's tool calls to a connected execution client
// (e.g. the REPL on the host) instead of running them locally. It implements
// tools.Provider so agent.RunTurn is unchanged: Run blocks until the client
// streams the command's output back, returning that output as a stream the
// agent reads to completion.
type remoteProvider struct {
	sess *Session
}

func (r *remoteProvider) Defs() []llm.Tool {
	return tools.Defs()
}

// Run registers a tool call with the connected execution client and returns a
// stream of the client's output. The client picks the call up on its exec
// subscription and streams the result back via POST /exec/{call_id}.
func (r *remoteProvider) Run(ctx context.Context, name string, args []byte) (io.ReadCloser, error) {
	s := r.sess
	callID := s.newCallID()
	pr, pw := io.Pipe()

	call := &execCall{pw: pw}
	s.mu.Lock()
	if s.execCalls == nil {
		s.execCalls = map[string]*execCall{}
	}
	s.execCalls[callID] = call
	ch := s.execCh
	s.mu.Unlock()

	if ch == nil {
		s.dropCall(callID, pw)
		return nil, errors.New("no execution client connected")
	}
	select {
	case ch <- api.ExecRequest{CallID: callID, Name: name, Arguments: string(args)}:
	case <-ctx.Done():
		s.dropCall(callID, pw)
		return nil, ctx.Err()
	}
	// If the run's context is cancelled (a user clicking Cancel in the UI), the
	// agent's Read on pr must unblock even if the client never sends a result:
	// close the write end so pr reads EOF. The session also pushes a
	// Cancel=true request down the exec channel (see CancelRun) so the client
	// actually stops its command; this close is what lets the agent stop
	// waiting promptly regardless of the client's behavior.
	go func() {
		<-ctx.Done()
		_ = pw.Close()
	}()
	return pr, nil
}

// execCall tracks one in-flight tool call awaiting the client's streamed result.
type execCall struct {
	pw *io.PipeWriter
}

// RegisterExec sets the session's execution client and points the session's
// provider at it. Called when a client connects to GET /api/sessions/{id}/exec.
func (s *Session) RegisterExec(ch chan api.ExecRequest) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.execCh = ch
	s.js = &remoteProvider{sess: s}
}

// UnregisterExec clears the execution client, closes any in-flight calls so the
// agent does not hang, and reverts to local execution.
func (s *Session) UnregisterExec() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.execCh = nil
	for id, c := range s.execCalls {
		_ = c.pw.CloseWithError(errors.New("execution client disconnected"))
		delete(s.execCalls, id)
	}
	s.js = tools.NewDispatcher()
}

// ExecResult pipes a client's streamed output into the in-flight call's writer,
// completing the agent's ReadAll.
func (s *Session) ExecResult(callID string, body io.Reader) error {
	s.mu.Lock()
	call, ok := s.execCalls[callID]
	delete(s.execCalls, callID)
	s.mu.Unlock()
	if !ok {
		return fmt.Errorf("unknown exec call %q", callID)
	}
	_, err := io.Copy(call.pw, body)
	_ = call.pw.Close()
	return err
}

// dropCall removes an in-flight call and closes its pipe with err.
func (s *Session) dropCall(callID string, pw io.WriteCloser) {
	s.mu.Lock()
	delete(s.execCalls, callID)
	s.mu.Unlock()
	_ = pw.Close()
}

// newCallID returns a fresh call id for this session.
func (s *Session) newCallID() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.execSeq++
	return fmt.Sprintf("call-%d", s.execSeq)
}
