package agent

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"

	"porter/internal/api"
	"porter/internal/config"
	"porter/internal/llm"
	"porter/internal/tools"
)

// toolServer serves one tool-calling turn then a plain-text reply, capturing
// each request's messages and whether tools were declared.
func toolServer(t *testing.T) (*httptest.Server, func() ([][]json.RawMessage, []bool)) {
	t.Helper()
	var mu sync.Mutex
	var captured [][]json.RawMessage
	var hadTools []bool

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req struct {
			Messages []json.RawMessage `json:"messages"`
			Tools    []json.RawMessage `json:"tools"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Errorf("decode request: %v", err)
		}
		mu.Lock()
		captured = append(captured, req.Messages)
		hadTools = append(hadTools, len(req.Tools) > 0)
		mu.Unlock()

		w.Header().Set("Content-Type", "text/event-stream")
		n := len(hadTools)
		if n == 1 {
			// First turn: ask for a tool call (arguments stream in pieces), with reasoning.
			fmt.Fprintf(w,
				`data: {"choices":[{"delta":{"reasoning_content":"deciding to call the shell"},"finish_reason":null}]}`+"\n\n"+
					`data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"shell","arguments":"{\"command\":\""}}]},"finish_reason":null}]}`+"\n\n"+
					`data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"echo hi\"}"}}]},"finish_reason":"tool_calls"}]}`+"\n\n"+
					`data: [DONE]`+"\n")
		} else {
			// Second turn: reply plainly (with reasoning), with usage.
			fmt.Fprintf(w,
				`data: {"choices":[{"delta":{"reasoning_content":"wrapping up"},"finish_reason":null}]}`+"\n\n"+
					`data: {"choices":[{"delta":{"content":"done"},"finish_reason":"stop"}],"usage":{"prompt_tokens":2,"completion_tokens":3}}`+"\n\n"+
					`data: [DONE]`+"\n")
		}
	}))
	return srv, func() ([][]json.RawMessage, []bool) {
		mu.Lock()
		defer mu.Unlock()
		return captured, hadTools
	}
}

func TestRunTurnExecutesToolAndLoops(t *testing.T) {
	srv, snapshot := toolServer(t)
	defer srv.Close()

	cfg := config.Config{BaseURL: srv.URL + "/v1", Model: "test", APIKey: "k"}
	client := llm.NewClient(cfg, nil)
	var text, jsonl bytes.Buffer

	emit := func(env api.Envelope) {
		switch env.Kind {
		case api.KindLLM:
			if env.Event == nil {
				return
			}
			ev := *env.Event
			EncodeJSON(&jsonl)(ev)
			Render(&text, false)(ev)
		case api.KindToolResult:
			data, _ := json.Marshal(env)
			jsonl.Write(append(data, '\n'))
		}
	}

	res, err := RunTurn(context.Background(), client, []llm.ChatMessage{llm.UserMessage("run it")}, tools.NewDispatcher(), emit, nil)
	if err != nil {
		t.Fatalf("RunTurn: %v", err)
	}

	if res.Text != "done" {
		t.Errorf("final text = %q, want %q", res.Text, "done")
	}
	if res.Usage.Input != 2 || res.Usage.Output != 3 {
		t.Errorf("usage = %+v, want 2 in / 3 out", res.Usage)
	}

	captured, hadTools := snapshot()
	if len(captured) != 2 {
		t.Fatalf("expected 2 requests, got %d", len(captured))
	}
	if !hadTools[0] || !hadTools[1] {
		t.Errorf("both requests should declare tools; got %v", hadTools)
	}

	// Second request must feed the tool result back in.
	found := false
	for _, m := range captured[1] {
		var msg struct {
			Role       string `json:"role"`
			ToolCallID string `json:"tool_call_id"`
			Content    string `json:"content"`
		}
		if err := json.Unmarshal(m, &msg); err != nil {
			t.Fatalf("unmarshal message: %v", err)
		}
		if msg.Role == "tool" && msg.ToolCallID == "call_1" && strings.Contains(msg.Content, "exit code: 0") {
			found = true
		}
	}
	if !found {
		t.Errorf("second request missing tool result; got %s", captured[1])
	}

	if !strings.Contains(text.String(), "shell") {
		t.Errorf("text view missing tool indicator; got:\n%s", text.String())
	}
	if !strings.Contains(jsonl.String(), `"type":"tool_call_delta"`) {
		t.Errorf("jsonl missing live tool_call_delta events; got:\n%s", jsonl.String())
	}
	if !strings.Contains(jsonl.String(), `"type":"tool_call"`) || !strings.Contains(jsonl.String(), `"kind":"tool_result"`) {
		t.Errorf("jsonl missing tool events; got:\n%s", jsonl.String())
	}
}

// TestRunTurnOnMessage commits each finalized message (assistant-with-calls,
// tool result, final reply) as it completes, in order.
func TestRunTurnOnMessage(t *testing.T) {
	srv, _ := toolServer(t)
	defer srv.Close()

	cfg := config.Config{BaseURL: srv.URL + "/v1", Model: "test", APIKey: "k"}
	client := llm.NewClient(cfg, nil)

	var got []llm.ChatMessage
	res, err := RunTurn(context.Background(), client, []llm.ChatMessage{llm.UserMessage("run it")}, tools.NewDispatcher(), nil, func(m llm.ChatMessage) error {
		got = append(got, m)
		return nil
	})
	if err != nil {
		t.Fatalf("RunTurn: %v", err)
	}

	var roles []string
	for _, m := range got {
		roles = append(roles, m.Role)
	}
	want := []string{"assistant", "tool", "assistant"}
	if len(roles) != len(want) {
		t.Fatalf("onMessage calls = %v, want %v", roles, want)
	}
	for i, w := range want {
		if roles[i] != w {
			t.Errorf("message[%d] role = %q, want %q", i, roles[i], w)
		}
	}
	// The first committed message carries the tool call AND the reasoning the
	// model streamed for that round (so a reload can render it).
	if len(got[0].ToolCalls) != 1 || got[0].ToolCalls[0].ID != "call_1" {
		t.Errorf("first committed message missing tool call; got %+v", got[0])
	}
	if !strings.Contains(got[0].Reasoning, "deciding to call the shell") {
		t.Errorf("first committed message missing streamed reasoning; got %+v", got[0])
	}
	// The tool result is keyed to that call.
	if got[1].Role != "tool" || got[1].ToolCallID != "call_1" {
		t.Errorf("tool result = %+v, want tool_call_id call_1", got[1])
	}
	// The final message is the plain answer, carrying its own reasoning.
	if got[2].Content != "done" || len(got[2].ToolCalls) != 0 {
		t.Errorf("final message = %+v, want content done", got[2])
	}
	if !strings.Contains(got[2].Reasoning, "wrapping up") {
		t.Errorf("final message missing reasoning; got %+v", got[2])
	}
	// onMessage output must match the assembled turn result.
	if !reflect.DeepEqual(got, res.History[1:]) {
		t.Errorf("onMessage order does not match turn history\ncommitted: %+v\nhistory:   %+v", got, res.History[1:])
	}
}

// chunkStream returns a fixed set of chunks, one per Read, so a test can assert
// that the agent emits one delta per chunk rather than a single buffered blob.
type chunkStream struct {
	chunks []string
	i      int
}

func (s *chunkStream) Read(p []byte) (int, error) {
	if s.i >= len(s.chunks) {
		return 0, io.EOF
	}
	n := copy(p, s.chunks[s.i])
	s.i++
	return n, nil
}

func (s *chunkStream) Close() error { return nil }

// fakeToolProvider streams a fixed set of chunks as a tool's output.
type fakeToolProvider struct {
	chunks []string
}

func (p *fakeToolProvider) Defs() []llm.Tool { return tools.Defs() }

func (p *fakeToolProvider) Run(ctx context.Context, name string, args []byte) (io.ReadCloser, error) {
	return &chunkStream{chunks: p.chunks}, nil
}

// TestRunTurnStreamsToolResultChunks verifies a tool's output is emitted live as
// tool_result_delta chunks and then reconciled by a terminal tool_result with
// the full result, while the committed tool message stores the full result.
func TestRunTurnStreamsToolResultChunks(t *testing.T) {
	srv, _ := toolServer(t)
	defer srv.Close()

	cfg := config.Config{BaseURL: srv.URL + "/v1", Model: "test", APIKey: "k"}
	client := llm.NewClient(cfg, nil)

	chunks := []string{"chunk one ", "chunk two ", "chunk three"}
	var got []api.Envelope
	res, err := RunTurn(context.Background(), client, []llm.ChatMessage{llm.UserMessage("run it")},
		&fakeToolProvider{chunks: chunks}, func(env api.Envelope) { got = append(got, env) }, nil)
	if err != nil {
		t.Fatalf("RunTurn: %v", err)
	}

	var streamed strings.Builder
	var deltas, terminals, starts int
	for _, env := range got {
		switch env.Kind {
		case api.KindToolResultDelta:
			deltas++
			streamed.WriteString(env.Delta)
		case api.KindToolResult:
			terminals++
			if env.Result != "chunk one chunk two chunk three" {
				t.Errorf("terminal result = %q, want full concatenation", env.Result)
			}
			// The terminal envelope carries server clocks and the arguments, so
			// the client can show a server-derived duration even if the start
			// signal was missed.
			if env.StartedAt <= 0 || env.FinishedAt < env.StartedAt {
				t.Errorf("terminal envelope timing = start %d finish %d, want start>0 and finish>=start", env.StartedAt, env.FinishedAt)
			}
			if env.Arguments == "" {
				t.Errorf("terminal envelope missing arguments")
			}
		case api.KindToolStarted:
			starts++
			if env.Name != "shell" || env.Arguments == "" || env.StartedAt <= 0 {
				t.Errorf("tool_started = name %q args %q started_at %d, want shell + args + start clock", env.Name, env.Arguments, env.StartedAt)
			}
		}
	}
	if deltas != len(chunks) {
		t.Errorf("tool_result_delta count = %d, want %d", deltas, len(chunks))
	}
	if streamed.String() != "chunk one chunk two chunk three" {
		t.Errorf("streamed deltas = %q, want full concatenation", streamed.String())
	}
	if terminals != 1 {
		t.Errorf("terminal tool_result count = %d, want 1", terminals)
	}
	if starts != 1 {
		t.Errorf("tool_started count = %d, want 1", starts)
	}

	// The committed history still stores the full result, plus the server
	// clocks so /view can render reload timing.
	var committed string
	var startedAt, finishedAt int64
	for _, m := range res.History {
		if m.Role == "tool" && m.ToolCallID == "call_1" {
			committed = m.Content
			startedAt = m.StartedAt
			finishedAt = m.FinishedAt
		}
	}
	if committed != "chunk one chunk two chunk three" {
		t.Errorf("committed tool result = %q, want full concatenation", committed)
	}
	if startedAt <= 0 || finishedAt < startedAt {
		t.Errorf("committed message timing = start %d finish %d, want start>0 and finish>=start", startedAt, finishedAt)
	}
}

// ctxStream is a tool-output stream that produces nothing until its context is
// cancelled, then returns EOF — simulating a long-running tool the user aborts.
type ctxStream struct {
	ctx context.Context
}

func (s *ctxStream) Read(p []byte) (int, error) {
	<-s.ctx.Done()
	return 0, io.EOF
}

func (s *ctxStream) Close() error { return nil }

// cancelProvider returns a stream that blocks until the run's context is
// cancelled, and records the stream so the test can observe when it ended.
type cancelProvider struct {
	stream *ctxStream
}

func (p *cancelProvider) Defs() []llm.Tool { return tools.Defs() }

func (p *cancelProvider) Run(ctx context.Context, name string, args []byte) (io.ReadCloser, error) {
	p.stream = &ctxStream{ctx: ctx}
	return p.stream, nil
}

// TestRunTurnCancelsRunningTool verifies the per-run cancellation hook: calling
// the cancel function a hook received stops the turn, emits a tool_cancelled
// envelope, and commits the partial tool result marked cancelled — instead of
// feeding it back to the model and looping.
func TestRunTurnCancelsRunningTool(t *testing.T) {
	srv, _ := toolServer(t) // first request asks for a tool call, then a reply
	defer srv.Close()

	cfg := config.Config{BaseURL: srv.URL + "/v1", Model: "test", APIKey: "k"}
	client := llm.NewClient(cfg, nil)

	var mu sync.Mutex
	var got []api.Envelope
	var cancelFn func()
	p := &cancelProvider{}
	hooks := RunHooks{OnRunStarted: func(callID string, cancel func()) {
		mu.Lock()
		cancelFn = cancel
		mu.Unlock()
	}}

	done := make(chan error, 1)
	var res TurnResult
	go func() {
		r, err := RunTurn(context.Background(), client, []llm.ChatMessage{llm.UserMessage("run it")},
			p, func(env api.Envelope) { got = append(got, env) }, nil, hooks)
		res = r
		done <- err
	}()

	// Wait until the tool started and the cancel func is registered.
	deadline := time.Now().Add(5 * time.Second)
	for {
		mu.Lock()
		cf := cancelFn
		mu.Unlock()
		if cf != nil {
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("cancel func never registered")
		}
		time.Sleep(10 * time.Millisecond)
	}

	cancelFn()
	select {
	case err := <-done:
		if !errors.Is(err, ErrToolCancelled) {
			t.Fatalf("RunTurn error = %v, want ErrToolCancelled", err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("RunTurn did not return after cancel")
	}

	// The run was cancelled: a tool_cancelled envelope (terminal, like
	// tool_result) went out, not a tool_result.
	var sawStarted, sawCancelled bool
	for _, env := range got {
		if env.Kind == api.KindToolStarted {
			sawStarted = true
		}
		if env.Kind == api.KindToolCancelled {
			sawCancelled = true
			if env.ToolCallID != "call_1" || env.Name != "shell" {
				t.Errorf("tool_cancelled = %+v, want call_1/shell", env)
			}
			if env.StartedAt <= 0 || env.FinishedAt < env.StartedAt {
				t.Errorf("tool_cancelled timing = start %d finish %d, want start>0 and finish>=start", env.StartedAt, env.FinishedAt)
			}
		}
		if env.Kind == api.KindToolResult {
			t.Errorf("normal tool_result should not be emitted for a cancelled run")
		}
	}
	if !sawStarted {
		t.Errorf("missing tool_started envelope")
	}
	if !sawCancelled {
		t.Errorf("missing tool_cancelled envelope; got %+v", got)
	}

	// The committed history ends at the cancelled tool result (marked
	// cancelled), and no final plain reply was produced — the turn stopped.
	var toolMsg *llm.ChatMessage
	for i := range res.History {
		if res.History[i].Role == "tool" {
			toolMsg = &res.History[i]
		}
	}
	if toolMsg == nil || !toolMsg.Cancelled || toolMsg.ToolCallID != "call_1" {
		t.Fatalf("committed tool message = %+v, want cancelled tool result for call_1", toolMsg)
	}
	if res.Text != "" {
		t.Errorf("turn text = %q, want empty (turn was cancelled before a reply)", res.Text)
	}
}

// TestRunTurnCancelsSilentToolCommitsContent guards the empty-result trap: a
// tool killed before it produced any output (sleep, a silent build) would
// otherwise commit a role-"tool" message with empty content, and the next LLM
// turn (which receives that message) would be rejected by providers that
// require a content field on tool messages. The cancelled run must commit (and
// emit) an explicit marker instead.
func TestRunTurnCancelsSilentToolCommitsContent(t *testing.T) {
	srv, _ := toolServer(t) // first request asks for a tool call, then a reply
	defer srv.Close()

	cfg := config.Config{BaseURL: srv.URL + "/v1", Model: "test", APIKey: "k"}
	client := llm.NewClient(cfg, nil)

	var mu sync.Mutex
	var got []api.Envelope
	var cancelFn func()
	p := &cancelProvider{}
	hooks := RunHooks{OnRunStarted: func(callID string, cancel func()) {
		mu.Lock()
		cancelFn = cancel
		mu.Unlock()
	}}

	done := make(chan error, 1)
	var res TurnResult
	go func() {
		r, err := RunTurn(context.Background(), client, []llm.ChatMessage{llm.UserMessage("run it")},
			p, func(env api.Envelope) { got = append(got, env) }, nil, hooks)
		res = r
		done <- err
	}()

	deadline := time.Now().Add(5 * time.Second)
	for {
		mu.Lock()
		cf := cancelFn
		mu.Unlock()
		if cf != nil {
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("cancel func never registered")
		}
		time.Sleep(10 * time.Millisecond)
	}
	cancelFn()

	select {
	case err := <-done:
		if !errors.Is(err, ErrToolCancelled) {
			t.Fatalf("RunTurn error = %v, want ErrToolCancelled", err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("RunTurn did not return after cancel")
	}

	// The committed tool message carries the "(cancelled)" marker so the next
	// turn's LLM request is well-formed (non-empty content).
	var toolMsg *llm.ChatMessage
	for i := range res.History {
		if res.History[i].Role == "tool" {
			toolMsg = &res.History[i]
		}
	}
	if toolMsg == nil {
		t.Fatalf("no committed tool message; history = %+v", res.History)
	}
	if !toolMsg.Cancelled {
		t.Errorf("committed tool message not marked cancelled: %+v", toolMsg)
	}
	if strings.TrimSpace(toolMsg.Content) == "" {
		t.Errorf("cancelled silent tool committed empty content %q; a tool message must carry content for the next turn", toolMsg.Content)
	}

	// The tool_cancelled envelope reconciles to the same marker.
	var sawCancelled bool
	for _, env := range got {
		if env.Kind == api.KindToolCancelled {
			sawCancelled = true
			if strings.TrimSpace(env.Result) == "" {
				t.Errorf("tool_cancelled envelope result = %q, want non-empty marker", env.Result)
			}
		}
	}
	if !sawCancelled {
		t.Errorf("missing tool_cancelled envelope; got %+v", got)
	}
}

// usageAfterFinishServer serves a plain reply in the OpenAI-compatible shape:
// usage arrives in a separate chunk AFTER the finish_reason chunk (with empty
// choices). When withDone is false it omits the [DONE] marker, exercising the
// EOF path that Final() must flush.
func usageAfterFinishServer(t *testing.T, withDone bool) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprintf(w,
			`data: {"choices":[{"delta":{"content":"done"},"finish_reason":null}]}`+"\n\n"+
				`data: {"choices":[{"delta":{},"finish_reason":"stop"}]}`+"\n\n"+
				`data: {"choices":[],"usage":{"prompt_tokens":2,"completion_tokens":3}}`+"\n\n")
		if withDone {
			fmt.Fprint(w, `data: [DONE]`+"\n")
		}
	}))
	return srv
}

// TestRunTurnUsageArrivesAfterFinishReason reproduces the exact bug: because
// the decoder stopped at finish_reason it never read the trailing usage chunk,
// so usage came back 0/0. After the fix the loop keeps reading to [DONE] and
// the separate usage chunk is captured.
func TestRunTurnUsageArrivesAfterFinishReason(t *testing.T) {
	srv := usageAfterFinishServer(t, true)
	defer srv.Close()

	cfg := config.Config{BaseURL: srv.URL + "/v1", Model: "test", APIKey: "k"}
	client := llm.NewClient(cfg, nil)
	res, err := RunTurn(context.Background(), client, []llm.ChatMessage{llm.UserMessage("hi")}, tools.NewDispatcher(), nil, nil)
	if err != nil {
		t.Fatalf("RunTurn: %v", err)
	}
	if res.Text != "done" {
		t.Errorf("final text = %q, want %q", res.Text, "done")
	}
	if res.Usage.Input != 2 || res.Usage.Output != 3 {
		t.Errorf("usage = %+v, want 2 in / 3 out", res.Usage)
	}
}

// TestRunTurnUsageFlushedOnEOFWithoutDone covers a provider that closes the
// SSE stream after the usage chunk without a [DONE] marker: Final() must flush
// the terminal events (and the usage) so the turn still records its tokens.
func TestRunTurnUsageFlushedOnEOFWithoutDone(t *testing.T) {
	srv := usageAfterFinishServer(t, false) // no [DONE]: EOF path -> dec.Final()
	defer srv.Close()

	cfg := config.Config{BaseURL: srv.URL + "/v1", Model: "test", APIKey: "k"}
	client := llm.NewClient(cfg, nil)
	res, err := RunTurn(context.Background(), client, []llm.ChatMessage{llm.UserMessage("hi")}, tools.NewDispatcher(), nil, nil)
	if err != nil {
		t.Fatalf("RunTurn: %v", err)
	}
	if res.Text != "done" {
		t.Errorf("final text = %q, want %q", res.Text, "done")
	}
	if res.Usage.Input != 2 || res.Usage.Output != 3 {
		t.Errorf("usage = %+v, want 2 in / 3 out (Final must flush the trailing usage chunk)", res.Usage)
	}
}

// TestRunTurnReportsEachQuery verifies the OnQuery hook fires once per model
// request, at the query's origin, with that request's usage. A tool-calling
// turn spans two queries: the tool call (the fixture sends no usage) and the
// final reply (usage 2/3). The summed turn usage still aggregates the same way.
func TestRunTurnReportsEachQuery(t *testing.T) {
	srv, _ := toolServer(t)
	defer srv.Close()

	cfg := config.Config{BaseURL: srv.URL + "/v1", Model: "test", APIKey: "k"}
	client := llm.NewClient(cfg, nil)
	var mu sync.Mutex
	var queries []Query
	hooks := RunHooks{OnQuery: func(q Query) error {
		mu.Lock()
		defer mu.Unlock()
		queries = append(queries, q)
		return nil
	}}

	res, err := RunTurn(context.Background(), client, []llm.ChatMessage{llm.UserMessage("run it")}, tools.NewDispatcher(), nil, nil, hooks)
	if err != nil {
		t.Fatalf("RunTurn: %v", err)
	}

	mu.Lock()
	defer mu.Unlock()
	if len(queries) != 2 {
		t.Fatalf("OnQuery calls = %+v, want 2 (one per request)", queries)
	}
	if queries[0].Idx != 0 || queries[0].Input != 0 || queries[0].Output != 0 || queries[0].Err != nil {
		t.Errorf("query 0 = %+v, want idx 0, no usage, no error", queries[0])
	}
	if queries[1].Idx != 1 || queries[1].Input != 2 || queries[1].Output != 3 || queries[1].Err != nil {
		t.Errorf("query 1 = %+v, want idx 1, usage 2/3, no error", queries[1])
	}
	if res.Usage.Input != 2 || res.Usage.Output != 3 {
		t.Errorf("turn usage = %+v, want 2/3", res.Usage)
	}
}

// TestRunTurnReportsFailedQuery verifies OnQuery is called with the error when
// a request fails (e.g. a provider 400), so the failure is persisted at the
// query's origin rather than only surfacing on the turn_completed marker.
func TestRunTurnReportsFailedQuery(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, `{"error":{"message":"boom"}}`, http.StatusBadRequest)
	}))
	defer srv.Close()

	cfg := config.Config{BaseURL: srv.URL + "/v1", Model: "test", APIKey: "k"}
	client := llm.NewClient(cfg, nil)
	var mu sync.Mutex
	var queries []Query
	hooks := RunHooks{OnQuery: func(q Query) error {
		mu.Lock()
		defer mu.Unlock()
		queries = append(queries, q)
		return nil
	}}

	_, err := RunTurn(context.Background(), client, []llm.ChatMessage{llm.UserMessage("hi")}, tools.NewDispatcher(), nil, nil, hooks)
	if err == nil {
		t.Fatal("RunTurn returned nil error on a failed request")
	}

	mu.Lock()
	defer mu.Unlock()
	if len(queries) != 1 {
		t.Fatalf("OnQuery calls = %+v, want 1", queries)
	}
	if queries[0].Err == nil || !strings.Contains(queries[0].Err.Error(), "boom") {
		t.Errorf("query error = %v, want the provider's 'boom'", queries[0].Err)
	}
}
