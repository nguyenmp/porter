package server

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"os/exec"
	"strings"
	"sync"
	"testing"
	"time"

	"porter/internal/client"
)

// TestViewDuringToolRunReportsNoResult covers a page reload while a tool call
// is still running. The stored rows hold the assistant message that asked for
// the call but no result yet, so the load-time repair fills one in, and a reader
// that trusts it shows a working tool as cancelled and interrupted. The server
// knows which calls are in flight, so the page must leave this one unanswered:
// it shows the call, and the live block that /runs feeds, until the real result
// commits.
func TestViewDuringToolRunReportsNoResult(t *testing.T) {
	var mu sync.Mutex
	calls := 0
	srv := newTestServer(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		mu.Lock()
		calls++
		first := calls == 1
		mu.Unlock()
		if !first {
			fmt.Fprint(w,
				`data: {"choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1}}`+"\n\n"+
					`data: [DONE]`+"\n")
			return
		}
		fmt.Fprint(w,
			`data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1","type":"function","function":{"name":"shell","arguments":"{\"command\":\"sleep 30\",\"porter_action_description\":\"slow command\",\"porter_timeout_seconds\":60}"}}]},"finish_reason":"tool_calls"}]}`+"\n\n"+
				`data: [DONE]`+"\n")
	})
	defer exec.Command("pkill", "-f", "^sleep 30$").Run() // tidy any stray survivor

	c := client.New(srv.URL)
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()

	info, err := c.Create(ctx)
	if err != nil {
		t.Fatalf("Create: %v", err)
	}
	if err := c.Append(ctx, info.ID, "run it"); err != nil {
		t.Fatalf("Append: %v", err)
	}

	// Wait for the run to reach the in-flight set.
	var callID string
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		rr, err := c.Runs(ctx, info.ID)
		if err != nil {
			t.Fatalf("Runs: %v", err)
		}
		if len(rr.Runs) > 0 {
			callID = rr.Runs[0].CallID
			break
		}
		time.Sleep(20 * time.Millisecond)
	}
	if callID == "" {
		t.Fatal("the tool never appeared on /runs")
	}

	// The page keeps the call, and reports no result for it.
	view := getBody(t, srv.URL+"/api/sessions/"+info.ID+"/view")
	if strings.Contains(view, "no result was recorded") {
		t.Errorf("view claims the running call was interrupted:\n%s", view)
	}
	if strings.Contains(view, `class="tool-result`) {
		t.Errorf("view rendered a result block for a call that is still running")
	}
	if !strings.Contains(view, "tool call: shell") {
		t.Errorf("view dropped the running call itself")
	}

	// The history API agrees: the call is unanswered, not answered with a
	// placeholder.
	h, err := c.History(ctx, info.ID)
	if err != nil {
		t.Fatalf("History: %v", err)
	}
	for _, m := range h.History {
		if m.Role == "tool" && m.ToolCallID == callID {
			t.Errorf("history answered the running call with %q", m.Content)
		}
	}

	// Once the call ends, its committed result renders as itself.
	if err := c.Cancel(ctx, info.ID, callID); err != nil {
		t.Fatalf("Cancel: %v", err)
	}
	deadline = time.Now().Add(5 * time.Second)
	for {
		view := getBody(t, srv.URL+"/api/sessions/"+info.ID+"/view")
		if strings.Contains(view, `class="tool-result`) {
			if strings.Contains(view, "no result was recorded") {
				t.Errorf("view still shows the interrupted marker after the call ended")
			}
			if got := strings.Count(view, `class="tool-result`); got != 1 {
				t.Errorf("result blocks after the call ended = %d, want 1", got)
			}
			return
		}
		if time.Now().After(deadline) {
			t.Fatalf("the cancelled result never reached /view")
		}
		time.Sleep(20 * time.Millisecond)
	}
}

// getBody GETs url and returns the body, failing the test on any error.
func getBody(t *testing.T, url string) string {
	t.Helper()
	resp, err := http.Get(url)
	if err != nil {
		t.Fatalf("GET %s: %v", url, err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("GET %s: status %d", url, resp.StatusCode)
	}
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("read %s: %v", url, err)
	}
	return string(body)
}
