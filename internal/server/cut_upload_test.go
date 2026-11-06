package server

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"
	"testing"
	"time"

	"porter/internal/client"
)

// cutReader stands in for a tool's output stream that dies partway through:
// what a dropped connection or a proxy idle-timeout does to the upload.
type cutReader struct {
	partial string
	stage   int
}

func (c *cutReader) Read(p []byte) (int, error) {
	if c.stage == 0 && c.partial != "" {
		c.stage++
		return copy(p, c.partial), nil
	}
	c.stage++
	return 0, errors.New("cut off for the test")
}

func (c *cutReader) Close() error { return nil }

// TestCutResultUploadReportsError guards the silent-loss bug: a tool result
// upload that ends early used to close the call's pipe cleanly, so the agent
// committed a truncated or empty result as a success and the model had no way
// to know its output was incomplete. Now the loss reaches the model as an
// error naming what happened.
func TestCutResultUploadReportsError(t *testing.T) {
	cases := []struct {
		name    string
		partial string
	}{
		{"partial output lost", "partial output\n"},
		{"no output at all", ""},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var mu sync.Mutex
			modelSaw := "<the model never got the result>"
			turns := 0
			srv := newTestServer(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				var req struct {
					Messages []struct {
						Role    string  `json:"role"`
						Content *string `json:"content"`
					} `json:"messages"`
				}
				if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
					t.Errorf("decode request: %v", err)
				}
				for _, m := range req.Messages {
					if m.Role == "tool" && m.Content != nil {
						mu.Lock()
						modelSaw = *m.Content
						mu.Unlock()
					}
				}
				w.Header().Set("Content-Type", "text/event-stream")
				turns++
				if turns == 1 {
					fmt.Fprint(w,
						`data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1","type":"function","function":{"name":"shell","arguments":"{\"command\":\"sleep 30\",\"porter_action_description\":\"run a quiet command\",\"porter_timeout_seconds\":60}"}}]},"finish_reason":"tool_calls"}]}`+"\n\n"+
							`data: [DONE]`+"\n")
					return
				}
				fmt.Fprint(w,
					`data: {"choices":[{"delta":{"content":"done"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1}}`+"\n\n"+
						`data: [DONE]`+"\n")
			}))

			c := client.New(srv.URL)
			ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
			defer cancel()
			info, err := c.Create(ctx)
			if err != nil {
				t.Fatalf("Create: %v", err)
			}

			// Serve as the session's remote execution provider, and wait until the
			// server has registered it so the tool call lands on it.
			execCtx, execCancel := context.WithCancel(ctx)
			defer execCancel()
			go func() {
				_ = c.ServeExec(execCtx, info.ID, func(context.Context, string, []byte) (io.ReadCloser, error) {
					return &cutReader{partial: tc.partial}, nil
				})
			}()
			deadline := time.Now().Add(5 * time.Second)
			for {
				st, serr := c.ExecStatus(ctx, info.ID)
				if serr != nil {
					t.Fatalf("ExecStatus: %v", serr)
				}
				if st.Kind == "remote" && st.Connected {
					break
				}
				if time.Now().After(deadline) {
					t.Fatalf("the test client never registered as the execution provider: %+v", st)
				}
				time.Sleep(20 * time.Millisecond)
			}

			if err := c.Append(ctx, info.ID, "run it"); err != nil {
				t.Fatalf("Append: %v", err)
			}

			// The committed result must carry the loss, not an empty success.
			var committed string
			deadline = time.Now().Add(10 * time.Second)
			for committed == "" && time.Now().Before(deadline) {
				h, herr := c.History(ctx, info.ID)
				if herr != nil {
					t.Fatalf("History: %v", herr)
				}
				for _, m := range h.History {
					if m.Role == "tool" && m.ToolCallID == "c1" {
						committed = m.Content
					}
				}
				if committed == "" {
					time.Sleep(20 * time.Millisecond)
				}
			}
			if committed == "" {
				t.Fatal("the cut upload committed no tool message at all")
			}
			if !strings.Contains(committed, "cut off") {
				t.Errorf("committed result does not report the loss: %q", committed)
			}
			if tc.partial != "" && !strings.Contains(committed, tc.partial) {
				t.Errorf("committed result dropped the output that arrived: %q", committed)
			}

			// The model reads the same text on the next request of the turn.
			deadline = time.Now().Add(5 * time.Second)
			for time.Now().Before(deadline) {
				mu.Lock()
				saw := modelSaw
				mu.Unlock()
				if strings.Contains(saw, "cut off") {
					return
				}
				time.Sleep(20 * time.Millisecond)
			}
			mu.Lock()
			defer mu.Unlock()
			t.Fatalf("the model never saw the cut reported: %q", modelSaw)
		})
	}
}
