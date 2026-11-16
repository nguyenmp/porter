package client

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"porter/internal/api"
)

// spoolLikeArguments builds a spool write's arguments with roughly size bytes of
// JSON-shaped content, so the line the server sends is escaped the way a real
// spool_output call is.
func spoolLikeArguments(t *testing.T, size int) string {
	t.Helper()
	var content strings.Builder
	for content.Len() < size {
		content.WriteString(`{"subdomain":"zengrid2764","users":"1","apps":"1"},` + "\n")
	}
	payload, err := json.Marshal(struct {
		Path    string `json:"path"`
		Content string `json:"content"`
	}{Path: "/tmp/orgs.json", Content: content.String()})
	if err != nil {
		t.Fatalf("marshal payload: %v", err)
	}
	return string(payload)
}

// TestServeExecReadsALongRequestLine checks that the exec stream dispatches a
// request line past bufio.Scanner's 64 KiB default. A spool write sends a whole
// tool result as one escaped string, so a 49 KB result became a 72 KB line, the
// scan failed, and the host never ran the write.
func TestServeExecReadsALongRequestLine(t *testing.T) {
	args := spoolLikeArguments(t, 1<<20)
	if len(args) < 64*1024 {
		t.Fatalf("arguments are %d bytes, too short to test the limit", len(args))
	}

	gotName := make(chan string, 1)
	uploads := make(chan int, 1)
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodGet {
			w.Header().Set("Content-Type", "application/x-ndjson")
			w.Header().Set(api.ExecResultFramingHeader, api.ExecResultFramingVersion)
			w.WriteHeader(http.StatusOK)
			_ = json.NewEncoder(w).Encode(api.ExecRequest{
				CallID:    "call-1",
				Name:      "_porter_spool_write",
				Arguments: args,
			})
			return
		}
		n, _ := io.Copy(io.Discard, r.Body)
		uploads <- int(n)
		w.WriteHeader(http.StatusOK)
	}))
	defer ts.Close()

	c := New(ts.URL, BasicAuth{})
	err := c.ServeExec(context.Background(), "s1", func(ctx context.Context, name string, a []byte) (io.ReadCloser, error) {
		if len(a) != len(args) {
			t.Errorf("dispatched %d argument bytes, want %d", len(a), len(args))
		}
		gotName <- name
		return io.NopCloser(strings.NewReader("wrote 49100 bytes to /tmp/orgs.json")), nil
	})
	if err != nil {
		t.Fatalf("ServeExec: %v", err)
	}

	select {
	case name := <-gotName:
		if name != "_porter_spool_write" {
			t.Errorf("dispatched tool = %q, want %q", name, "_porter_spool_write")
		}
	case <-time.After(2 * time.Second):
		t.Fatal("the long request line was never dispatched")
	}
	select {
	case n := <-uploads:
		if n == 0 {
			t.Error("the result upload carried no bytes")
		}
	case <-time.After(2 * time.Second):
		t.Fatal("no result upload arrived")
	}
}
