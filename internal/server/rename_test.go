package server

import (
	"context"
	"io"
	"net/http"
	"strings"
	"testing"
	"time"

	"porter/internal/api"
	"porter/internal/client"
)

// TestWebRendersRenameControl verifies the chat page renders the rename UI:
// the header title carries the custom name (with the raw name and preview
// fallback as data attributes), the pencil and input are present, and the SSE
// stream subscribes to session_renamed so the title and sidebar update live.
func TestWebRendersRenameControl(t *testing.T) {
	srv := newTestServer(t, plainLLM())
	c := client.New(srv.URL)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	info, err := c.Create(ctx)
	if err != nil {
		t.Fatalf("Create: %v", err)
	}
	if err := c.Rename(ctx, info.ID, "Deploy notes"); err != nil {
		t.Fatalf("Rename: %v", err)
	}

	resp, err := http.Get(srv.URL + "/?session=" + info.ID)
	if err != nil {
		t.Fatalf("GET /: %v", err)
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	s := string(body)

	for _, want := range []string{
		`id="chat-title"`,
		`data-name="Deploy notes"`,
		">Deploy notes<",
		`id="rename-btn"`,
		`id="rename-input"`,
		`sse-swap="message_committed,llm,tool_started,tool_result,tool_result_delta,tool_cancelled,turn_completed,exec_status,session_renamed"`,
	} {
		if !strings.Contains(s, want) {
			t.Errorf("page missing %q", want)
		}
	}
	// And the sidebar list carries the name too.
	var list api.SessionsResponse
	if err := fetchJSON(srv.URL+"/api/sessions", &list); err != nil {
		t.Fatalf("list: %v", err)
	}
	if len(list.Sessions) != 1 || list.Sessions[0].Name != "Deploy notes" {
		t.Fatalf("list = %+v, want one renamed row", list.Sessions)
	}
}
