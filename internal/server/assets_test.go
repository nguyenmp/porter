package server

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"testing"

	"porter/internal/api"
)

// TestAssetServing publishes files through a real session and fetches them
// over HTTP: an image serves with its content type and nosniff (never
// sniffed), an HTML asset additionally gets the sandboxing CSP so content
// produced by the model can never script the porter origin, and unknown or
// malformed asset paths 404 without touching the filesystem.
func TestAssetServing(t *testing.T) {
	dbDir := t.TempDir()
	llm := func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
	}
	s, ts := startServerDB(t, filepath.Join(dbDir, "porter.db"), llm)

	ses, err := s.store.Create(nil)
	if err != nil {
		t.Fatalf("Create: %v", err)
	}
	png, err := s.store.PublishAsset(ses.ID(), "chart.png", []byte{0x89, 'P', 'N', 'G', 0x0d, 0x0a, 0x1a, 0x0a})
	if err != nil {
		t.Fatalf("publish png: %v", err)
	}
	html, err := s.store.PublishAsset(ses.ID(), "page.html", []byte("<h1>hi</h1>"))
	if err != nil {
		t.Fatalf("publish html: %v", err)
	}

	rest := strings.TrimPrefix(png, "/assets/sessions/"+ses.ID()+"/")
	parts := strings.SplitN(rest, "/", 2)
	if _, ferr := s.store.AssetFilePath(ses.ID(), parts[0], parts[1]); ferr != nil {
		t.Fatalf("store cannot resolve its own URL %q: %v", png, ferr)
	}

	check := func(url, wantType string, wantCSP bool) {
		t.Helper()
		resp, err := http.Get(ts.URL + url)
		if err != nil {
			t.Fatalf("GET %s: %v", url, err)
		}
		defer resp.Body.Close()
		if resp.StatusCode != http.StatusOK {
			t.Fatalf("GET %s = %d, want 200", url, resp.StatusCode)
		}
		if ct := resp.Header.Get("Content-Type"); ct != wantType {
			t.Errorf("GET %s content-type = %q, want %q", url, ct, wantType)
		}
		if got := resp.Header.Get("X-Content-Type-Options"); got != "nosniff" {
			t.Errorf("GET %s nosniff = %q, want nosniff", url, got)
		}
		csp := resp.Header.Get("Content-Security-Policy")
		if wantCSP && csp != "sandbox allow-scripts" {
			t.Errorf("GET %s CSP = %q, want sandbox allow-scripts", url, csp)
		} else if !wantCSP && csp != "" {
			t.Errorf("GET %s CSP = %q, want none", url, csp)
		}
	}
	check(png, "image/png", false)
	check(html, "text/html; charset=utf-8", true)

	for _, bad := range []string{
		"/assets/sessions/" + ses.ID() + "/zzzzzzzz/chart.png", // unknown asset id
		"/assets/sessions/nope/aaaaaaaa/chart.png",             // unknown session
	} {
		resp, err := http.Get(ts.URL + bad)
		if err != nil {
			t.Fatalf("GET %s: %v", bad, err)
		}
		resp.Body.Close()
		if resp.StatusCode != http.StatusNotFound {
			t.Errorf("GET %s = %d, want 404", bad, resp.StatusCode)
		}
	}
	if !strings.HasPrefix(png, "/assets/sessions/"+ses.ID()+"/") {
		t.Errorf("published URL %q does not carry the session id", png)
	}
}

// TestRenderIframeView verifies the render_iframe special case end to end: a
// model that calls render_iframe gets a committed caption, and the /view
// history renders that tool result as an always-expanded sandboxed iframe
// (never a collapsed details/code block), matching what the live client
// draws.
func TestRenderIframeView(t *testing.T) {
	llm := iframeThenReplyLLM()
	_, ts := startServerDB(t, filepath.Join(t.TempDir(), "porter.db"), llm)
	id, got, _ := runOneTurnID(t, ts.URL, "show the page")

	const src = "/assets/sessions/session_1/0a0a0a0a/page.html"
	var sawCaption bool
	for _, env := range got {
		if env.Kind != api.KindToolResult || env.Name != "render_iframe" {
			continue
		}
		if !strings.Contains(env.Result, src) {
			t.Errorf("render_iframe result = %q, want it to name %q", env.Result, src)
		}
		sawCaption = true
	}
	if !sawCaption {
		t.Fatalf("no render_iframe tool_result on the bus; envelopes = %+v", got)
	}
	view, err := http.Get(ts.URL + "/api/sessions/" + id + "/view")
	if err != nil {
		t.Fatalf("GET /view: %v", err)
	}
	defer view.Body.Close()
	body := new(strings.Builder)
	_, _ = io.Copy(body, view.Body)
	html := body.String()
	if !strings.Contains(html, `class="tool-iframe-frame"`) {
		t.Errorf("/view missing the iframe frame element:\n%s", html)
	}
	if !strings.Contains(html, `sandbox="allow-scripts"`) {
		t.Errorf("/view iframe missing the sandbox attribute:\n%s", html)
	}
	if !strings.Contains(html, `src="`+src+`"`) {
		t.Errorf("/view iframe missing src %q:\n%s", src, html)
	}
	if strings.Contains(html, "Rendered in a sandboxed iframe") && strings.Contains(html, "<pre><code>") {
		t.Errorf("/view rendered the render_iframe result as a collapsed code block instead of an iframe:\n%s", html)
	}
}

// iframeThenReplyLLM asks for a render_iframe call on the first request, then
// replies plainly on the second.
func iframeThenReplyLLM() http.HandlerFunc {
	var mu sync.Mutex
	n := 0
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		mu.Lock()
		n++
		call := n
		mu.Unlock()
		if call == 1 {
			fmt.Fprint(w,
				`data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1","type":"function","function":{"name":"render_iframe","arguments":"{\"src\":\"/assets/sessions/session_1/0a0a0a0a/page.html\",\"height\":600,\"porter_action_description\":\"show the generated page\",\"porter_timeout_seconds\":30}"}}]},"finish_reason":"tool_calls"}]}`+"\n\n"+
					`data: [DONE]`+"\n")
			return
		}
		fmt.Fprint(w,
			`data: {"choices":[{"delta":{"content":"done"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1}}`+"\n\n"+
				`data: [DONE]`+"\n")
	}
}

// readImageThenReplyLLM asks for a read_image call on the given file on the
// first request, then replies plainly on the second.
func readImageThenReplyLLM(path string) http.HandlerFunc {
	args, err := json.Marshal(map[string]any{
		"path":                      path,
		"porter_action_description": "look at the screenshot",
		"porter_timeout_seconds":    30,
	})
	if err != nil {
		panic(err)
	}
	var mu sync.Mutex
	n := 0
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		mu.Lock()
		n++
		call := n
		mu.Unlock()
		if call == 1 {
			fmt.Fprint(w,
				`data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1","type":"function","function":{"name":"read_image","arguments":`+strconv.Quote(string(args))+`}}]},"finish_reason":"tool_calls"}]}`+"\n\n"+
					`data: [DONE]`+"\n")
			return
		}
		fmt.Fprint(w,
			`data: {"choices":[{"delta":{"content":"done"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1}}`+"\n\n"+
				`data: [DONE]`+"\n")
	}
}

// TestReadImageViewDrawsPicture drives a whole read_image turn through the
// server and checks that the picture reaches both views: the live tool_result
// envelope carries it in its metadata, and /view draws it below the collapsed
// code box, outside the <details>, where a reload shows it without a click.
func TestReadImageViewDrawsPicture(t *testing.T) {
	pic := filepath.Join(t.TempDir(), "shot.png")
	if err := os.WriteFile(pic, tinyPNG(t), 0o644); err != nil {
		t.Fatalf("write picture: %v", err)
	}
	_, ts := startServerDB(t, filepath.Join(t.TempDir(), "porter.db"), readImageThenReplyLLM(pic))
	id, got, _ := runOneTurnID(t, ts.URL, "look at shot.png")

	var url string
	for _, env := range got {
		if env.Kind != api.KindToolResult || env.Name != "read_image" {
			continue
		}
		if env.ToolOutput == nil || len(env.ToolOutput.Images) != 1 {
			t.Fatalf("read_image result carries no picture: %+v", env.ToolOutput)
		}
		url = env.ToolOutput.Images[0].URL
	}
	if url == "" {
		t.Fatalf("no read_image tool_result on the bus; envelopes = %+v", got)
	}
	if !strings.HasPrefix(url, "/assets/sessions/"+id+"/") {
		t.Errorf("picture URL = %q, want an asset published for this session", url)
	}

	view, err := http.Get(ts.URL + "/api/sessions/" + id + "/view")
	if err != nil {
		t.Fatalf("GET /view: %v", err)
	}
	defer view.Body.Close()
	body := new(strings.Builder)
	_, _ = io.Copy(body, view.Body)
	html := body.String()
	if !strings.Contains(html, `class="tool-images"`) {
		t.Errorf("/view has no picture strip:\n%s", html)
	}
	if !strings.Contains(html, `src="`+url+`"`) {
		t.Errorf("/view does not draw the picture %q:\n%s", url, html)
	}
	if !strings.Contains(html, `href="`+url+`"`) {
		t.Errorf("/view does not link the full-size picture %q:\n%s", url, html)
	}
	// Outside the <details>: a closed result hides what it holds, so a picture
	// inside one would never be seen.
	if close := strings.Index(html, "</details>"); close < 0 || strings.Index(html, `class="tool-images"`) < close {
		t.Errorf("/view put the picture inside the tool result's closed <details>:\n%s", html)
	}

	// The URL it draws must actually serve the picture.
	resp, err := http.Get(ts.URL + url)
	if err != nil {
		t.Fatalf("GET %s: %v", url, err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK || resp.Header.Get("Content-Type") != "image/png" {
		t.Errorf("GET %s = %d %q, want 200 image/png", url, resp.StatusCode, resp.Header.Get("Content-Type"))
	}
}
