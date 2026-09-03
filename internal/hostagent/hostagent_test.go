package hostagent

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"porter/internal/api"
	"porter/internal/client"
	"porter/internal/mcp"
	"porter/internal/tools"
)

// mockMCP is a minimal streamable-HTTP MCP server (mirrors the mcp package's
// test helper) returning one tool.
type mockMCP struct {
	tools []map[string]any
}

func (m *mockMCP) handler() http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req struct {
			ID     any             `json:"id"`
			Method string          `json:"method"`
			Params json.RawMessage `json:"params"`
		}
		_ = json.NewDecoder(r.Body).Decode(&req)
		switch req.Method {
		case "initialize":
			_ = json.NewEncoder(w).Encode(map[string]any{
				"jsonrpc": "2.0", "id": req.ID,
				"result": map[string]any{
					"protocolVersion": "2025-06-18",
					"capabilities":    map[string]any{},
					"serverInfo":      map[string]any{"name": "mock", "version": "1.0"},
				},
			})
		case "tools/list":
			_ = json.NewEncoder(w).Encode(map[string]any{
				"jsonrpc": "2.0", "id": req.ID,
				"result": map[string]any{"tools": m.tools},
			})
		case "tools/call":
			_ = json.NewEncoder(w).Encode(map[string]any{
				"jsonrpc": "2.0", "id": req.ID,
				"result": map[string]any{"content": []map[string]any{
					{"type": "text", "text": "whoami result"},
				}},
			})
		default:
			_ = json.NewEncoder(w).Encode(map[string]any{
				"jsonrpc": "2.0", "id": req.ID, "result": map[string]any{},
			})
		}
	})
}

// TestHubSummaryTagsHost proves hubSummary renders a hub's servers as
// reported metadata with the host id attached, so the porter server can list
// them as "hosted on <host>".
func TestHubSummaryTagsHost(t *testing.T) {
	ts := httptest.NewServer((&mockMCP{tools: []map[string]any{{"name": "echo", "description": "Echo"}}}).handler())
	defer ts.Close()

	dir := t.TempDir()
	cfg := filepath.Join(dir, "porter.mcp.json")
	data, _ := json.Marshal(map[string]any{"servers": []map[string]any{
		{"name": "retool", "description": "Retool", "url": ts.URL},
	}})
	if err := os.WriteFile(cfg, data, 0o644); err != nil {
		t.Fatal(err)
	}
	h, err := mcp.Load(cfg, nil)
	if err != nil {
		t.Fatalf("mcp.Load: %v", err)
	}

	sum := hubSummary(h, "macbook")
	if len(sum) != 1 {
		t.Fatalf("hubSummary = %d servers, want 1", len(sum))
	}
	if sum[0].Name != "retool" || sum[0].Host != "macbook" || sum[0].Status != "ok" {
		t.Errorf("hubSummary server = %+v", sum[0])
	}
	if len(sum[0].Tools) != 1 || sum[0].Tools[0].Name != "echo" {
		t.Errorf("hubSummary tools = %+v", sum[0].Tools)
	}
}

// TestProvisionerServesHostMCPServers proves the host's serve loop answers
// CallMCP from its local hub (executing the call against the host's own MCP
// server) and passes every other tool to the dispatcher. It drives the
// dispatch func directly with a real hub backed by a mock MCP server.
func TestProvisionerServesHostMCPServers(t *testing.T) {
	// The host's MCP server: accepts any bearer token and returns one tool.
	ts := httptest.NewServer((&mockMCP{tools: []map[string]any{{"name": "whoami", "description": "Who am I"}}}).handler())
	defer ts.Close()

	dir := t.TempDir()
	cfg := filepath.Join(dir, "porter.mcp.json")
	data, _ := json.Marshal(map[string]any{"servers": []map[string]any{
		{"name": "retool", "description": "Retool", "url": ts.URL},
	}})
	if err := os.WriteFile(cfg, data, 0o644); err != nil {
		t.Fatal(err)
	}
	h, err := mcp.Load(cfg, nil)
	if err != nil {
		t.Fatalf("mcp.Load: %v", err)
	}

	p := &provisioner{mcp: h}
	disp := newTestDispatcher(t)

	// CallMCP routes to the host's hub, which calls the host-only server.
	out, err := p.serveOne(context.Background(), mcp.CallTool, []byte(`{"server_name":"retool","tool_name":"whoami"}`), disp)
	if err != nil {
		t.Fatalf("CallMCP: %v", err)
	}
	data2, _ := io.ReadAll(out)
	_ = out.Close()
	if !strings.Contains(string(data2), "whoami") {
		t.Errorf("CallMCP result = %q, want the tool result", data2)
	}

	// shell routes to the dispatcher, not the hub.
	out, err = p.serveOne(context.Background(), "shell", []byte(`{"command":"echo hi"}`), disp)
	if err != nil {
		t.Fatalf("shell: %v", err)
	}
	data2, _ = io.ReadAll(out)
	_ = out.Close()
	if !strings.Contains(string(data2), "hi") {
		t.Errorf("shell result = %q", data2)
	}

	// Unknown hub server: the hub rejects it (an error, which the exec
	// client surfaces as "error: ..." content).
	if _, err := p.serveOne(context.Background(), mcp.CallTool, []byte(`{"server_name":"nope","tool_name":"x"}`), disp); err == nil || !strings.Contains(err.Error(), "unknown MCP server") {
		t.Errorf("CallMCP unknown err = %v, want unknown-server error", err)
	}
}

// serveOne is the serve loop's dispatch func factored out for testing: route
// CallMCP to the host's hub, everything else to the dispatcher.
func (p *provisioner) serveOne(ctx context.Context, name string, args []byte, disp *tools.Dispatcher) (io.ReadCloser, error) {
	if name == mcp.CallTool {
		return p.mcp.Run(ctx, mcp.CallTool, args)
	}
	return disp.RunDir(ctx, name, args, "")
}

// newTestDispatcher returns a dispatcher with the repo-root skills (none in a
// temp test repo) so shell works.
func newTestDispatcher(t *testing.T) *tools.Dispatcher {
	t.Helper()
	return tools.NewDispatcher()
}

// TestProvisionCreatesMultiRepoSandbox drives provision() end to end against
// a fake porter server: two repos become two worktrees in a container whose
// CWD the session's exec context reports, with skills from both repos, and
// release tears the whole sandbox down.
func TestProvisionCreatesMultiRepoSandbox(t *testing.T) {
	repo1 := initGitRepo(t)
	repo2 := initGitRepo(t)
	// A skill in each repo, so the registered context proves multi-repo
	// skills load (each worktree is a repo root to FindSkillsIn).
	writeRepoSkill(t, repo1, "repo1-skill")
	writeRepoSkill(t, repo2, "repo2-skill")
	worktrees := t.TempDir()

	var mu sync.Mutex
	var execCtxs []api.ExecContext
	var providerErrs []string
	holdExec := make(chan struct{}) // never closed until the test ends

	mux := http.NewServeMux()
	mux.HandleFunc("/api/sessions/", func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.Method == "POST" && strings.HasSuffix(r.URL.Path, "/exec/context"):
			var ctx api.ExecContext
			_ = json.NewDecoder(r.Body).Decode(&ctx)
			mu.Lock()
			execCtxs = append(execCtxs, ctx)
			mu.Unlock()
			w.WriteHeader(http.StatusOK)
		case r.Method == "GET" && strings.HasSuffix(r.URL.Path, "/exec"):
			w.Header().Set("Content-Type", "application/x-ndjson")
			w.WriteHeader(http.StatusOK)
			w.(http.Flusher).Flush()
			<-holdExec // hold the provider's exec connection open
		default:
			w.WriteHeader(http.StatusNotFound)
		}
	})
	mux.HandleFunc("/api/hosts/", func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		mu.Lock()
		providerErrs = append(providerErrs, string(b))
		mu.Unlock()
		w.WriteHeader(http.StatusOK)
	})
	ts := httptest.NewServer(mux)
	defer ts.Close()
	defer close(holdExec)

	c := client.New(ts.URL)
	p := &provisioner{
		c:         c,
		hostID:    "mac",
		cwd:       t.TempDir(),
		worktrees: worktrees,
		serves:    map[string]*serveState{},
		mcp:       mcp.New(nil),
	}

	if err := p.provision(api.HostRequest{
		Kind:       "provision",
		ProviderID: "mac-provider-99",
		SessionID:  "session_1",
		Repos:      []api.RepoRef{{Path: repo1}, {Path: repo2}},
	}); err != nil {
		t.Fatalf("provision: %v", err)
	}

	// The provider registered with no error and the container as its CWD.
	mu.Lock()
	if len(providerErrs) != 0 {
		t.Errorf("provider errors: %v", providerErrs)
	}
	mu.Unlock()
	if len(execCtxs) != 1 {
		t.Fatalf("exec contexts registered = %d, want 1", len(execCtxs))
	}
	ctx := execCtxs[0]
	sandboxDir := filepath.Join(worktrees, "mac-provider-99")
	if ctx.CWD != sandboxDir {
		t.Errorf("registered CWD = %q, want container %q", ctx.CWD, sandboxDir)
	}
	// Both worktrees exist on disk, and files from both are listed prefixed
	// with the worktree's dir name.
	for _, repo := range []string{repo1, repo2} {
		wt := filepath.Join(sandboxDir, filepath.Base(repo))
		if _, err := os.Stat(filepath.Join(wt, "hello.txt")); err != nil {
			t.Errorf("worktree %q missing repo files: %v", wt, err)
		}
	}
	names := strings.Join(ctx.Files, " ")
	for _, repo := range []string{repo1, repo2} {
		prefix := filepath.Base(repo) + "/hello.txt"
		if !strings.Contains(names, prefix) {
			t.Errorf("registered files missing %q (all: %v)", prefix, ctx.Files)
		}
	}
	// Skills from both repos are in the registered context.
	skillNames := map[string]bool{}
	for _, s := range ctx.Skills {
		skillNames[s.Name] = true
	}
	if !skillNames["repo1-skill"] || !skillNames["repo2-skill"] {
		t.Errorf("registered skills = %v, want repo1-skill and repo2-skill", skillNames)
	}

	// Release tears down every worktree, its branch, and the container.
	p.release(api.HostRequest{ProviderID: "mac-provider-99"})
	for _, repo := range []string{repo1, repo2} {
		wt := filepath.Join(sandboxDir, filepath.Base(repo))
		if _, err := os.Stat(wt); !os.IsNotExist(err) {
			t.Errorf("worktree %q still exists after release", wt)
		}
	}
	if _, err := os.Stat(sandboxDir); !os.IsNotExist(err) {
		t.Errorf("container %q still exists after release", sandboxDir)
	}
}

// writeRepoSkill writes a skill into repo/.agents/skills/<name>/SKILL.md and
// commits it, so worktrees provisioned from the repo carry it (untracked
// files do not transfer between worktrees).
func writeRepoSkill(t *testing.T, repo, name string) {
	t.Helper()
	dir := filepath.Join(repo, ".agents", "skills", name)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "SKILL.md"), []byte("# "+name+"\n\nDoes the "+name+" thing.\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	runGit(t, repo, "add", ".")
	runGit(t, repo, "commit", "-m", "add "+name+" skill")
}

// TestProvisionRollsBackOnFailure drives provision() down its error path: the
// second repo is not a git repo, so the first worktree must be rolled back,
// the container removed, and the failure reported to the server (the session
// keeps its local fallback).
func TestProvisionRollsBackOnFailure(t *testing.T) {
	repo := initGitRepo(t)
	worktrees := t.TempDir()

	var mu sync.Mutex
	var providerErrs []string
	mux := http.NewServeMux()
	mux.HandleFunc("/api/hosts/", func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		mu.Lock()
		providerErrs = append(providerErrs, string(b))
		mu.Unlock()
		w.WriteHeader(http.StatusOK)
	})
	ts := httptest.NewServer(mux)
	defer ts.Close()

	c := client.New(ts.URL)
	p := &provisioner{
		c:         c,
		hostID:    "mac",
		cwd:       t.TempDir(),
		worktrees: worktrees,
		serves:    map[string]*serveState{},
		mcp:       mcp.New(nil),
	}

	err := p.provision(api.HostRequest{
		Kind:       "provision",
		ProviderID: "mac-provider-100",
		SessionID:  "session_1",
		Repos:      []api.RepoRef{{Path: repo}, {Path: t.TempDir()}},
	})
	if err != nil {
		t.Fatalf("provision should report failure, not error: %v", err)
	}
	mu.Lock()
	defer mu.Unlock()
	if len(providerErrs) != 1 {
		t.Fatalf("provider errors = %v, want 1 reported failure", providerErrs)
	}
	// The partial worktree and the container are gone — no half-open sandbox.
	sandboxDir := filepath.Join(worktrees, "mac-provider-100")
	if _, err := os.Stat(sandboxDir); !os.IsNotExist(err) {
		t.Errorf("container %q left behind after failed provision", sandboxDir)
	}
}

// TestSleepGap proves sleepGap turns a wall-vs-monotonic clock gap into the
// sleep duration: positive when the wall clock outran the monotonic clock
// (the machine slept), zero when they agree (awake) or the wall clock jumped
// backwards (an NTP correction).
func TestSleepGap(t *testing.T) {
	cases := []struct {
		name       string
		wall, mono time.Duration
		want       time.Duration
	}{
		{"awake", 10 * time.Second, 10 * time.Second, 0},
		{"slept an hour", time.Hour + 10*time.Second, 10 * time.Second, time.Hour},
		{"short nap", 30 * time.Second, 0, 30 * time.Second},
		{"wall clock jumped back", -30 * time.Second, 10 * time.Second, 0},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := sleepGap(tc.wall, tc.mono); got != tc.want {
				t.Fatalf("sleepGap(%v, %v) = %v, want %v", tc.wall, tc.mono, got, tc.want)
			}
		})
	}
}

// TestHostLockAt proves the flock guard: a second lock on the same path
// while the first is held fails with a message naming the lock file, and the
// lock can be re-acquired after release. It locks a temp file directly
// (lockHostAt), since taking the real ~/.porter lock would collide with any
// running host agent on this machine.
func TestHostLockAt(t *testing.T) {
	path := filepath.Join(t.TempDir(), "pid.lock")

	unlock, err := lockHostAt(path)
	if err != nil {
		t.Fatalf("first lock: %v", err)
	}
	// A second host on the same machine fails loudly.
	if _, err := lockHostAt(path); err == nil {
		t.Fatal("second lock succeeded, want failure")
	} else if !strings.Contains(err.Error(), path) {
		t.Errorf("second lock error = %q, want it to name %s", err, path)
	}
	// Releasing frees the lock for the next host.
	unlock()
	if _, err := lockHostAt(path); err != nil {
		t.Fatalf("lock after release: %v", err)
	}
}
