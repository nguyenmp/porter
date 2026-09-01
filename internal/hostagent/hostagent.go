// Package hostagent is the persistent execution host: a long-running process
// on a machine (e.g. a laptop) that connects to a porter server once and
// provisions execution contexts for any session on demand. Each provisioned
// context is an isolated environment — a working directory, or a sandbox
// container holding one git worktree per requested repo — that the host
// serves as that session's execution provider, so many chats can run on the
// same machine and repos without sharing state. It is the roadmap's Execution
// Host: a host creates an execution environment and returns an Execution
// Provider.
package hostagent

import (
	"context"
	"fmt"
	"io"
	"log"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"porter/internal/api"
	"porter/internal/client"
	"porter/internal/config"
	"porter/internal/exec"
	"porter/internal/mcp"
	"porter/internal/tools"
)

// Run starts the persistent execution host agent and blocks until ctx is
// cancelled. The agent identifies itself by hostname (override with
// PORTER_HOST_ID), reports its working directory (the process cwd) and the
// git repositories it discovered under the user's home (for the web UI's
// sandbox picker), and reconnects to the server forever, provisioning a
// provider for every session the server asks for. A provision that names a
// repo is served from a fresh git worktree sandbox under ~/.porter/sandboxes
// (not configurable); stale sandboxes left by a previous run are cleaned up
// on startup.
func Run(ctx context.Context, cfg config.ClientConfig) error {
	c := client.New(cfg.ServerURL, client.BasicAuth{Username: cfg.Username, Password: cfg.Password})

	hostID := os.Getenv("PORTER_HOST_ID")
	if hostID == "" {
		if host, err := os.Hostname(); err == nil && host != "" {
			hostID = host
		} else {
			hostID = "porter-host"
		}
	}
	// The host's default working directory is the directory the host process
	// runs from (there is no PORTER_HOST_CWD); a chat can still request a
	// different one when it's created.
	cwd, err := os.Getwd()
	if err != nil {
		return fmt.Errorf("get host working directory: %w", err)
	}
	root := worktreeRoot()
	cleanupStaleWorktrees(root)

	// Report the host's base environment so the web UI's "new chat on" picker
	// can show where this host runs, what skills it has, and the repos it can
	// sandbox chats in (discovered under the home directory). The context is
	// posted on every (re)connect, like the REPL's exec context.
	env, err := exec.Discover(cwd)
	if err != nil {
		return fmt.Errorf("discover host environment: %w", err)
	}
	env.ID = hostID
	env.Name = hostID
	env.Repos = discoverRepos("")

	// Load the host's own MCP servers — e.g. a laptop-only server behind a
	// corporate VPN — from ~/.porter/porter.mcp.json. The host serves these
	// servers' tool calls itself (CallMCP routed down this connection); the
	// porter server lists them via the host's reported environment. A
	// malformed config is logged and skipped: MCP is optional, and a broken
	// config must not take down the execution host.
	hub := mcp.New(nil)
	if home, err := os.UserHomeDir(); err == nil {
		hub, err = mcp.Load(filepath.Join(home, ".porter", "porter.mcp.json"), nil)
		if err != nil {
			log.Printf("execution host: mcp config: %v (continuing without host MCP servers)", err)
			hub = mcp.New(nil)
		}
	}
	if n := len(hub.Names()); n > 0 {
		log.Printf("execution host: serving %d MCP server(s): %s", n, strings.Join(hub.Names(), ", "))
	}

	env.MCPServers = hubSummary(hub, hostID)
	log.Printf("execution host %s: %s @ %s (%d skills, sandboxes in %s, %d repos, %d mcp servers)",
		hostID, env.System, env.CWD, len(env.Skills), root, len(env.Repos), len(env.MCPServers))

	prov := &provisioner{c: c, hostID: hostID, cwd: cwd, worktrees: root, serves: map[string]*serveState{}, mcp: hub}
	for {
		if ctx.Err() != nil {
			return nil
		}
		log.Printf("execution host %s: connecting to %s", hostID, cfg.ServerURL)
		if err := c.PostHostContext(ctx, hostID, env); err != nil && ctx.Err() == nil {
			log.Printf("execution host: context register failed: %v", err)
		}
		if err := c.ServeHost(ctx, hostID, prov.provision, func() {
			log.Printf("execution host %s: connected", hostID)
		}); err != nil {
			if ctx.Err() != nil {
				return nil
			}
			log.Printf("execution host: connection dropped: %v", err)
		}
		select {
		case <-ctx.Done():
			return nil
		case <-time.After(time.Second):
		}
	}
}

// provisioner owns the host's per-session sandboxes and serves their tool
// calls. serves tracks active worktree sandboxes by provider id so a release
// request can stop them and remove the worktree; plain working-directory
// provisions (no repo) are not tracked — their serve loop lives for the
// process, matching the phase-1 host behavior.
type provisioner struct {
	c         *client.Client
	hostID    string
	cwd       string // default working directory for provisions without a CWD
	worktrees string // root directory for worktree sandboxes
	mu        sync.Mutex
	serves    map[string]*serveState
	mcp       *mcp.Hub // the host's own MCP servers, served on this machine
}

// serveState is one live worktree sandbox: the context that bounds its serve
// loop (cancelled on release), the container directory the worktrees live in
// (removed on release), and each worktree's repo/path/branch to tear down.
type serveState struct {
	cancel    context.CancelFunc
	dir       string
	worktrees []worktreeRef
}

// provision handles one request from the server. Kind "release" tears down a
// worktree sandbox the host created earlier. Kind "provision" creates the
// execution environment a session asked for and starts serving it as that
// session's execution provider: a working directory (req.CWD or the host
// default) when no repos are named, or a multi-repo sandbox when they are — a
// container directory holding one git worktree per repo (each a fresh branch
// porter/<providerID>, -2, -3... for later worktrees of the same repo, based
// at the repo's requested branch or HEAD), so many chats can work on the same
// repos without trampling each other and one chat can work across several at
// once. It returns an error only for internal failures — a bad request is
// reported to the server (PostHostProviderError) and the host connection
// keeps serving.
func (p *provisioner) provision(req api.HostRequest) error {
	if req.Kind == "release" {
		p.release(req)
		return nil
	}
	dir := req.CWD
	if dir == "" {
		dir = p.cwd
	}
	// A short timeout keeps one bad provision from stalling the host
	// connection's read loop (PostExecContext is quick, but never trust the
	// network; worktree creation on a large repo can take a moment).
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	var serveCtx context.Context = context.Background()
	var extraRoots []string
	sandboxed := len(req.Repos) > 0
	if sandboxed {
		// A multi-repo sandbox is a container directory holding one git
		// worktree per requested repo, so the session's working directory
		// shows every repo as a sibling and the model can work across them.
		// Resolving and branch/directory naming live in provisionWorktrees;
		// on any failure the worktrees created so far are rolled back so a
		// bad repo never leaves a half-open sandbox behind.
		sandboxDir := filepath.Join(p.worktrees, req.ProviderID)
		wts, err := provisionWorktrees(ctx, req.ProviderID, req.Repos, sandboxDir)
		if err != nil {
			for _, w := range wts {
				if rerr := removeWorktree(w.repo, w.path, w.branch); rerr != nil {
					log.Printf("execution host: rollback %s: %v", w.path, rerr)
				}
			}
			_ = os.RemoveAll(sandboxDir)
			// The provision ctx may have expired (git worktree add on a slow
			// repo can eat the whole budget), so report the failure on a
			// fresh context: the server's provision wait resolves with the
			// real error instead of its own timeout.
			errCtx, errCancel := context.WithTimeout(context.Background(), 5*time.Second)
			_ = p.c.PostHostProviderError(errCtx, p.hostID, req.ProviderID, err.Error())
			errCancel()
			return nil
		}
		sctx, scancel := context.WithCancel(context.Background())
		p.mu.Lock()
		p.serves[req.ProviderID] = &serveState{cancel: scancel, dir: sandboxDir, worktrees: wts}
		p.mu.Unlock()
		dir = sandboxDir
		serveCtx = sctx
		for _, w := range wts {
			extraRoots = append(extraRoots, w.path)
		}
	}

	// DiscoverRoots lists files from the container and every worktree (each
	// bounded by its own file budget) and finds skills across every worktree
	// too, so a repo's skills load wherever it sits in the sandbox. With no
	// extra roots it behaves exactly like Discover.
	env, err := exec.DiscoverRoots(dir, extraRoots)
	if err != nil {
		if sandboxed {
			p.release(api.HostRequest{ProviderID: req.ProviderID})
		}
		_ = p.c.PostHostProviderError(ctx, p.hostID, req.ProviderID, err.Error())
		return nil
	}
	env.ID = req.ProviderID
	env.Name = p.hostID
	env.MCPServers = hubSummary(p.mcp, p.hostID)

	// Register the sandbox's environment with the session, then open the
	// session's exec connection in a goroutine: registering makes the provider
	// active and resolves the server's provision wait. ServeExec retries on
	// drops, so a brief network blip reconnects the provider without
	// re-provisioning. Sandboxed providers serve under a cancellable context
	// so a release request can stop them and remove the worktrees; plain-dir
	// provisions serve for the process.
	if err := p.c.PostExecContext(ctx, req.SessionID, env); err != nil {
		if sandboxed {
			p.release(api.HostRequest{ProviderID: req.ProviderID})
		}
		_ = p.c.PostHostProviderError(ctx, p.hostID, req.ProviderID, err.Error())
		return nil
	}
	disp := tools.NewDispatcherWithSkills(env.Skills)
	go p.serve(serveCtx, req.SessionID, req.ProviderID, disp, dir)
	return nil
}

// release tears down a worktree sandbox the host created for a session: it
// stops the sandbox's serve loop, removes every worktree (and its branch),
// prunes each repo's worktree admin entries, and removes the container
// directory. It is idempotent — an unknown provider id (a duplicate release,
// or a non-sandboxed provision, whose serve loop lives for the process) is a
// no-op — so a stale release from a reconnecting server is harmless.
func (p *provisioner) release(req api.HostRequest) {
	p.mu.Lock()
	st, ok := p.serves[req.ProviderID]
	if ok {
		delete(p.serves, req.ProviderID)
	}
	p.mu.Unlock()
	if !ok {
		return
	}
	if st.cancel != nil {
		st.cancel()
	}
	pruned := map[string]bool{}
	for _, w := range st.worktrees {
		if err := removeWorktree(w.repo, w.path, w.branch); err != nil {
			log.Printf("execution host: release %s: %v", req.ProviderID, err)
		}
		pruned[w.repo] = true
	}
	if st.dir != "" {
		_ = os.RemoveAll(st.dir)
	}
	for repo := range pruned {
		pruneRepoWorktrees(repo)
	}
}

// serve holds a session's exec connection open, running every tool call in
// the sandbox directory and streaming the output back. It lives for the
// process unless ctx is cancelled (a release request for a worktree sandbox);
// on a dropped connection it retries, so the provider re-registers on
// reconnect (e.g. after a server restart) without re-provisioning.
func (p *provisioner) serve(ctx context.Context, sessionID, providerID string, disp *tools.Dispatcher, dir string) {
	for {
		_ = p.c.ServeExec(ctx, sessionID, func(ctx context.Context, name string, args []byte) (io.ReadCloser, error) {
			// MCP servers hosted on this machine (e.g. behind a VPN) are
			// served here: the porter server routes CallMCP for a host-owned
			// server down this exec channel, and the local hub runs the call
			// against the server. Everything else goes to the dispatcher.
			if name == mcp.CallTool {
				return p.mcp.Run(ctx, mcp.CallTool, args)
			}
			return disp.RunDir(ctx, name, args, dir)
		}, client.ExecConn{ID: providerID, Name: p.hostID, Kind: "host"})
		if ctx.Err() != nil {
			return
		}
		time.Sleep(time.Second)
	}
}

// hubSummary renders a hub's servers as reported metadata, tagged with the
// host id so the server-side hub can show where each server lives.
func hubSummary(h *mcp.Hub, hostID string) []api.MCPServer {
	out := h.Summary()
	for i := range out {
		out[i].Host = hostID
	}
	return out
}
