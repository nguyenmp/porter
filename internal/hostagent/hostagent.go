// Package hostagent is the persistent execution host: a long-running process
// on a machine (e.g. a laptop) that connects to a porter server once and
// provisions execution contexts for any session on demand. Each provisioned
// context is an isolated environment — a working directory, or a git worktree
// sandbox on a shared repo — that the host serves as that session's execution
// provider, so many chats can run on the same machine (and, later, the same
// repo) without sharing state. It is the roadmap's Execution Host: a host
// creates an execution environment and returns an Execution Provider.
package hostagent

import (
	"context"
	"fmt"
	"io"
	"log"
	"os"
	"sync"
	"time"

	"porter/internal/api"
	"porter/internal/client"
	"porter/internal/config"
	"porter/internal/exec"
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

	log.Printf("execution host %s: %s @ %s (%d skills, sandboxes in %s, %d repos)",
		hostID, env.System, env.CWD, len(env.Skills), root, len(env.Repos))

	prov := &provisioner{c: c, hostID: hostID, cwd: cwd, worktrees: root, serves: map[string]*serveState{}}
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
}

// serveState is one live worktree sandbox: the context that bounds its serve
// loop (cancelled on release) and the repo/path/branch to tear down.
type serveState struct {
	cancel context.CancelFunc
	repo   string
	path   string
	branch string
}

// provision handles one request from the server. Kind "release" tears down a
// worktree sandbox the host created earlier. Kind "provision" creates the
// execution environment a session asked for and starts serving it as that
// session's execution provider: a working directory (req.CWD or the host
// default) when no repo is named, or a git worktree sandbox on req.Repo — a
// fresh branch porter/<providerID> based at req.Branch (or the repo's HEAD)
// when one is — so many chats can work on the same repo without trampling
// each other. It returns an error only for internal failures — a bad request
// is reported to the server (PostHostProviderError) and the host connection
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
	if req.Repo != "" {
		// Resolve the repo against the user's home directory (the same root
		// discoverRepos scans), so a bare name like "porter" means ~/porter
		// regardless of where the host process runs. The resolved path is also
		// what release/cleanup tear down, so they agree on the repo.
		repo := resolveRepo(req.Repo)
		path, branch, err := provisionWorktree(ctx, repo, req.Branch, p.worktrees, req.ProviderID)
		if err != nil {
			_ = p.c.PostHostProviderError(ctx, p.hostID, req.ProviderID, err.Error())
			return nil
		}
		sctx, scancel := context.WithCancel(context.Background())
		p.mu.Lock()
		p.serves[req.ProviderID] = &serveState{cancel: scancel, repo: repo, path: path, branch: branch}
		p.mu.Unlock()
		dir = path
		serveCtx = sctx
	}

	env, err := exec.Discover(dir)
	if err != nil {
		if req.Repo != "" {
			p.release(api.HostRequest{ProviderID: req.ProviderID})
		}
		_ = p.c.PostHostProviderError(ctx, p.hostID, req.ProviderID, err.Error())
		return nil
	}
	env.ID = req.ProviderID
	env.Name = p.hostID

	// Register the sandbox's environment with the session, then open the
	// session's exec connection in a goroutine: registering makes the provider
	// active and resolves the server's provision wait. ServeExec retries on
	// drops, so a brief network blip reconnects the provider without
	// re-provisioning. Sandboxed providers serve under a cancellable context
	// so a release request can stop them and remove the worktree; plain-dir
	// provisions serve for the process.
	if err := p.c.PostExecContext(ctx, req.SessionID, env); err != nil {
		if req.Repo != "" {
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
// stops the sandbox's serve loop and removes the worktree and its branch. It
// is idempotent — an unknown provider id (a duplicate release, or a
// non-sandboxed provision, whose serve loop lives for the process) is a
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
	st.cancel()
	if err := removeWorktree(st.repo, st.path, st.branch); err != nil {
		log.Printf("execution host: release %s: %v", req.ProviderID, err)
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
			return disp.RunDir(ctx, name, args, dir)
		}, client.ExecConn{ID: providerID, Name: p.hostID, Kind: "host"})
		if ctx.Err() != nil {
			return
		}
		time.Sleep(time.Second)
	}
}
