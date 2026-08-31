// Package hostagent is the persistent execution host: a long-running process
// on a machine (e.g. a laptop) that connects to a porter server once and
// provisions execution contexts for any session on demand. Each provisioned
// context is an isolated environment — today a working directory, later a git
// worktree sandbox — that the host serves as that session's execution
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
	"time"

	"porter/internal/api"
	"porter/internal/client"
	"porter/internal/config"
	"porter/internal/exec"
	"porter/internal/tools"
)

// Run starts the persistent execution host agent and blocks until ctx is
// cancelled. The agent identifies itself by hostname (override with
// PORTER_HOST_ID), reports its default working directory (PORTER_HOST_CWD,
// else the process cwd), and reconnects to the server forever, provisioning a
// provider for every session the server asks for.
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
	cwd := os.Getenv("PORTER_HOST_CWD")
	if cwd == "" {
		if wd, err := os.Getwd(); err == nil {
			cwd = wd
		}
	}

	// Report the host's base environment so the web UI's "new chat on" picker
	// can show where this host runs and what skills it has. The context is
	// posted on every (re)connect, like the REPL's exec context.
	env, err := exec.Discover(cwd)
	if err != nil {
		return fmt.Errorf("discover host environment: %w", err)
	}
	env.ID = hostID
	env.Name = hostID

	log.Printf("execution host %s: %s @ %s (%d skills)", hostID, env.System, env.CWD, len(env.Skills))

	prov := &provisioner{c: c, hostID: hostID, cwd: cwd}
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
// calls.
type provisioner struct {
	c      *client.Client
	hostID string
	cwd    string // default working directory for provisions without a CWD
}

// provision creates the execution environment a session asked for and starts
// serving it as that session's execution provider. Phase 1: the environment
// is a working directory (req.CWD or the host default); the phase-2 sandbox
// (a git worktree per session on a shared repo) slots in here. It returns an
// error only for internal failures — a bad request is reported to the server
// (PostHostProviderError) and the host connection keeps serving.
func (p *provisioner) provision(req api.HostRequest) error {
	dir := req.CWD
	if dir == "" {
		dir = p.cwd
	}
	// A short timeout keeps one bad provision from stalling the host
	// connection's read loop (PostExecContext is quick, but never trust the
	// network).
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	env, err := exec.Discover(dir)
	if err != nil {
		_ = p.c.PostHostProviderError(ctx, p.hostID, req.ProviderID, err.Error())
		return nil
	}
	env.ID = req.ProviderID
	env.Name = p.hostID

	// Register the sandbox's environment with the session, then open the
	// session's exec connection in a goroutine: registering makes the provider
	// active and resolves the server's provision wait. ServeExec retries on
	// drops, so a brief network blip reconnects the provider without
	// re-provisioning.
	if err := p.c.PostExecContext(ctx, req.SessionID, env); err != nil {
		_ = p.c.PostHostProviderError(ctx, p.hostID, req.ProviderID, err.Error())
		return nil
	}
	disp := tools.NewDispatcherWithSkills(env.Skills)
	go p.serve(req.SessionID, req.ProviderID, disp, dir)
	return nil
}

// serve holds a session's exec connection open, running every tool call in
// the sandbox directory and streaming the output back. It lives for the
// process: on a dropped connection it retries, so the provider re-registers
// on reconnect (e.g. after a server restart) without re-provisioning.
func (p *provisioner) serve(sessionID, providerID string, disp *tools.Dispatcher, dir string) {
	ctx := context.Background()
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
