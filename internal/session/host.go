package session

import (
	"context"
	"errors"
	"fmt"
	"sort"
	"time"

	"porter/internal/api"
)

// provisionTimeout bounds how long creating a session waits for a host to
// provision its execution provider. A connected host responds in well under a
// second for a working-directory sandbox, and git worktree creation on a
// large repo can take tens of seconds; the bound covers both while still
// surfacing a dead host promptly (a timeout is non-fatal: the session is
// created with a warning and the provider attaches whenever it arrives). It
// is a var so tests can shorten it.
var provisionTimeout = 60 * time.Second

// host is one persistent execution agent (e.g. on a laptop) that can
// provision execution contexts for sessions. It is machine-level — a single
// connection serves any number of sessions — and is the roadmap's Execution
// Host: provisioning creates an execution environment (a sandbox on the host)
// and returns an Execution Provider for one session. ch is the host's exec
// subscription, the channel provision requests are sent down; it stays live
// while the host is connected. ctx is the base environment the host reported
// (system, default working directory, files, skills), shown in the web UI's
// "new chat on" picker.
type host struct {
	id        string
	name      string
	kind      string // "host"
	ch        chan api.HostRequest
	ctx       *api.ExecContext
	connected bool
}

// pendingProvision tracks one session's request for a provider on a host,
// from the moment the server pushes the provision request until the host's
// provider registers (RegisterExec resolves it) or the host reports failure.
// done closes when the provider registers; err is set when provisioning
// failed. The map lives on the Store (keyed by provider id) because both the
// host channel (Store-owned) and the session's RegisterExec (server-called)
// resolve it.
type pendingProvision struct {
	providerID string
	sessionID  string
	hostID     string
	done       chan struct{}
	err        error
	// sandbox reports whether the provision requested a repo sandbox (a git
	// worktree on the host), which is recorded in st.sandboxes when the
	// provider registers so the server can release it when the session ends.
	sandbox bool
}

// sandbox is one session's provisioned worktree on a host, recorded once the
// provider registers so a later ReleaseSession can tell that host to tear the
// worktree down.
type sandbox struct {
	hostID     string
	providerID string
}

// RegisterHost adds a connected execution host to the registry. A connecting
// host is upserted by id (re-registering after a reconnect updates it in
// place); an id-less host (legacy or tests) gets a generated one, returned to
// the caller so its deferred UnregisterHost can name the same host. A base
// context posted before the connection (pendingHostCtx) attaches here.
func (st *Store) RegisterHost(ch chan api.HostRequest, id, name, kind string) string {
	st.mu.Lock()
	defer st.mu.Unlock()
	if id == "" {
		st.hostSeq++
		id = fmt.Sprintf("host-%d", st.hostSeq)
	}
	h, ok := st.hosts[id]
	if !ok {
		h = &host{id: id}
		st.hosts[id] = h
	}
	h.name = name
	h.kind = kind
	if h.kind == "" {
		h.kind = "host"
	}
	h.ch = ch
	h.connected = true
	if ctx, ok := st.pendingHostCtx[id]; ok {
		h.ctx = ctx
		delete(st.pendingHostCtx, id)
	}
	return id
}

// UnregisterHost removes a connected execution host from the registry and
// fails its in-flight provisions, so a session waiting on a host that dropped
// is told promptly instead of hanging until the timeout. The host's
// per-session providers are not touched: they hold their own exec
// connections and reconnect independently.
func (st *Store) UnregisterHost(id string) {
	st.mu.Lock()
	_, ok := st.hosts[id]
	if !ok {
		st.mu.Unlock()
		return
	}
	delete(st.hosts, id)
	for pid, p := range st.pending {
		if p.hostID == id {
			delete(st.pending, pid)
			p.err = errors.New("execution host disconnected")
			close(p.done)
		}
	}
	for sid, sb := range st.sandboxes {
		if sb.hostID == id {
			delete(st.sandboxes, sid)
		}
	}
	st.mu.Unlock()
}

// SetHostContext stores the base environment context a connected execution
// host reported (system, default working directory, files, skills). Like a
// client's exec context, it is keyed by the host's id and attaches when the
// host registers (or updates a registered host in place); it is what the "new
// chat on" picker shows.
func (st *Store) SetHostContext(ctx api.ExecContext) {
	st.mu.Lock()
	defer st.mu.Unlock()
	id := ctx.ID
	if h, ok := st.hosts[id]; ok {
		h.ctx = &ctx
		return
	}
	// The context can arrive before the host connection registers (the agent
	// posts it before opening the connection, like the REPL); hold it for the
	// next RegisterHost with that id.
	if st.pendingHostCtx == nil {
		st.pendingHostCtx = map[string]*api.ExecContext{}
	}
	st.pendingHostCtx[id] = &ctx
}

// Hosts returns a summary of every registered execution host, for the web
// UI's "new chat on" picker. Disconnected hosts are included with
// Connected=false so the UI can show them greyed out.
func (st *Store) Hosts() []api.HostSummary {
	st.mu.Lock()
	defer st.mu.Unlock()
	out := make([]api.HostSummary, 0, len(st.hosts))
	for _, h := range st.hosts {
		out = append(out, api.HostSummary{
			ID:        h.id,
			Name:      h.name,
			Kind:      h.kind,
			Connected: h.connected,
			Context:   h.ctx,
		})
	}
	sort.Slice(out, func(i, j int) bool { return out[i].ID < out[j].ID })
	return out
}

// Provision asks the connected execution host with the given id to create an
// execution environment (a provider) for the session, and waits — bounded by
// provisionTimeout or ctx — for the host's provider to register. The provider
// auto-activates on registration (a connecting provider takes over, matching a
// REPL client), so the first message in a session created "on" a host runs
// there. It returns an error when the host is unknown or disconnected, when
// the host reports a provisioning failure, or on timeout; in every failure
// case the session keeps its local fallback and the provider attaches
// whenever (if ever) it arrives.
func (st *Store) Provision(ctx context.Context, sessionID, hostID string, req api.HostRequest) error {
	st.mu.Lock()
	h, ok := st.hosts[hostID]
	if !ok || !h.connected || h.ch == nil {
		st.mu.Unlock()
		return fmt.Errorf("execution host %q is not connected", hostID)
	}
	st.hostSeq++
	req.Kind = "provision"
	req.ProviderID = fmt.Sprintf("%s-provider-%d", hostID, st.hostSeq)
	req.SessionID = sessionID
	p := &pendingProvision{providerID: req.ProviderID, sessionID: sessionID, hostID: hostID, done: make(chan struct{}), sandbox: req.Repo != ""}
	st.pending[req.ProviderID] = p
	ch := h.ch
	st.mu.Unlock()

	select {
	case ch <- req:
	case <-ctx.Done():
		st.dropProvision(req.ProviderID)
		return ctx.Err()
	}

	select {
	case <-p.done:
		if p.err != nil {
			return p.err
		}
		return nil
	case <-ctx.Done():
		st.dropProvision(req.ProviderID)
		return ctx.Err()
	case <-time.After(provisionTimeout):
		st.dropProvision(req.ProviderID)
		return fmt.Errorf("execution host %q did not provision a provider within %s", hostID, provisionTimeout)
	}
}

// dropProvision removes a pending provision without resolving it (ctx
// cancelled or timed out). The host may still register the provider later; a
// RegisterExec for an id with no pending provision is a plain registration.
func (st *Store) dropProvision(providerID string) {
	st.mu.Lock()
	delete(st.pending, providerID)
	st.mu.Unlock()
}

// ProvisionRegistered resolves a pending provision whose provider just
// registered on its session (the host opened the exec connection). It is
// called by the server after RegisterExec, so the session-create wait returns
// as soon as the provider is live and active.
func (st *Store) ProvisionRegistered(sessionID, providerID string) {
	st.mu.Lock()
	p, ok := st.pending[providerID]
	if ok && p.sessionID == sessionID {
		delete(st.pending, providerID)
		close(p.done)
		// A sandboxed provision is recorded so archiving the session can
		// release the worktree. Plain-dir provisions are not: their serve
		// loop lives for the host process and nothing needs releasing.
		if p.sandbox {
			st.sandboxes[sessionID] = &sandbox{hostID: p.hostID, providerID: providerID}
		}
	}
	st.mu.Unlock()
}

// ReleaseSession tells the execution host that provisioned a session's
// worktree sandbox to tear it down (the server sends this when the session is
// archived). It is best-effort and non-blocking: the session is already gone
// from the user's flow, so a missed release (host disconnected, channel full)
// only leaks the sandbox until the host restarts (startup cleanup) — never
// blocks or fails the archive. Idempotent: a session with no sandbox (plain
// dir, or already released) is a no-op.
func (st *Store) ReleaseSession(sessionID string) {
	st.mu.Lock()
	sb, ok := st.sandboxes[sessionID]
	if !ok {
		st.mu.Unlock()
		return
	}
	delete(st.sandboxes, sessionID)
	h, ok := st.hosts[sb.hostID]
	if !ok || !h.connected || h.ch == nil {
		st.mu.Unlock()
		return
	}
	ch := h.ch
	st.mu.Unlock()

	select {
	case ch <- api.HostRequest{Kind: "release", ProviderID: sb.providerID, SessionID: sessionID}:
	default: // host channel full: the host will reconnect; startup cleanup covers the leak
	}
}

// HostProviderError resolves a pending provision as failed: the host could
// not create the requested environment. The session keeps its local fallback
// and the waiting create request gets the host's error.
func (st *Store) HostProviderError(hostID, providerID, msg string) error {
	st.mu.Lock()
	defer st.mu.Unlock()
	p, ok := st.pending[providerID]
	if !ok {
		return fmt.Errorf("unknown provision %q", providerID)
	}
	delete(st.pending, providerID)
	p.err = fmt.Errorf("provision failed: %s", msg)
	close(p.done)
	return nil
}
