package session

import (
	"context"
	"strings"
	"testing"
	"time"

	"porter/internal/api"
)

// newTestStore returns a Store with an in-memory persister, no sessions.
func newTestStore(t *testing.T) *Store {
	t.Helper()
	return NewStore(memPersister(t), nil)
}

func TestRegisterHostAndHosts(t *testing.T) {
	st := newTestStore(t)
	st.RegisterHost(make(chan api.HostRequest, 8), "mac", "macbook", "host")
	st.RegisterHost(make(chan api.HostRequest, 8), "vps", "vps", "")

	hosts := st.Hosts()
	if len(hosts) != 2 {
		t.Fatalf("Hosts len = %d, want 2", len(hosts))
	}
	// Hosts are sorted by id.
	if hosts[0].ID != "mac" || hosts[1].ID != "vps" {
		t.Errorf("host ids = %q, %q; want mac, vps", hosts[0].ID, hosts[1].ID)
	}
	if !hosts[0].Connected || hosts[0].Kind != "host" || hosts[0].Name != "macbook" {
		t.Errorf("host[0] = %+v, want connected host 'macbook'", hosts[0])
	}
	if hosts[1].Kind != "host" { // empty kind defaults to "host"
		t.Errorf("host[1].Kind = %q, want host", hosts[1].Kind)
	}
}

func TestSetHostContextBeforeRegister(t *testing.T) {
	st := newTestStore(t)
	// The agent posts its context before opening the host connection; it must
	// attach when the host registers.
	st.SetHostContext(api.ExecContext{ID: "mac", Name: "macbook", System: "test", CWD: "/tmp"})
	st.RegisterHost(make(chan api.HostRequest, 8), "mac", "", "host")

	hosts := st.Hosts()
	if len(hosts) != 1 || hosts[0].Context == nil || hosts[0].Context.CWD != "/tmp" {
		t.Fatalf("host context not attached: %+v", hosts)
	}
}

func TestProvisionHappyPath(t *testing.T) {
	st := newTestStore(t)
	ch := make(chan api.HostRequest, 8)
	st.RegisterHost(ch, "mac", "macbook", "host")

	got := make(chan api.HostRequest, 1)
	go func() {
		req := <-ch
		got <- req
		st.ProvisionRegistered("session_1", req.ProviderID)
	}()

	if err := st.Provision(context.Background(), "session_1", "mac", api.HostRequest{CWD: "/tmp"}); err != nil {
		t.Fatalf("Provision: %v", err)
	}
	req := <-got
	if req.Kind != "provision" {
		t.Errorf("request kind = %q, want provision", req.Kind)
	}
	if req.SessionID != "session_1" {
		t.Errorf("request session = %q, want session_1", req.SessionID)
	}
	if req.CWD != "/tmp" {
		t.Errorf("request cwd = %q, want /tmp", req.CWD)
	}
	if !strings.HasPrefix(req.ProviderID, "mac-provider-") {
		t.Errorf("provider id = %q, want mac-provider-N", req.ProviderID)
	}
}

func TestProvisionHostNotConnected(t *testing.T) {
	st := newTestStore(t)
	err := st.Provision(context.Background(), "session_1", "mac", api.HostRequest{})
	if err == nil || !strings.Contains(err.Error(), "not connected") {
		t.Fatalf("Provision on unknown host = %v, want 'not connected'", err)
	}
}

func TestProvisionTimeout(t *testing.T) {
	st := newTestStore(t)
	st.RegisterHost(make(chan api.HostRequest, 8), "mac", "macbook", "host")
	old := provisionTimeout
	provisionTimeout = 50 * time.Millisecond
	defer func() { provisionTimeout = old }()

	start := time.Now()
	err := st.Provision(context.Background(), "session_1", "mac", api.HostRequest{})
	if err == nil || !strings.Contains(err.Error(), "did not provision") {
		t.Fatalf("Provision timeout = %v, want timeout error", err)
	}
	if time.Since(start) > time.Second {
		t.Fatalf("timeout took %s, want ~50ms", time.Since(start))
	}
}

func TestHostProviderError(t *testing.T) {
	st := newTestStore(t)
	ch := make(chan api.HostRequest, 8)
	st.RegisterHost(ch, "mac", "macbook", "host")

	done := make(chan error, 1)
	go func() {
		req := <-ch
		done <- st.HostProviderError("mac", req.ProviderID, "no such directory")
	}()

	err := st.Provision(context.Background(), "session_1", "mac", api.HostRequest{})
	if err == nil || !strings.Contains(err.Error(), "no such directory") {
		t.Fatalf("Provision after host error = %v, want host's error", err)
	}
	if err := <-done; err != nil {
		t.Errorf("HostProviderError = %v, want nil", err)
	}
}

func TestReleaseSessionSendsRelease(t *testing.T) {
	st := newTestStore(t)
	ch := make(chan api.HostRequest, 8)
	st.RegisterHost(ch, "mac", "macbook", "host")

	// A sandboxed provision (repo named) is recorded when the provider
	// registers, so archiving the session can release the worktree.
	go func() {
		req := <-ch
		st.ProvisionRegistered("session_1", req.ProviderID)
	}()
	if err := st.Provision(context.Background(), "session_1", "mac", api.HostRequest{Repos: []api.RepoRef{{Path: "/tmp/repo"}}}); err != nil {
		t.Fatalf("Provision: %v", err)
	}

	got := make(chan api.HostRequest, 1)
	go func() { got <- <-ch }()
	st.ReleaseSession("session_1")
	req := <-got
	if req.Kind != "release" || req.SessionID != "session_1" || !strings.HasPrefix(req.ProviderID, "mac-provider-") {
		t.Fatalf("release request = %+v, want kind=release for session_1", req)
	}

	// Releasing again is a no-op: nothing is sent.
	select {
	case r := <-ch:
		t.Fatalf("unexpected request after second release: %+v", r)
	case <-time.After(20 * time.Millisecond):
	}
}

func TestReleaseSessionNoSandbox(t *testing.T) {
	st := newTestStore(t)
	ch := make(chan api.HostRequest, 8)
	st.RegisterHost(ch, "mac", "macbook", "host")

	// A plain-dir provision (no repo) has no sandbox to release: archiving
	// the session must not send anything to the host.
	go func() {
		req := <-ch
		st.ProvisionRegistered("session_1", req.ProviderID)
	}()
	if err := st.Provision(context.Background(), "session_1", "mac", api.HostRequest{}); err != nil {
		t.Fatalf("Provision: %v", err)
	}

	st.ReleaseSession("session_1")
	select {
	case r := <-ch:
		t.Fatalf("unexpected release for non-sandboxed session: %+v", r)
	case <-time.After(20 * time.Millisecond):
	}
}

func TestReleaseSessionHostGone(t *testing.T) {
	st := newTestStore(t)
	ch := make(chan api.HostRequest, 8)
	conn := st.RegisterHost(ch, "mac", "macbook", "host")
	go func() {
		req := <-ch
		st.ProvisionRegistered("session_1", req.ProviderID)
	}()
	if err := st.Provision(context.Background(), "session_1", "mac", api.HostRequest{Repos: []api.RepoRef{{Path: "/tmp/repo"}}}); err != nil {
		t.Fatalf("Provision: %v", err)
	}

	// The host disconnects (which also drops its sandbox records); releasing
	// afterwards must not block, panic, or send.
	st.UnregisterHost(conn)
	st.ReleaseSession("session_1")
}

func TestUnregisterHostFailsPending(t *testing.T) {
	st := newTestStore(t)
	ch := make(chan api.HostRequest, 8)
	conn := st.RegisterHost(ch, "mac", "macbook", "host")

	// The host drops while a provision is in flight: the goroutine consumes
	// the request the main thread's Provision sends, then disconnects the
	// host, which must fail the pending provision promptly.
	unregistered := make(chan struct{})
	go func() {
		<-ch
		st.UnregisterHost(conn)
		close(unregistered)
	}()

	err := st.Provision(context.Background(), "session_1", "mac", api.HostRequest{})
	if err == nil || !strings.Contains(err.Error(), "disconnected") {
		t.Fatalf("Provision while host drops = %v, want disconnected error", err)
	}
	<-unregistered

	// The host is gone; a second provision fails immediately.
	if err := st.Provision(context.Background(), "session_1", "mac", api.HostRequest{}); err == nil {
		t.Fatal("Provision after host removed should fail")
	}
}

// TestUnregisterHostSupersededConnection proves UnregisterHost only removes
// the registration owned by the given connection token: a late disconnect
// from a connection that was replaced by a newer registration for the same
// host id must not unregister the newer host.
func TestUnregisterHostSupersededConnection(t *testing.T) {
	st := newTestStore(t)
	ch := make(chan api.HostRequest, 8)
	old := st.RegisterHost(ch, "mac", "macbook", "host")
	// A reconnect replaces the old registration for the same host id.
	newConn := st.RegisterHost(ch, "mac", "macbook", "host")

	// The superseded connection's disconnect must be a no-op.
	st.UnregisterHost(old)
	hosts := st.Hosts()
	if len(hosts) != 1 || hosts[0].ID != "mac" || !hosts[0].Connected {
		t.Fatalf("hosts after superseded disconnect = %+v, want mac still connected", hosts)
	}
	// The owning connection's disconnect removes it.
	st.UnregisterHost(newConn)
	if hosts := st.Hosts(); len(hosts) != 0 {
		t.Fatalf("hosts after owner disconnect = %+v, want none", hosts)
	}
}
