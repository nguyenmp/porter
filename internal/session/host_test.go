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

func TestUnregisterHostFailsPending(t *testing.T) {
	st := newTestStore(t)
	ch := make(chan api.HostRequest, 8)
	st.RegisterHost(ch, "mac", "macbook", "host")

	// The host drops while a provision is in flight: the goroutine consumes
	// the request the main thread's Provision sends, then disconnects the
	// host, which must fail the pending provision promptly.
	unregistered := make(chan struct{})
	go func() {
		<-ch
		st.UnregisterHost("mac")
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
