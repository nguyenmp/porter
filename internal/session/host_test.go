package session

import (
	"context"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"porter/internal/api"
	"porter/internal/db"
)

// newTestStore returns a Store with an in-memory persister, no sessions.
func newTestStore(t *testing.T) *Store {
	t.Helper()
	return NewStore(memPersister(t), nil)
}

func TestRegisterHostAndHosts(t *testing.T) {
	st := newTestStore(t)
	st.RegisterHost(make(chan api.HostRequest, 8), "mac", "macbook", "host", "mac-inst")
	st.RegisterHost(make(chan api.HostRequest, 8), "vps", "vps", "", "vps-inst")

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
	// attach when the host registers. The context carries the agent's
	// instance so a different process's context can never attach to this
	// registration.
	_ = st.SetHostContext(api.ExecContext{ID: "mac", Name: "macbook", System: "test", CWD: "/tmp", Instance: "mac-inst"})
	st.RegisterHost(make(chan api.HostRequest, 8), "mac", "", "host", "mac-inst")

	hosts := st.Hosts()
	if len(hosts) != 1 || hosts[0].Context == nil || hosts[0].Context.CWD != "/tmp" {
		t.Fatalf("host context not attached: %+v", hosts)
	}
}

func TestProvisionHappyPath(t *testing.T) {
	st := newTestStore(t)
	ch := make(chan api.HostRequest, 8)
	st.RegisterHost(ch, "mac", "macbook", "host", "mac-inst")

	got := make(chan api.HostRequest, 1)
	go func() {
		req := <-ch
		got <- req
		st.ProvisionRegistered("session_1", req.ProviderID)
	}()

	if err := st.Provision(context.Background(), "session_1", "mac", api.HostRequest{}); err != nil {
		t.Fatalf("Provision: %v", err)
	}
	req := <-got
	if req.Kind != "provision" {
		t.Errorf("request kind = %q, want provision", req.Kind)
	}
	if req.SessionID != "session_1" {
		t.Errorf("request session = %q, want session_1", req.SessionID)
	}
	if len(req.Repos) != 0 {
		t.Errorf("request repos = %v, want none", req.Repos)
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
	st.RegisterHost(make(chan api.HostRequest, 8), "mac", "macbook", "host", "mac-inst")
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
	st.RegisterHost(ch, "mac", "macbook", "host", "mac-inst")

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
	st.RegisterHost(ch, "mac", "macbook", "host", "mac-inst")

	// Every provision is a sandbox on the host — even one with no repos — so
	// it is recorded when the provider registers and archiving the session
	// releases it.
	go func() {
		req := <-ch
		st.ProvisionRegistered("session_1", req.ProviderID)
	}()
	if err := st.Provision(context.Background(), "session_1", "mac", api.HostRequest{}); err != nil {
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

func TestReleaseSessionHostGone(t *testing.T) {
	st := newTestStore(t)
	ch := make(chan api.HostRequest, 8)
	conn, _ := st.RegisterHost(ch, "mac", "macbook", "host", "mac-inst")
	go func() {
		req := <-ch
		st.ProvisionRegistered("session_1", req.ProviderID)
	}()
	if err := st.Provision(context.Background(), "session_1", "mac", api.HostRequest{Repos: []api.RepoRef{{Path: "/tmp/repo"}}}); err != nil {
		t.Fatalf("Provision: %v", err)
	}

	// The host disconnects (its sandbox mappings are deliberately kept so the
	// chat can reconnect later); releasing afterwards must not block, panic, or
	// send.
	st.UnregisterHost(conn)
	st.ReleaseSession("session_1")
}

func TestUnregisterHostFailsPending(t *testing.T) {
	st := newTestStore(t)
	ch := make(chan api.HostRequest, 8)
	conn, _ := st.RegisterHost(ch, "mac", "macbook", "host", "mac-inst")

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
	old, _ := st.RegisterHost(ch, "mac", "macbook", "host", "mac-inst")
	// A reconnect replaces the old registration for the same host id.
	newConn, _ := st.RegisterHost(ch, "mac", "macbook", "host", "mac-inst")

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

// TestRegisterHostRejectsSecondProcess proves a second host agent claiming
// the same host id while the first is connected is rejected with an error
// naming the owner, while a reconnect from the same process (same instance)
// is allowed.
func TestRegisterHostRejectsSecondProcess(t *testing.T) {
	st := newTestStore(t)
	first, err := st.RegisterHost(make(chan api.HostRequest, 8), "mac", "macbook", "host", "inst-1")
	if err != nil {
		t.Fatalf("first RegisterHost: %v", err)
	}
	// A second process (different instance) is rejected.
	if _, err := st.RegisterHost(make(chan api.HostRequest, 8), "mac", "macbook", "host", "inst-2"); err == nil {
		t.Fatal("second process RegisterHost succeeded, want rejection")
	}
	// A reconnect from the same process (same instance) upserts in place.
	if _, err := st.RegisterHost(make(chan api.HostRequest, 8), "mac", "macbook", "host", "inst-1"); err != nil {
		t.Fatalf("same-instance reconnect rejected: %v", err)
	}
	st.UnregisterHost(first)
}

// TestSetHostContextRejectsSecondProcess proves a second process's context
// POST is rejected when the host id is owned by another instance, so a
// rejected agent cannot clobber the connected host's reported environment.
func TestSetHostContextRejectsSecondProcess(t *testing.T) {
	st := newTestStore(t)
	if _, err := st.RegisterHost(make(chan api.HostRequest, 8), "mac", "macbook", "host", "inst-1"); err != nil {
		t.Fatalf("RegisterHost: %v", err)
	}
	// The owner's context updates the host in place.
	if err := st.SetHostContext(api.ExecContext{ID: "mac", Instance: "inst-1", CWD: "/owner"}); err != nil {
		t.Fatalf("owner SetHostContext: %v", err)
	}
	// A different process's context is rejected.
	err := st.SetHostContext(api.ExecContext{ID: "mac", Instance: "inst-2", CWD: "/intruder"})
	if err == nil || !strings.Contains(err.Error(), "another execution host") {
		t.Fatalf("second process SetHostContext = %v, want 'another execution host' error", err)
	}
	// The owner's context is untouched.
	hosts := st.Hosts()
	if hosts[0].Context == nil || hosts[0].Context.CWD != "/owner" {
		t.Fatalf("owner context clobbered: %+v", hosts[0].Context)
	}
}

// TestDuplicateHostErrorNamesOwnerPID proves the rejection message includes
// the connected host agent's process id when it reported one, so the user
// knows which process to stop.
func TestDuplicateHostErrorNamesOwnerPID(t *testing.T) {
	st := newTestStore(t)
	if _, err := st.RegisterHost(make(chan api.HostRequest, 8), "mac", "macbook", "host", "inst-1"); err != nil {
		t.Fatalf("RegisterHost: %v", err)
	}
	if err := st.SetHostContext(api.ExecContext{ID: "mac", Instance: "inst-1", PID: 4242}); err != nil {
		t.Fatalf("SetHostContext: %v", err)
	}
	_, err := st.RegisterHost(make(chan api.HostRequest, 8), "mac", "macbook", "host", "inst-2")
	if err == nil || !strings.Contains(err.Error(), "PID 4242") {
		t.Fatalf("rejection = %v, want it to name PID 4242", err)
	}
}

// TestProvisionPersistsSandboxAndOfferAnswers covers the durable mapping and
// the offer/accept handshake end to end at the store level: a provision is
// recorded (in memory and persisted) under a unique provider id; the offer
// answers keep for a live chat, refuse for an archived chat (its release was
// missed because the host was down) and for unknown folders; releasing a chat
// removes its row, so a later offer refuses its folder.
func TestProvisionPersistsSandboxAndOfferAnswers(t *testing.T) {
	st := newTestStore(t)
	ch := make(chan api.HostRequest, 8)
	st.RegisterHost(ch, "mac", "macbook", "host", "mac-inst")

	live, err := st.Create(nil)
	if err != nil {
		t.Fatalf("Create live: %v", err)
	}
	archived, err := st.Create(nil)
	if err != nil {
		t.Fatalf("Create archived: %v", err)
	}
	// Provision both chats (the goroutines resolve the provisions the way the
	// host's provider registration does).
	provision := func(ses *Session) string {
		got := make(chan string, 1)
		go func() {
			req := <-ch
			st.ProvisionRegistered(ses.ID(), req.ProviderID)
			got <- req.ProviderID
		}()
		if err := st.Provision(context.Background(), ses.ID(), "mac", api.HostRequest{}); err != nil {
			t.Fatalf("Provision(%s): %v", ses.ID(), err)
		}
		return <-got
	}
	liveP := provision(live)
	archivedP := provision(archived)
	if liveP == archivedP {
		t.Fatalf("provider ids reused: %q", liveP)
	}
	for _, pid := range []string{liveP, archivedP} {
		if !strings.HasPrefix(pid, "mac-provider-") {
			t.Errorf("provider id %q, want mac-provider-N", pid)
		}
	}

	// Archiving while the host is "down" (no release delivered) leaves the
	// row in place.
	if err := st.Archive(archived.ID()); err != nil {
		t.Fatalf("Archive: %v", err)
	}

	// The offer answers every folder: keep the live chat's, refuse the
	// archived chat's and the unknown folder.
	verdicts, err := st.OfferSandboxes("mac", []string{liveP, archivedP, "mac-provider-999"})
	if err != nil {
		t.Fatalf("OfferSandboxes: %v", err)
	}
	if len(verdicts) != 3 {
		t.Fatalf("verdicts = %d, want 3", len(verdicts))
	}
	if !verdicts[0].Keep || verdicts[0].SessionID != live.ID() {
		t.Errorf("verdict[0] = %+v, want keep for %s", verdicts[0], live.ID())
	}
	if verdicts[1].Keep {
		t.Errorf("verdict[1] = %+v, want refuse (archived)", verdicts[1])
	}
	if verdicts[2].Keep {
		t.Errorf("verdict[2] = %+v, want refuse (unknown)", verdicts[2])
	}

	// Releasing removes the persisted row: the next offer refuses the folder.
	st.ReleaseSession(live.ID())
	verdicts, err = st.OfferSandboxes("mac", []string{liveP})
	if err != nil {
		t.Fatalf("OfferSandboxes after release: %v", err)
	}
	if verdicts[0].Keep {
		t.Errorf("verdict after release = %+v, want refuse", verdicts[0])
	}

	// An offer from an unregistered host is refused outright.
	if _, err := st.OfferSandboxes("vps", []string{liveP}); err == nil {
		t.Fatal("OfferSandboxes from unknown host succeeded, want error")
	}
}

// TestProviderSeqSurvivesStoreRestart proves provider ids are never reused
// across a server restart: the sequence lives in the database (not in a
// counter that resets), so a new provision after a restart cannot claim an id
// a sandbox folder on disk already holds.
func TestProviderSeqSurvivesStoreRestart(t *testing.T) {
	dbPath := filepath.Join(t.TempDir(), "porter.db")
	d, err := db.Open(dbPath)
	if err != nil {
		t.Fatalf("open db: %v", err)
	}
	st := NewStore(d, nil)
	ch := make(chan api.HostRequest, 8)
	st.RegisterHost(ch, "mac", "macbook", "host", "mac-inst")
	ses, err := st.Create(nil)
	if err != nil {
		t.Fatalf("Create: %v", err)
	}
	got := make(chan string, 1)
	go func() {
		req := <-ch
		st.ProvisionRegistered(ses.ID(), req.ProviderID)
		got <- req.ProviderID
	}()
	if err := st.Provision(context.Background(), ses.ID(), "mac", api.HostRequest{}); err != nil {
		t.Fatalf("Provision: %v", err)
	}
	first := <-got

	// "Restart" the server on the same database.
	st.Close()
	d2, err := db.Open(dbPath)
	if err != nil {
		t.Fatalf("reopen db: %v", err)
	}
	defer d2.Close()
	st2 := NewStore(d2, nil)
	st2.RegisterHost(ch, "mac", "macbook", "host", "mac-inst")
	ses2, err := st2.Create(nil)
	if err != nil {
		t.Fatalf("Create after restart: %v", err)
	}
	got2 := make(chan string, 1)
	go func() {
		req := <-ch
		st2.ProvisionRegistered(ses2.ID(), req.ProviderID)
		got2 <- req.ProviderID
	}()
	if err := st2.Provision(context.Background(), ses2.ID(), "mac", api.HostRequest{}); err != nil {
		t.Fatalf("Provision after restart: %v", err)
	}
	second := <-got2
	if first == second {
		t.Fatalf("provider id %q reused across restart", first)
	}
}
