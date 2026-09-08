package session

import (
	"context"
	"io"
	"strings"
	"testing"

	"porter/internal/api"
)

// TestSandboxedChatPausesAndResumes covers the pause rule end to end at the
// session level: a sandboxed chat whose host provider is offline is paused
// (it refuses new messages and tool calls fail fast) rather than silently
// falling back to the server; selecting local by hand is the explicit opt-out;
// and the host reconnecting resumes the chat automatically.
func TestSandboxedChatPausesAndResumes(t *testing.T) {
	st := newTestStore(t)
	ses, err := st.Create(nil)
	if err != nil {
		t.Fatalf("Create: %v", err)
	}

	// The host provider connects and the chat becomes sandboxed (as happens
	// when a provision resolves).
	providerID := ses.RegisterExec(make(chan api.ExecRequest, 8), "mac-provider-1", "macbook", "host")
	ses.setSandbox(providerID, "mac")
	if reason := ses.PausedReason(); reason != "" {
		t.Fatalf("running chat reported paused: %v", reason)
	}

	// The host drops: the chat pauses instead of reverting to local.
	ses.UnregisterExec(providerID)
	if reason := ses.PausedReason(); reason == "" {
		t.Fatal("chat with offline host provider is not paused")
	}

	// The active provider is paused, so a tool call fails fast with a clear
	// error instead of running on the server.
	p := ses.provider()
	stream, err := p.Run(context.Background(), "shell", []byte("ls"))
	if err == nil || !strings.Contains(err.Error(), "offline") {
		t.Fatalf("Run while paused = %v, want an offline error", err)
	}
	if stream != nil {
		io.Copy(io.Discard, stream)
		stream.Close()
	}

	// The exec status reports the host provider disconnected — not "local",
	// which would imply execution moved to the server.
	stt := ses.ExecStatus()
	if stt.Connected || stt.Kind != "host" || stt.ActiveID != providerID {
		t.Fatalf("ExecStatus while paused = %+v, want host provider disconnected", stt)
	}

	// Selecting local by hand is the explicit opt-out: the chat runs on the
	// server and is no longer paused.
	if err := ses.SelectExec("local"); err != nil {
		t.Fatalf("SelectExec(local): %v", err)
	}
	if reason := ses.PausedReason(); reason != "" {
		t.Fatalf("chat on local reported paused: %v", reason)
	}

	// The host provider reconnects: it takes over again and the chat resumes.
	ses.RegisterExec(make(chan api.ExecRequest, 8), providerID, "macbook", "host")
	if reason := ses.PausedReason(); reason != "" {
		t.Fatalf("reconnected chat reported paused: %v", reason)
	}

	// Releasing the sandbox (archive) unsandboxes the chat.
	ses.clearSandbox()
	if reason := ses.PausedReason(); reason != "" {
		t.Fatalf("released chat reported paused: %v", reason)
	}
}

// TestNonSandboxedChatStillRevertsToLocal proves the pause rule does not
// change behavior for plain (never-sandboxed) chats: a disconnected provider
// still falls back to the server.
func TestNonSandboxedChatStillRevertsToLocal(t *testing.T) {
	st := newTestStore(t)
	ses, err := st.Create(nil)
	if err != nil {
		t.Fatalf("Create: %v", err)
	}
	id := ses.RegisterExec(make(chan api.ExecRequest, 8), "", "laptop", "remote")
	ses.UnregisterExec(id)
	if reason := ses.PausedReason(); reason != "" {
		t.Fatalf("plain chat reported paused: %v", reason)
	}
	if stt := ses.ExecStatus(); stt.Kind != "local" || stt.Connected {
		t.Fatalf("ExecStatus after plain disconnect = %+v, want local", stt)
	}
}

// TestSandboxedHostProviderTombstoneKeptWhileInactive proves a sandboxed
// chat whose user switched to another provider does not pause when the host
// drops (the user chose to run elsewhere), and the host stays listed as an
// offline provider in the registry.
func TestSandboxedHostProviderTombstoneKeptWhileInactive(t *testing.T) {
	st := newTestStore(t)
	ses, err := st.Create(nil)
	if err != nil {
		t.Fatalf("Create: %v", err)
	}
	providerID := ses.RegisterExec(make(chan api.ExecRequest, 8), "mac-provider-2", "macbook", "host")
	ses.setSandbox(providerID, "mac")
	if err := ses.SelectExec("local"); err != nil {
		t.Fatalf("SelectExec(local): %v", err)
	}
	// The host drops while local is selected: no pause, but the provider stays
	// listed (offline) in the registry.
	ses.UnregisterExec(providerID)
	if reason := ses.PausedReason(); reason != "" {
		t.Fatalf("chat on local reported paused: %v", reason)
	}
	stt := ses.ExecStatus()
	found := false
	for _, cl := range stt.Clients {
		if cl.ID == providerID {
			found = true
			if cl.Connected {
				t.Fatal("host provider still listed as connected after drop")
			}
		}
	}
	if !found {
		t.Fatal("host provider missing from registry after drop")
	}
}

// TestStaleUnregisterCannotDisconnectReplacement proves a disconnect from a
// superseded registration (same provider id, newer connection) is a no-op:
// a host that restarts before the server notices the old exec connection must
// not pause the chat that the new connection just resumed.
func TestStaleUnregisterCannotDisconnectReplacement(t *testing.T) {
	st := newTestStore(t)
	ses, err := st.Create(nil)
	if err != nil {
		t.Fatalf("Create: %v", err)
	}
	providerID := "mac-provider-9"

	// First connection registers; the chat is sandboxed and running.
	ch1 := make(chan api.ExecRequest, 8)
	ses.RegisterExec(ch1, providerID, "macbook", "host")
	ses.setSandbox(providerID, "mac")

	// The host process restarts: a second connection replaces the first, and
	// the chat resumes on it.
	ch2 := make(chan api.ExecRequest, 8)
	ses.RegisterExec(ch2, providerID, "macbook", "host")

	// The server finally notices the first connection died: its teardown must
	// not disconnect the live replacement.
	ses.UnregisterExec(providerID, ch1)
	if reason := ses.PausedReason(); reason != "" {
		t.Fatalf("stale unregister paused a live chat: %v", reason)
	}
	stt := ses.ExecStatus()
	if !stt.Connected || stt.Kind != "host" {
		t.Fatalf("ExecStatus after stale unregister = %+v, want host connected", stt)
	}

	// The live connection's own teardown pauses the chat as expected.
	ses.UnregisterExec(providerID, ch2)
	if reason := ses.PausedReason(); reason == "" {
		t.Fatal("live unregister did not pause the chat")
	}
}
