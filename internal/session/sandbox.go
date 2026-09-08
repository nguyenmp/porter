package session

import "fmt"

// setSandbox marks the chat as sandboxed: it runs in a sandbox served by the
// given provider on the given host. It is called when a sandboxed provider
// registers (a provision resolved) and when the persisted mapping loads at
// server startup. The chat's active provider is pointed at the sandbox
// provider so the pause rule holds from the moment the chat is sandboxed — on
// a server restart that is before the host reconnects, so the chat pauses
// instead of silently running on the server.
func (s *Session) setSandbox(providerID, hostID string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.sandboxProvider = providerID
	s.sandboxHost = hostID
	s.activeExec = providerID
}

// clearSandbox removes the chat's sandbox marker, used when the chat is
// archived (its sandbox released). The chat stops being sandboxed, so the
// pause rule no longer applies to it.
func (s *Session) clearSandbox() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.sandboxProvider = ""
	s.sandboxHost = ""
}

// PausedReason reports why a sandboxed chat cannot run right now, or "" when
// it can. A chat is paused when it is sandboxed on a host, its host provider
// is the active provider, and that provider is offline: rather than silently
// falling back to running on the server — a different environment, whose file
// changes the sandbox would never see — the chat refuses new messages and
// fails tool calls fast until the host reconnects. A chat whose user picked
// another provider (e.g. local) by hand is not paused: that was an explicit
// choice to run elsewhere.
func (s *Session) PausedReason() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.sandboxProvider == "" || s.activeExec != s.sandboxProvider {
		return ""
	}
	if c, ok := s.execClients[s.sandboxProvider]; ok && c.connected {
		return ""
	}
	return fmt.Sprintf("this chat runs on host %q, which is offline; it will resume when the host reconnects", s.sandboxHost)
}
