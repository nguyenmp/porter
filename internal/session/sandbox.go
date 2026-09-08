package session

import "os"

// setLocalDir records that this chat's local provider works in its own
// per-chat sandbox folder (localDir, "" = the chat predates local sandboxing
// and runs in the server's working directory). Set at creation and when the
// persisted flag loads at startup; never cleared, so an archived chat can be
// unarchived back into its sandbox.
func (s *Session) setLocalDir(dir string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.localDir = dir
}

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
// it can. A chat is paused when the provider it would run on cannot serve it:
// a host-sandboxed chat whose host provider is the active one and is offline,
// or a locally-sandboxed chat whose folder has gone missing. Rather than
// silently falling back to a different environment — the server's own working
// directory, whose file changes the sandbox would never see — the chat
// refuses new messages and fails tool calls fast. The precedence matches what
// would run the turn: a connected remote client first, then the host-sandbox
// pause rule, then the local folder check. A chat whose user picked another
// provider by hand (e.g. local on a host chat) is not paused for that host:
// that was an explicit choice to run elsewhere.
func (s *Session) PausedReason() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	// A connected remote client (a REPL, or a live host provider) runs the
	// turn; its environment is fine even if the local folder is missing.
	if c, ok := s.execClients[s.activeExec]; ok && c.connected && c.kind != "local" {
		return ""
	}
	if s.sandboxProvider != "" && s.activeExec == s.sandboxProvider {
		if c, ok := s.execClients[s.sandboxProvider]; ok && c.connected {
			return ""
		}
		return hostOfflineReason(s.sandboxHost)
	}
	// The turn would run on the local provider: if this chat owns a sandbox
	// folder, that folder must exist. A missing folder means the chat's file
	// state is gone (deleted by hand); running in the server's working
	// directory would silently resurrect the exact bug local sandboxing
	// fixes, so the chat stays paused with a clear reason instead.
	if s.localDir != "" {
		if _, err := os.Stat(s.localDir); err != nil {
			return localSandboxMissingReason(s.localDir)
		}
	}
	return ""
}
