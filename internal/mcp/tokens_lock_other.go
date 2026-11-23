//go:build !unix

package mcp

// lockTokens is a no-op where flock is unavailable (Windows). Writes are still
// atomic there, so a reader never sees a partial file; only the merge between
// two processes is unguarded.
func lockTokens(path string) (func(), error) { return func() {}, nil }
