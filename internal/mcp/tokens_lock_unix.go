//go:build unix

package mcp

import (
	"os"
	"syscall"
)

// lockTokens takes an exclusive lock on path, creating it, so the token
// store's read-merge-write cycle is serialized across processes. It returns
// the function that releases the lock.
func lockTokens(path string) (func(), error) {
	f, err := os.OpenFile(path, os.O_CREATE|os.O_RDWR, 0o600)
	if err != nil {
		return nil, err
	}
	if err := syscall.Flock(int(f.Fd()), syscall.LOCK_EX); err != nil {
		_ = f.Close()
		return nil, err
	}
	return func() {
		_ = syscall.Flock(int(f.Fd()), syscall.LOCK_UN)
		_ = f.Close()
	}, nil
}
