package tools

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"strings"
	"sync"
	"syscall"
	"time"
)

// pollInterval is how often the tailer re-checks the output file while the
// command is still running. Regular files never block on read and give no
// data-available signal, so output is delivered by polling; 30ms is
// imperceptible in a terminal while keeping CPU use negligible.
const pollInterval = 30 * time.Millisecond

// runShell runs a shell command in the process's working directory and
// returns a stream of its combined stdout/stderr, with a trailing exit-status
// line appended once it finishes.
func runShell(ctx context.Context, args []byte) (io.ReadCloser, error) {
	return runShellDir(ctx, args, "")
}

// runShellDir is runShell with an explicit working directory. Execution hosts
// run each provisioned sandbox's commands in the sandbox's directory this
// way; an empty dir inherits the process cwd. It returns a stream of the
// command's combined stdout/stderr, with a trailing exit-status line appended
// once it finishes.
//
// Output goes to a temp file (not a pipe) and the returned stream tails it.
// The point is what "the command is done" means. A pipe delivers EOF only when
// every process that inherited the write end has exited, so a backgrounded or
// daemonized descendant keeps the pipe open forever: cmd.Wait never returns,
// the exit-status line is never written, and the agent loop reading the stream
// hangs no matter which visible PIDs were killed. A file makes completion
// deterministic: cmd.Wait returns as soon as the direct child exits, the stream
// drains whatever output arrived, appends the exit-status line, and returns
// EOF. Descendants that outlive the tool call keep writing to the (unlinked)
// temp file; their late output is not part of the stream.
func runShellDir(ctx context.Context, args []byte, dir string) (io.ReadCloser, error) {
	var in struct {
		Command string `json:"command"`
	}
	if err := json.Unmarshal(args, &in); err != nil {
		return nil, fmt.Errorf("parse shell arguments: %w", err)
	}
	if strings.TrimSpace(in.Command) == "" {
		return nil, errors.New("shell command is empty")
	}

	// Create the output file and unlink it immediately: the open descriptor
	// keeps the inode alive, so a crash never leaves a stray file on disk, and
	// a descendant that outlives the call just writes to the unlinked inode.
	f, err := os.CreateTemp("", "porter-*.out")
	if err != nil {
		return nil, fmt.Errorf("create output file: %w", err)
	}
	if err := os.Remove(f.Name()); err != nil {
		_ = f.Close()
		return nil, fmt.Errorf("unlink output file: %w", err)
	}

	cmd := exec.CommandContext(ctx, "sh", "-c", in.Command)
	if dir != "" {
		cmd.Dir = dir
	}
	// Run the whole tree in its own process group so that stopping the run can
	// reach every descendant, not just the shell. A shell command that forks
	// (foo; bar, backgrounded jobs, test runners) keeps its children in the
	// same group, so a stop that signals -pid kills the shell and each of those
	// children together — the direct child's exit is what unblocks the stream,
	// and the group kill is what stops the actual work.
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
	// Cancellation kills the entire group, not just the direct child.
	// CommandContext's default Cancel only kills cmd.Process (the shell), which
	// would orphan its children; with a file-backed stream that unblocks the
	// agent, but the work would keep running. Killing -pid stops the whole tree.
	cmd.Cancel = func() error {
		return syscall.Kill(-cmd.Process.Pid, syscall.SIGKILL)
	}
	// One *os.File for both streams: os/exec dup2s the same fd to stdout and
	// stderr, so the combined stream preserves their relative order (as the old
	// single pipe did) while Wait returns on process exit instead of on
	// "every writer gone".
	cmd.Stdout = f
	cmd.Stderr = f
	if err := cmd.Start(); err != nil {
		_ = f.Close()
		return nil, fmt.Errorf("start shell command: %w", err)
	}

	t := &tailReader{ctx: ctx, f: f, done: make(chan struct{})}
	go func() {
		defer close(t.done)
		if err := cmd.Wait(); err != nil {
			if ee, ok := err.(*exec.ExitError); ok {
				t.code = ee.ExitCode()
			} else {
				t.err = err
			}
		}
	}()
	return t, nil
}

// tailReader streams the contents of a command's output file as they are
// written. Read returns new output as it lands, polling while the command is
// still running; once the command's direct child exits it drains the file,
// appends the exit-status line, and returns io.EOF (or the wait error, if the
// process was not reaped cleanly). Cancellation ends the stream promptly.
// Read is safe for concurrent use; Close is idempotent.
type tailReader struct {
	ctx  context.Context
	f    *os.File
	done chan struct{} // closed when the direct child has exited

	mu      sync.Mutex
	code    int
	err     error
	pos     int64  // read position in f
	pending []byte // exit-status line not yet emitted
	eof     bool   // EOF has already been returned
	closed  bool
}

func (t *tailReader) Read(p []byte) (int, error) {
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.eof {
		return 0, io.EOF
	}
	if t.closed {
		return 0, io.ErrClosedPipe
	}

	// Emit any pending exit-status line before looking for new output.
	if n := copy(p, t.pending); n > 0 {
		t.pending = t.pending[n:]
		if len(t.pending) == 0 {
			t.eof = true
		}
		return n, nil
	}

	for {
		// New output since the last read position?
		if n, err := t.f.ReadAt(p, t.pos); n > 0 {
			t.pos += int64(n)
			return n, nil
		} else if err != nil && err != io.EOF {
			t.eof = true
			return 0, err
		}

		select {
		case <-t.done:
			// The direct child exited. Drain anything written right at exit,
			// then emit the exit-status line and EOF.
			if n, _ := t.f.ReadAt(p, t.pos); n > 0 {
				t.pos += int64(n)
				return n, nil
			}
			if t.err != nil {
				t.eof = true
				return 0, t.err
			}
			t.pending = []byte(fmt.Sprintf("\nexit code: %d\n", t.code))
			if n := copy(p, t.pending); n > 0 {
				t.pending = t.pending[n:]
				if len(t.pending) == 0 {
					t.eof = true
				}
				return n, nil
			}
			t.eof = true
			return 0, io.EOF
		case <-t.ctx.Done():
			// Cancelled: the whole process group is being killed; drain what
			// arrived and end so the agent loop is not left waiting on a tool
			// call that will never finish.
			if n, _ := t.f.ReadAt(p, t.pos); n > 0 {
				t.pos += int64(n)
				return n, nil
			}
			t.eof = true
			return 0, io.EOF
		case <-time.After(pollInterval):
			// Command still running; keep polling for output.
		}
	}
}

// Close releases the output file. It is idempotent. A descendant that outlives
// the call keeps writing to the unlinked inode, which is freed when the last
// descriptor closes.
func (t *tailReader) Close() error {
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.closed {
		return nil
	}
	t.closed = true
	return t.f.Close()
}
