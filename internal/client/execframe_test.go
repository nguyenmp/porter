package client

import (
	"context"
	"errors"
	"io"
	"testing"
	"time"

	"porter/internal/api"
)

// quietThenOutput stays silent until release is closed, then returns data once
// and io.EOF. It stands in for a command that prints nothing for a while.
type quietThenOutput struct {
	release <-chan struct{}
	data    []byte
	sent    bool
}

func (q *quietThenOutput) Read(p []byte) (int, error) {
	if !q.sent {
		<-q.release
		q.sent = true
		return copy(p, q.data), nil
	}
	return 0, io.EOF
}

// errReader fails immediately, standing in for a tool stream that breaks.
type errReader struct{ err error }

func (e errReader) Read([]byte) (int, error) { return 0, e.err }

// readFrames decodes frames from body on its own goroutine, so a test can wait
// on a channel instead of blocking forever when the body stops producing.
func readFrames(body io.Reader) <-chan apiFrame {
	out := make(chan apiFrame, 64)
	go func() {
		defer close(out)
		for {
			kind, payload, err := api.ReadFrame(body)
			if err != nil {
				return
			}
			out <- apiFrame{kind: kind, payload: string(payload)}
		}
	}()
	return out
}

type apiFrame struct {
	kind    byte
	payload string
}

func nextFrame(t *testing.T, frames <-chan apiFrame) (apiFrame, bool) {
	t.Helper()
	select {
	case f, ok := <-frames:
		return f, ok
	case <-time.After(2 * time.Second):
		t.Fatal("no frame within 2s")
		return apiFrame{}, false
	}
}

// TestFramedBodyHeartbeatsWhileQuiet checks the point of the framing: a command
// that prints nothing still uploads, because heartbeats keep the body moving.
func TestFramedBodyHeartbeatsWhileQuiet(t *testing.T) {
	old := execHeartbeatInterval
	execHeartbeatInterval = 10 * time.Millisecond
	t.Cleanup(func() { execHeartbeatInterval = old })

	release := make(chan struct{})
	body := framedBody(context.Background(), &quietThenOutput{release: release, data: []byte("late output\n")})
	frames := readFrames(body)

	// Nothing has been printed yet, so the body must be all heartbeats.
	for i := 0; i < 2; i++ {
		f, ok := nextFrame(t, frames)
		if !ok {
			t.Fatalf("heartbeat %d: body ended while the command was quiet", i)
		}
		if f.kind != api.FrameHeartbeat || f.payload != "" {
			t.Fatalf("frame %d = (%q, %q), want a heartbeat", i, f.kind, f.payload)
		}
	}

	close(release)
	for {
		f, ok := nextFrame(t, frames)
		if !ok {
			t.Fatal("body ended without the output frame")
		}
		if f.kind == api.FrameHeartbeat {
			continue
		}
		if want := "late output\n"; f.kind != api.FrameOutput || f.payload != want {
			t.Fatalf("frame = (%q, %q), want output %q", f.kind, f.payload, want)
		}
		break
	}
	// The source said io.EOF, so the body ends after the output.
	for {
		if f, ok := nextFrame(t, frames); ok {
			if f.kind != api.FrameHeartbeat {
				t.Fatalf("unexpected frame after the output: (%q, %q)", f.kind, f.payload)
			}
			continue
		}
		break
	}
}

// TestFramedBodyPassesSourceError checks that a broken tool stream reaches the
// server as a broken upload, which it reports as a cut result.
func TestFramedBodyPassesSourceError(t *testing.T) {
	old := execHeartbeatInterval
	execHeartbeatInterval = time.Hour // no heartbeats in this test
	t.Cleanup(func() { execHeartbeatInterval = old })

	boom := errors.New("stream broke")
	body := framedBody(context.Background(), errReader{err: boom})
	if _, _, err := api.ReadFrame(body); !errors.Is(err, boom) {
		t.Fatalf("ReadFrame err = %v, want %v", err, boom)
	}
}
