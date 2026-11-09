package client

import (
	"context"
	"io"
	"time"

	"porter/internal/api"
)

// execHeartbeatInterval is how long an exec upload may stay silent before the
// client sends a heartbeat frame. A proxy in front of the server cuts a request
// body that sends nothing for its idle timeout — Caddy's default is a minute —
// so the interval stays well under the smallest limit a proxy is likely to use.
var execHeartbeatInterval = 10 * time.Second

// framedBody turns a tool's output stream into the framed body of an exec
// result upload, inserting a heartbeat frame whenever the output stays quiet
// for execHeartbeatInterval. A Read on the tool's stream blocks until the
// command writes something, so the reads run on their own goroutine and this
// returns a pipe: the transport then receives either the next output frame or a
// heartbeat, whichever comes first.
func framedBody(ctx context.Context, src io.Reader) io.Reader {
	pr, pw := io.Pipe()
	go func() {
		defer pw.Close()

		type chunk struct {
			data []byte
			err  error
		}
		chunks := make(chan chunk)
		go func() {
			buf := make([]byte, 32*1024)
			for {
				n, err := src.Read(buf)
				var data []byte
				if n > 0 {
					data = append([]byte(nil), buf[:n]...)
				}
				select {
				case chunks <- chunk{data: data, err: err}:
				case <-ctx.Done():
					return
				}
				if err != nil {
					return
				}
			}
		}()

		last := time.Now()
		tick := time.NewTicker(execHeartbeatInterval)
		defer tick.Stop()
		for {
			select {
			case <-ctx.Done():
				pw.CloseWithError(ctx.Err())
				return
			case c := <-chunks:
				if len(c.data) > 0 {
					if err := api.WriteFrame(pw, api.FrameOutput, c.data); err != nil {
						return
					}
					last = time.Now()
				}
				if c.err != nil {
					// A clean end of output ends the body here, which the
					// server reads as a complete result. Any other error has to
					// reach the server as a broken upload, so the model is told
					// the result is cut off rather than complete.
					if c.err != io.EOF {
						pw.CloseWithError(c.err)
					}
					return
				}
			case <-tick.C:
				if time.Since(last) < execHeartbeatInterval {
					continue
				}
				if err := api.WriteFrame(pw, api.FrameHeartbeat, nil); err != nil {
					return
				}
				last = time.Now()
			}
		}
	}()
	return pr
}
