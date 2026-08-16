package llm

import (
	"bufio"
	"bytes"
	"io"
)

// dataPrefix marks server-sent-event payload lines we care about.
const dataPrefix = "data:"

// SplitSSE is a bufio.SplitFunc that splits a raw SSE stream into its
// payload lines. It returns each `data:` line verbatim (with the leading
// `data:` marker) as a single token, discarding blank keep-alive lines and
// carrying over partial lines across read boundaries.
func SplitSSE(data []byte, atEOF bool) (advance int, token []byte, err error) {
	if atEOF && len(data) == 0 {
		return 0, nil, nil
	}
	if i := bytes.IndexByte(data, '\n'); i >= 0 {
		return i + 1, data[:i], nil
	}
	if atEOF {
		return len(data), data, nil
	}
	// No complete line yet; request more data.
	return 0, nil, nil
}

// SSELines returns a channel that yields the raw payload lines (each
// including its `data:` marker) from an SSE stream, then closes. Reading
// from the channel is the way this milestone consumes the wire as-is.
func SSELines(r io.Reader) <-chan string {
	out := make(chan string)
	go func() {
		defer close(out)
		sc := bufio.NewScanner(r)
		sc.Buffer(make([]byte, 64*1024), 1024*1024)
		sc.Split(SplitSSE)
		for sc.Scan() {
			line := sc.Text()
			if !bytes.HasPrefix([]byte(line), []byte(dataPrefix)) {
				continue
			}
			if bytes.Equal(bytes.TrimSpace([]byte(line)), []byte(dataPrefix)) {
				continue
			}
			out <- line
		}
	}()
	return out
}
