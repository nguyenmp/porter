package llm

import (
	"bufio"
	"io"
	"reflect"
	"strings"
	"testing"
)

func parseLines(t *testing.T, raw string) []string {
	t.Helper()
	var out []string
	sc := bufio.NewScanner(strings.NewReader(raw))
	sc.Buffer(make([]byte, 64*1024), 1024*1024)
	sc.Split(SplitSSE)
	for sc.Scan() {
		out = append(out, sc.Text())
	}
	if err := sc.Err(); err != nil {
		t.Fatalf("scan error: %v", err)
	}
	return out
}

func TestSplitSSE(t *testing.T) {
	raw := "data: {\"a\":1}\n\n" +
		"data: {\"a\":2}\n" +
		"event: ping\n" +
		"data: [DONE]\n"
	got := parseLines(t, raw)
	want := []string{
		`data: {"a":1}`,
		``,
		`data: {"a":2}`,
		`event: ping`,
		`data: [DONE]`,
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("got %q want %q", got, want)
	}
}

// chunkedReader plays back a string in fixed-size pieces so scanner read
// boundaries fall inside the middle of lines.
type chunkedReader struct {
	data []byte
	pos  int
	size int
}

func (c *chunkedReader) Read(p []byte) (int, error) {
	if c.pos >= len(c.data) {
		return 0, io.EOF
	}
	end := c.pos + c.size
	if end > len(c.data) {
		end = len(c.data)
	}
	n := copy(p, c.data[c.pos:end])
	c.pos += n
	return n, nil
}

func TestSplitSSEPartialCrossingBoundary(t *testing.T) {
	full := "data: {\"a\":1}\n\ndata: {\"a\":2}\n"
	sc := bufio.NewScanner(&chunkedReader{data: []byte(full), size: 3})
	sc.Buffer(make([]byte, 64*1024), 1024*1024)
	sc.Split(SplitSSE)
	var out []string
	for sc.Scan() {
		out = append(out, sc.Text())
	}
	if err := sc.Err(); err != nil {
		t.Fatalf("scan error: %v", err)
	}
	want := []string{`data: {"a":1}`, ``, `data: {"a":2}`}
	if !reflect.DeepEqual(out, want) {
		t.Fatalf("got %q want %q", out, want)
	}
}
