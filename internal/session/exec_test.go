package session

import (
	"bytes"
	"errors"
	"io"
	"strings"
	"testing"

	"porter/internal/api"
)

func TestCopyFramedDropsHeartbeats(t *testing.T) {
	var body bytes.Buffer
	for _, kind := range []byte{api.FrameOutput, api.FrameHeartbeat, api.FrameHeartbeat, api.FrameOutput} {
		payload := []byte("out\n")
		if kind == api.FrameHeartbeat {
			payload = nil
		}
		if err := api.WriteFrame(&body, kind, payload); err != nil {
			t.Fatalf("WriteFrame: %v", err)
		}
	}

	var got bytes.Buffer
	if err := copyFramed(&got, &body); err != nil {
		t.Fatalf("copyFramed: %v", err)
	}
	if want := "out\nout\n"; got.String() != want {
		t.Fatalf("output = %q, want %q", got.String(), want)
	}
}

// TestCopyFramedReportsCutStream checks that a body cut inside a frame reads as
// a lost stream, so the agent is told the result is incomplete rather than
// short.
func TestCopyFramedReportsCutStream(t *testing.T) {
	var body bytes.Buffer
	if err := api.WriteFrame(&body, api.FrameOutput, []byte("partial")); err != nil {
		t.Fatalf("WriteFrame: %v", err)
	}
	raw := body.Bytes()

	var got bytes.Buffer
	err := copyFramed(&got, bytes.NewReader(raw[:len(raw)-2]))
	if !errors.Is(err, io.ErrUnexpectedEOF) {
		t.Fatalf("err = %v, want io.ErrUnexpectedEOF", err)
	}
}

func TestCopyFramedRejectsUnknownFrame(t *testing.T) {
	var body bytes.Buffer
	if err := api.WriteFrame(&body, '$', []byte("nope")); err != nil {
		t.Fatalf("WriteFrame: %v", err)
	}
	var got bytes.Buffer
	if err := copyFramed(&got, &body); err == nil {
		t.Fatal("copyFramed accepted an unknown frame kind")
	}
}

// execResult runs ExecResult on a fresh call with a real pipe and returns the
// bytes the agent reads plus the error each side saw.
func execResult(t *testing.T, body io.Reader, framed bool) (read []byte, readErr, resultErr error) {
	t.Helper()
	s := &Session{execCalls: map[string]*execCall{}}
	pr, pw := io.Pipe()
	s.execCalls["call-1"] = &execCall{pw: pw}

	done := make(chan error, 1)
	go func() { done <- s.ExecResult("call-1", body, framed) }()
	read, readErr = io.ReadAll(pr)
	resultErr = <-done
	return read, readErr, resultErr
}

func TestExecResultFramedDropsHeartbeats(t *testing.T) {
	var body bytes.Buffer
	for _, kind := range []byte{api.FrameHeartbeat, api.FrameOutput, api.FrameHeartbeat} {
		payload := []byte("hello\n")
		if kind == api.FrameHeartbeat {
			payload = nil
		}
		if err := api.WriteFrame(&body, kind, payload); err != nil {
			t.Fatalf("WriteFrame: %v", err)
		}
	}

	read, readErr, resultErr := execResult(t, &body, true)
	if readErr != nil || resultErr != nil {
		t.Fatalf("read err = %v, result err = %v", readErr, resultErr)
	}
	if want := "hello\n"; string(read) != want {
		t.Fatalf("agent read %q, want %q", read, want)
	}
}

func TestExecResultRawUnchanged(t *testing.T) {
	read, readErr, resultErr := execResult(t, strings.NewReader("raw output\n"), false)
	if readErr != nil || resultErr != nil {
		t.Fatalf("read err = %v, result err = %v", readErr, resultErr)
	}
	if want := "raw output\n"; string(read) != want {
		t.Fatalf("agent read %q, want %q", read, want)
	}
}

func TestExecResultCutFramed(t *testing.T) {
	var body bytes.Buffer
	if err := api.WriteFrame(&body, api.FrameOutput, []byte("start\n")); err != nil {
		t.Fatalf("WriteFrame: %v", err)
	}
	raw := body.Bytes()

	_, readErr, resultErr := execResult(t, bytes.NewReader(raw[:len(raw)-1]), true)
	if resultErr == nil || !strings.Contains(resultErr.Error(), "cut off") {
		t.Fatalf("result err = %v, want the cut-off message", resultErr)
	}
	if readErr == nil {
		t.Fatal("agent read a cut result as a complete one")
	}
}
