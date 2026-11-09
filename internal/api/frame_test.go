package api

import (
	"bytes"
	"encoding/binary"
	"errors"
	"io"
	"testing"
)

func TestFrameRoundTrip(t *testing.T) {
	var buf bytes.Buffer
	frames := []struct {
		kind    byte
		payload string
	}{
		{FrameOutput, "first\n"},
		{FrameHeartbeat, ""},
		{FrameOutput, "second\n"},
	}
	for _, f := range frames {
		if err := WriteFrame(&buf, f.kind, []byte(f.payload)); err != nil {
			t.Fatalf("WriteFrame(%q): %v", f.kind, err)
		}
	}
	for i, want := range frames {
		kind, payload, err := ReadFrame(&buf)
		if err != nil {
			t.Fatalf("frame %d: ReadFrame: %v", i, err)
		}
		if kind != want.kind || string(payload) != want.payload {
			t.Fatalf("frame %d = (%q, %q), want (%q, %q)", i, kind, payload, want.kind, want.payload)
		}
	}
	if _, _, err := ReadFrame(&buf); err != io.EOF {
		t.Fatalf("after the last frame: err = %v, want io.EOF", err)
	}
}

func TestReadFrameTruncated(t *testing.T) {
	// A body cut partway through the header, and one cut partway through a
	// payload, must both read as unexpected EOF: the server turns that into
	// "the output stream was cut off", and never into a short result.
	var full bytes.Buffer
	if err := WriteFrame(&full, FrameOutput, []byte("hello")); err != nil {
		t.Fatalf("WriteFrame: %v", err)
	}
	b := full.Bytes()

	for _, tc := range []struct {
		name string
		body []byte
	}{
		{"header", b[:3]},
		{"payload", b[:len(b)-2]},
	} {
		if _, _, err := ReadFrame(bytes.NewReader(tc.body)); !errors.Is(err, io.ErrUnexpectedEOF) {
			t.Errorf("%s cut: err = %v, want io.ErrUnexpectedEOF", tc.name, err)
		}
	}
}

func TestReadFrameRejectsOversizedLength(t *testing.T) {
	var hdr [5]byte
	hdr[0] = FrameOutput
	binary.BigEndian.PutUint32(hdr[1:], maxFrameLength+1)
	_, _, err := ReadFrame(bytes.NewReader(hdr[:]))
	if err == nil {
		t.Fatal("ReadFrame accepted a length above maxFrameLength")
	}
	if errors.Is(err, io.ErrUnexpectedEOF) {
		t.Fatalf("oversized length reported as a cut stream: %v", err)
	}
}
