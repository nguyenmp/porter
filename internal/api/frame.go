package api

import (
	"encoding/binary"
	"fmt"
	"io"
)

// Exec result frames. A client whose exec subscription response carries
// ExecResultFramingHeader may send the output of a tool call in these frames
// instead of as raw bytes. Framing exists for the heartbeat: a command that
// prints nothing for a minute sends nothing, and a proxy in front of the
// server cuts a request body that stays silent past its idle timeout, so the
// client fills the silence with an empty frame the server drops.
const (
	// ExecResultFramingHeader names the frame format a client may use for
	// SessionExecResultPath. The server sets it on the exec subscription
	// response, so a client only frames when the server understands frames,
	// and a server still reads a raw body from a client that does not.
	ExecResultFramingHeader = "X-Porter-Exec-Framing"
	// ExecResultFramingVersion is the value of ExecResultFramingHeader that
	// names the format below: one kind byte, a big-endian uint32 length, then
	// that many payload bytes.
	ExecResultFramingVersion = "1"
	// FrameOutput is a frame whose payload belongs in the tool result.
	FrameOutput byte = 'o'
	// FrameHeartbeat is an empty frame sent while a command is quiet. The
	// server drops it.
	FrameHeartbeat byte = 'h'
	// maxFrameLength rejects a corrupt length before it can hold a decoder in
	// a long read. The client sends at most one read of the tool's stream per
	// frame, so real frames stay far below it.
	maxFrameLength = 1 << 24
)

// WriteFrame writes one frame to w.
func WriteFrame(w io.Writer, kind byte, payload []byte) error {
	if len(payload) > maxFrameLength {
		return fmt.Errorf("frame payload of %d bytes is too large", len(payload))
	}
	var hdr [5]byte
	hdr[0] = kind
	binary.BigEndian.PutUint32(hdr[1:], uint32(len(payload)))
	if _, err := w.Write(hdr[:]); err != nil {
		return err
	}
	if len(payload) == 0 {
		return nil
	}
	_, err := w.Write(payload)
	return err
}

// ReadFrame reads one frame. It returns io.EOF only when the body ends on a
// frame boundary; a body cut partway through a frame returns
// io.ErrUnexpectedEOF, so a proxy that drops the upload mid-frame still reads
// as a cut stream rather than a short result.
func ReadFrame(r io.Reader) (kind byte, payload []byte, err error) {
	var hdr [5]byte
	if _, err := io.ReadFull(r, hdr[:]); err != nil {
		return 0, nil, err
	}
	n := binary.BigEndian.Uint32(hdr[1:])
	if n > maxFrameLength {
		return 0, nil, fmt.Errorf("frame of %d bytes is too large", n)
	}
	if n == 0 {
		return hdr[0], nil, nil
	}
	payload = make([]byte, n)
	if _, err := io.ReadFull(r, payload); err != nil {
		if err == io.EOF {
			err = io.ErrUnexpectedEOF
		}
		return 0, nil, err
	}
	return hdr[0], payload, nil
}
