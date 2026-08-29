// Package recall owns the model-view projection that trims oversized tool
// results and the read_output tool that loads the trimmed bytes back.
//
// A tool result is stored in the database and broadcast to the UI in full;
// only the model's view — the history sent on each LLM request — is truncated
// to a head + tail slice, with a read_output tool the model can call to read
// any byte window of the original output. read_output results are the one
// exception to truncation: the window bytes are served to the model's context
// in full (for the current turn only), while the persisted/broadcast copy is a
// short placeholder, so the window is never duplicated in the database.
//
// Truncation is byte-exact: the head is [0, HeadBytes), the tail is the last
// TailBytes, and read_output offsets address the raw output bytes, so the head
// and a read_output(offset=HeadBytes) continue with no gap. A UTF-8 rune split
// at a seam is tolerated (tokenizers are byte-tolerant); line-based reads are a
// future option. HeadBytes/TailBytes are hard-coded here — deliberately not
// configurable — and tuned with code edits when needed.
package recall

import (
	"encoding/json"
	"errors"
	"fmt"
	"strconv"
	"strings"

	"porter/internal/llm"
)

// ReadOutputTool is the model-facing name of the recall tool.
const ReadOutputTool = "read_output"

// HeadBytes and TailBytes are the hard-coded head and tail slice sizes of the
// model-view truncation. They are not configurable; tune them here if the
// budget needs to change.
const (
	HeadBytes = 8192
	TailBytes = 8192
)

// truncationHeaderPrefix marks a message already in truncated model-view form,
// so the projection stays idempotent even if it ever runs over an
// already-truncated message.
const truncationHeaderPrefix = "[tool output:"

// Meta describes how a tool result's full content is presented to the model.
// It returns non-nil for every tool result (small ones included) so the UI
// badge and the model-view projection have a uniform record.
func Meta(content string) *llm.ToolOutputMeta {
	total := len(content)
	m := &llm.ToolOutputMeta{TotalBytes: total}
	if total > HeadBytes+TailBytes {
		m.Truncated = true
		m.ShownBytes = HeadBytes + TailBytes
	} else {
		m.ShownBytes = total
	}
	return m
}

// ProjectModelView returns the model's view of committed history: every
// role-"tool" message whose content is larger than the head+tail budget is
// replaced with its truncated form, unless it is a Recall message (a read_output
// window, which is already the bytes the model asked for). The returned slice
// is a fresh copy and the input is never mutated, because History always holds
// the full output — the projection is applied fresh on each request, so it can
// never compound.
func ProjectModelView(msgs []llm.ChatMessage) []llm.ChatMessage {
	out := make([]llm.ChatMessage, len(msgs))
	for i, m := range msgs {
		if m.Role == "tool" {
			if meta := m.ToolOutput; meta != nil && meta.Recall {
				// read_output window: keep the full bytes in the model view.
			} else if !strings.HasPrefix(m.Content, truncationHeaderPrefix) {
				if m.ToolOutput == nil {
					// A tool message without metadata (e.g. committed before
					// this feature, or a bare role-"tool" message): derive the
					// presentation from the content so the model view is
					// uniformly truncated.
					m.ToolOutput = Meta(m.Content)
				}
				m.Content = Truncate(m.Content, m.ToolCallID, m.ToolOutput)
			}
		}
		out[i] = m
	}
	return out
}

// Truncate renders content in the model view's truncated form: a header that
// states the sizes and how to load the rest, the first HeadBytes bytes, an
// omitted marker, and the last TailBytes bytes. It returns content unchanged
// when it fits within the head+tail budget. callID is the tool_call_id the
// read_output hint addresses (the message's ToolCallID).
func Truncate(content, callID string, meta *llm.ToolOutputMeta) string {
	if meta.TotalBytes <= HeadBytes+TailBytes {
		return content
	}
	if strings.HasPrefix(content, truncationHeaderPrefix) {
		return content
	}
	head := content[:HeadBytes]
	tail := content[meta.TotalBytes-TailBytes:]
	omitted := meta.TotalBytes - HeadBytes - TailBytes
	var b strings.Builder
	fmt.Fprintf(&b, "[tool output: %s of %s bytes (head); last %s shown below.  To load more: read_output(call_id=%q, offset=%d, max_bytes=%d).]\n",
		comma(HeadBytes), comma(meta.TotalBytes), bytesLabel(TailBytes), callID, HeadBytes, meta.TotalBytes-HeadBytes)
	b.WriteString(head)
	fmt.Fprintf(&b, "\n[... %s omitted ...]\n", bytesLabel(omitted))
	b.WriteString(tail)
	return b.String()
}

// bytesLabel renders a byte count in prose: "1 byte" or "1,024 bytes".
func bytesLabel(n int) string {
	if n == 1 {
		return "1 byte"
	}
	return comma(n) + " bytes"
}

// comma formats n with thousands separators ("1048576" -> "1,048,576").
func comma(n int) string {
	s := strconv.Itoa(n)
	var b strings.Builder
	for i, c := range s {
		if i > 0 && (len(s)-i)%3 == 0 {
			b.WriteByte(',')
		}
		b.WriteRune(c)
	}
	return b.String()
}

// readOutputArgs is the parsed read_output tool call.
type readOutputArgs struct {
	CallID   string `json:"call_id"`
	Offset   int    `json:"offset"`
	MaxBytes int    `json:"max_bytes"`
}

// ServeWindow handles a read_output call against the turn's committed history.
// It returns the window text served to the model (a short header plus the
// window bytes), the metadata describing the recall, or an error when the
// arguments are invalid or the call_id is unknown. The agent serves it from
// in-memory history so it works for any execution provider and never crosses
// the exec channel. Omitting max_bytes reads to the end of the output, so
// loading a large result takes a single call rather than a loop.
func ServeWindow(history []llm.ChatMessage, args string) (string, *llm.ToolOutputMeta, error) {
	var in readOutputArgs
	if err := json.Unmarshal([]byte(args), &in); err != nil {
		return "", nil, fmt.Errorf("parse read_output arguments: %w", err)
	}
	if in.CallID == "" {
		return "", nil, errors.New("read_output: call_id is required")
	}
	if in.Offset < 0 {
		return "", nil, errors.New("read_output: offset must be >= 0")
	}
	if in.MaxBytes < 0 {
		return "", nil, errors.New("read_output: max_bytes must be >= 0")
	}
	var data []byte
	var total int
	for _, m := range history {
		if m.Role == "tool" && m.ToolCallID == in.CallID {
			data = []byte(m.Content)
			total = len(m.Content)
			break
		}
	}
	if data == nil {
		return "", nil, fmt.Errorf("read_output: unknown call_id %q (it must be a tool result in this conversation's history)", in.CallID)
	}
	offset := in.Offset
	if offset > total {
		offset = total
	}
	max := total - offset
	if in.MaxBytes > 0 && in.MaxBytes < max {
		max = in.MaxBytes
	}
	meta := &llm.ToolOutputMeta{
		Recall:       true,
		SourceCallID: in.CallID,
		Offset:       offset,
		MaxBytes:     max,
		TotalBytes:   total,
		ShownBytes:   max,
	}
	window := string(data[offset : offset+max])
	return WindowText(meta, window), meta, nil
}

// WindowText renders the text served to the model for a read_output window: a
// header identifying the window, then the window bytes.
func WindowText(meta *llm.ToolOutputMeta, window string) string {
	return fmt.Sprintf("[recall: read_output(call_id=%q, offset=%d, max_bytes=%d) -> bytes %d-%d of %d (%d bytes) served to the model]\n%s",
		meta.SourceCallID, meta.Offset, meta.MaxBytes, meta.Offset, meta.Offset+meta.MaxBytes, meta.TotalBytes, meta.MaxBytes, window)
}

// Placeholder renders the persisted/broadcast copy of a read_output result: a
// short notice instead of the window bytes, so the window is never duplicated
// in the database (the full output lives once, under the source tool result).
// The placeholder still keys the tool message to its call, so the assistant's
// tool_call keeps a role-"tool" sibling (providers reject a call with no
// result), and it records what was served so the UI — and the model on the
// next turn — can see the recall happened.
func Placeholder(meta *llm.ToolOutputMeta) string {
	return fmt.Sprintf("[recall: read_output(call_id=%q, offset=%d, max_bytes=%d) returned bytes %d-%d of %d (%d bytes) to the model context; call read_output to load a different window]",
		meta.SourceCallID, meta.Offset, meta.MaxBytes, meta.Offset, meta.Offset+meta.MaxBytes, meta.TotalBytes, meta.MaxBytes)
}

// Def is the model-facing definition of the read_output tool. The agent
// declares it on every request so a model that receives a truncated tool result
// can load more. Omitted max_bytes reads to the end of the output, so loading a
// large result takes a single call.
func Def() llm.Tool {
	return llm.Tool{
		Type: "function",
		Function: llm.Function{
			Name: ReadOutputTool,
			Description: "Load more of a tool result that was truncated in your context. Tool results larger than " +
				strconv.Itoa(HeadBytes+TailBytes) + " bytes are shown as a head and tail with a read_output hint. " +
				"call_id is the id of the original tool call (shown in the truncation header), offset is the byte offset " +
				"to start at (default 0), and max_bytes limits the returned window — omit it to read the rest of the " +
				"output in one call. Offsets address the raw output bytes.",
			Parameters: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"call_id": map[string]any{
						"type":        "string",
						"description": "The tool_call_id of the original tool result to read (shown in its truncation header).",
					},
					"offset": map[string]any{
						"type":        "integer",
						"description": "Byte offset into the output to start at (default 0).",
					},
					"max_bytes": map[string]any{
						"type":        "integer",
						"description": "Maximum bytes to return (default: the rest of the output, to the end).",
					},
				},
				"required": []string{"call_id"},
			},
		},
	}
}
