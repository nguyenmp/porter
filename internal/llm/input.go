package llm

import "encoding/json"

// The Responses API has no `messages` array. It takes `input`: one flat list of
// items, where the item's `type` says what it is — a message, a function call
// the model asked for, or the output of running that call. One porter message
// can therefore become more than one item; see wireInput.

// inputMessage is a message item: prose, plus any images attached to it.
type inputMessage struct {
	Type    string `json:"type"` // always "message"
	Role    string `json:"role"`
	Content []any  `json:"content"`
}

// textPart is the prose of a message item. The type says who wrote it:
// "input_text" for what goes in, "output_text" for prose the assistant wrote on
// an earlier turn.
//
// Text carries no omitempty so an empty message still serializes as an explicit
// empty string. Providers reject a part with no text field, and porter sends a
// text-first part for every message, including one that arrived empty.
type textPart struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

// imagePart is one image attached to a message, as a complete data URL (bytes
// inline, prefix included), so the bytes never depend on the provider reaching
// the porter server. Unlike Chat Completions, which nested the location under
// an image_url object, Responses takes the URL as a plain string.
type imagePart struct {
	Type     string `json:"type"` // always "input_image"
	ImageURL string `json:"image_url"`
}

// functionCallItem is a tool call the assistant made on an earlier turn, sent
// back so the model sees what it asked for before it sees the result.
//
// The Responses API splits one identifier into two: the item's own id, and the
// call_id that pairs a call with its output. porter stores only the id the
// model streamed (which is the call id), so it fills CallID and leaves the
// item id out. A gateway that demands an item id can be served by echoing the
// same value in both fields.
type functionCallItem struct {
	Type      string `json:"type"` // always "function_call"
	CallID    string `json:"call_id"`
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

// functionCallOutputItem is the result of running one tool call: the output the
// tool produced, paired to its call by call id.
type functionCallOutputItem struct {
	Type   string `json:"type"` // always "function_call_output"
	CallID string `json:"call_id"`
	Output string `json:"output"`
}

// HasImages reports whether any of the messages carries an attached image, so
// callers can avoid dumping image bytes where they would only get in the way.
func HasImages(msgs []ChatMessage) bool {
	for _, m := range msgs {
		if len(m.Images) > 0 {
			return true
		}
	}
	return false
}

// wireInput converts internal messages to the item list the provider expects.
//
// The mapping is not one to one. An assistant turn that both wrote prose and
// asked for tools becomes a message item, then one function_call item per
// call, so every call precedes the output that answers it. A role-"tool"
// message becomes a function_call_output item, paired by call id. Everything
// else — a system, user, or developer turn — becomes a message item holding
// the text and one part per attached image.
//
// Reasoning is deliberately left out: the model's chain of thought is kept for
// the user to read, and is never sent back to the model.
func wireInput(msgs []ChatMessage) ([]json.RawMessage, error) {
	out := make([]json.RawMessage, 0, len(msgs))
	add := func(item any) error {
		b, err := json.Marshal(item)
		if err != nil {
			return err
		}
		out = append(out, b)
		return nil
	}

	for _, m := range msgs {
		switch m.Role {
		case "assistant":
			if m.Content != "" {
				if err := add(inputMessage{
					Type:    "message",
					Role:    m.Role,
					Content: []any{textPart{Type: "output_text", Text: m.Content}},
				}); err != nil {
					return nil, err
				}
			}
			for _, call := range m.ToolCalls {
				if err := add(functionCallItem{
					Type:      "function_call",
					CallID:    call.ID,
					Name:      call.Function.Name,
					Arguments: call.Function.Arguments,
				}); err != nil {
					return nil, err
				}
			}
		case "tool":
			if err := add(functionCallOutputItem{
				Type:   "function_call_output",
				CallID: m.ToolCallID,
				Output: m.Content,
			}); err != nil {
				return nil, err
			}
		default:
			// The text part leads, so a message whose text is empty still
			// carries one part and the shape stays valid.
			parts := []any{textPart{Type: "input_text", Text: m.Content}}
			for _, img := range m.Images {
				parts = append(parts, imagePart{Type: "input_image", ImageURL: img.URL})
			}
			if err := add(inputMessage{Type: "message", Role: m.Role, Content: parts}); err != nil {
				return nil, err
			}
		}
	}
	return out, nil
}
