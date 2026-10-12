package llm

import "encoding/json"

// contentPart is one element of a multi-part Chat Completions content array:
// either prose or an image the provider must look at. The shape follows the
// Chat Completions format, where an image part carries its URL (here a data
// URL, bytes inline) under a nested image_url object.
type contentPart struct {
	Type     string    `json:"type"`
	Text     string    `json:"text,omitempty"`
	ImageURL *imageURL `json:"image_url,omitempty"`
}

// imageURL is the payload of an image content part: the image location. porter
// always sends a data URL, so the bytes never depend on the provider reaching
// the porter server.
type imageURL struct {
	URL string `json:"url"`
}

// contentParts renders a message's content in the multi-part form: the text
// first (when the message has any), then one part per attached image. A
// message with images and no prose therefore has no text part at all, which is
// what a pure image upload should look like.
func contentParts(m ChatMessage) (json.RawMessage, error) {
	parts := make([]contentPart, 0, len(m.Images)+1)
	if m.Content != "" {
		parts = append(parts, contentPart{Type: "text", Text: m.Content})
	}
	for _, img := range m.Images {
		parts = append(parts, contentPart{Type: "image_url", ImageURL: &imageURL{URL: img.URL}})
	}
	return json.Marshal(parts)
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

// wireMessages converts internal messages to the shape the provider expects.
// A message without images is sent as it is: its own JSON encoding, unchanged.
// A message with images has its `content` replaced by a parts array, because
// Chat Completions only accepts image input in that form, and its `images`
// field dropped, because the provider must not see a field it does not know.
//
// Only the content key is rewritten, so every other field on the message —
// including ones added later — keeps traveling automatically.
func wireMessages(msgs []ChatMessage) ([]json.RawMessage, error) {
	out := make([]json.RawMessage, 0, len(msgs))
	for _, m := range msgs {
		b, err := json.Marshal(m)
		if err != nil {
			return nil, err
		}
		if len(m.Images) == 0 {
			out = append(out, b)
			continue
		}
		var obj map[string]json.RawMessage
		if err := json.Unmarshal(b, &obj); err != nil {
			return nil, err
		}
		parts, err := contentParts(m)
		if err != nil {
			return nil, err
		}
		obj["content"] = parts
		delete(obj, "images")
		b, err = json.Marshal(obj)
		if err != nil {
			return nil, err
		}
		out = append(out, b)
	}
	return out, nil
}
