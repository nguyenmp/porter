package llm

import (
	"encoding/json"
	"fmt"
	"strings"
)

// ResolveImage turns an image location stored on a message into the data URL
// the provider needs. A message names an image by its hosted URL
// ("/assets/sessions/..."), and the provider cannot reach that URL, so the
// bytes must travel inline. The caller supplies this function: the server knows
// how to read its own assets, and this package does not. A location that is
// already a data URL never reaches the resolver.
type ResolveImage func(url string) (string, error)

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
//
// Each image is turned into a data URL through resolve. An image the resolver
// cannot read becomes a short text note instead of an error, because the
// message stays in history: a request that failed here would fail on every
// later turn too.
func contentParts(m ChatMessage, resolve ResolveImage) (json.RawMessage, error) {
	parts := make([]contentPart, 0, len(m.Images)+1)
	if m.Content != "" {
		parts = append(parts, contentPart{Type: "text", Text: m.Content})
	}
	for _, img := range m.Images {
		url, err := resolveImageURL(img, resolve)
		if err != nil {
			parts = append(parts, contentPart{Type: "text", Text: imageUnavailable(img, err)})
			continue
		}
		parts = append(parts, contentPart{Type: "image_url", ImageURL: &imageURL{URL: url}})
	}
	return json.Marshal(parts)
}

// resolveImageURL returns the data URL for one attached image. A stored data
// URL is already the wire form and comes back unchanged, so a message written
// before images became assets keeps working.
func resolveImageURL(img ImagePart, resolve ResolveImage) (string, error) {
	if strings.HasPrefix(img.URL, "data:") {
		return img.URL, nil
	}
	if resolve == nil {
		return "", fmt.Errorf("no image resolver is set")
	}
	return resolve(img.URL)
}

// imageUnavailable is the text that stands in for an attached image the request
// could not read. It names the image and the reason, so the model and the user
// both see that a picture was meant to be there.
func imageUnavailable(img ImagePart, err error) string {
	name := img.Name
	if name == "" {
		name = img.URL
	}
	return fmt.Sprintf("[image unavailable: %s (%v)]", name, err)
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
// Each image location is resolved to a data URL through resolve; see
// ResolveImage. Only the content key is rewritten, so every other field on the
// message — including ones added later — keeps traveling automatically.
func wireMessages(msgs []ChatMessage, resolve ResolveImage) ([]json.RawMessage, error) {
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
		parts, err := contentParts(m, resolve)
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
