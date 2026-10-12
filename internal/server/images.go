package server

import (
	"fmt"
	"io"
	"net/http"
	"strings"

	"porter/internal/asset"
)

// Image upload limits. An attached image is stored as an asset, but its bytes
// also travel inline in every request that sends the history, so a big image
// adds to each later turn. These caps keep one message from stalling a turn.
const (
	// maxImageBytes caps one attached image, measured before base64 encoding
	// (base64 makes the bytes on the wire about a third larger).
	maxImageBytes = 5 << 20 // 5 MiB
	// maxImagesPerMessage caps how many images one message may attach.
	maxImagesPerMessage = 4
)

// imageUpload is one validated picture from an upload, ready to be published as
// an asset: the safe file name and the raw bytes.
type imageUpload struct {
	Name string
	Data []byte
}

// readImageUpload validates one uploaded file and returns it ready to publish.
//
// The media type comes from the bytes (http.DetectContentType), never from the
// client's claim, so a renamed file cannot pass itself off as an image, and
// only the types the model providers accept are allowed. SVG is deliberately
// not accepted: it is a script-carrying document rather than a picture.
func readImageUpload(name string, data []byte) (imageUpload, error) {
	if len(data) == 0 {
		return imageUpload{}, fmt.Errorf("image %s is empty", name)
	}
	if len(data) > maxImageBytes {
		return imageUpload{}, fmt.Errorf("image %s is %d bytes; the limit is %d bytes (%d MiB)", name, len(data), maxImageBytes, maxImageBytes/(1<<20))
	}
	mime := http.DetectContentType(data)
	if !allowedImageType(mime) {
		return imageUpload{}, fmt.Errorf("image %s looks like %s, which is not an accepted image type (png, jpeg, gif, or webp)", name, mime)
	}
	return imageUpload{Name: asset.SanitizeFilename(name), Data: data}, nil
}

// allowedImageType reports whether the detected media type is one the model
// providers accept as an image input.
func allowedImageType(mime string) bool {
	switch mime {
	case "image/png", "image/jpeg", "image/gif", "image/webp":
		return true
	default:
		return false
	}
}

// readUploadedImages reads the "images" file fields of a parsed multipart form
// and returns one validated upload per file, in the order the browser sent
// them. Every read is bounded, and a file over the cap fails the whole message
// rather than arriving silently cut short.
func readUploadedImages(r *http.Request) ([]imageUpload, error) {
	if r.MultipartForm == nil {
		return nil, nil
	}
	files := r.MultipartForm.File["images"]
	if len(files) > maxImagesPerMessage {
		return nil, fmt.Errorf("too many images: %d attached, the limit is %d", len(files), maxImagesPerMessage)
	}
	uploads := make([]imageUpload, 0, len(files))
	for _, fh := range files {
		f, err := fh.Open()
		if err != nil {
			return nil, fmt.Errorf("read image %s: %w", fh.Filename, err)
		}
		// One byte past the cap, so readImageUpload can tell "exactly at the
		// limit" from "over it" without reading the rest of a huge file.
		data, err := io.ReadAll(io.LimitReader(f, maxImageBytes+1))
		f.Close()
		if err != nil {
			return nil, fmt.Errorf("read image %s: %w", fh.Filename, err)
		}
		up, err := readImageUpload(fh.Filename, data)
		if err != nil {
			return nil, err
		}
		uploads = append(uploads, up)
	}
	return uploads, nil
}

// checkImageURL validates an image sent as JSON rather than as a file upload.
// Two forms are accepted: a hosted asset URL the server already holds, or a
// data URL carrying an allowed image type (bytes inline). Anything else is
// refused so a stray or nonsensical value is never stored and forwarded.
//
// A data URL's bytes are left to the provider to check; an asset URL's segments
// are validated when the bytes are read (Store.AssetDataURL), so only the shape
// is checked here.
func checkImageURL(url string) error {
	if strings.HasPrefix(url, asset.URLPrefix) {
		return nil
	}
	if !strings.HasPrefix(url, "data:") {
		return fmt.Errorf("image url must be a data: URL or a %s URL (the porter server is not reachable from the model provider, so inline bytes are the default)", asset.URLPrefix)
	}
	header, _, ok := strings.Cut(strings.TrimPrefix(url, "data:"), ",")
	if !ok {
		return fmt.Errorf("image data URL is missing its comma")
	}
	mime, encoding, _ := strings.Cut(header, ";")
	if !allowedImageType(mime) {
		return fmt.Errorf("image type %s is not an accepted image type (png, jpeg, gif, or webp)", mime)
	}
	if encoding != "base64" {
		return fmt.Errorf("image data URL must be base64-encoded")
	}
	return nil
}
