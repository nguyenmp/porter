package server

import (
	"encoding/base64"
	"fmt"
	"io"
	"net/http"
	"strings"

	"porter/internal/asset"
	"porter/internal/llm"
)

// Image upload limits. A data URL carries the picture's bytes inline, so a big
// image adds to every later request that sends the history. These caps keep one
// message from stalling a turn.
const (
	// maxImageBytes caps one attached image, measured before base64 encoding
	// (base64 makes the bytes on the wire about a third larger).
	maxImageBytes = 5 << 20 // 5 MiB
	// maxImagesPerMessage caps how many images one message may attach.
	maxImagesPerMessage = 4
)

// readImagePart turns one uploaded file into an ImagePart: the bytes inline as
// a data URL, plus the file's name for display.
//
// The media type comes from the bytes (http.DetectContentType), never from the
// client's claim, so a renamed file cannot pass itself off as an image, and
// only the types the model providers accept are allowed. SVG is deliberately
// not accepted: it is a script-carrying document rather than a picture.
func readImagePart(name string, data []byte) (llm.ImagePart, error) {
	if len(data) == 0 {
		return llm.ImagePart{}, fmt.Errorf("image %s is empty", name)
	}
	if len(data) > maxImageBytes {
		return llm.ImagePart{}, fmt.Errorf("image %s is %d bytes; the limit is %d bytes (%d MiB)", name, len(data), maxImageBytes, maxImageBytes/(1<<20))
	}
	mime := http.DetectContentType(data)
	if !allowedImageType(mime) {
		return llm.ImagePart{}, fmt.Errorf("image %s looks like %s, which is not an accepted image type (png, jpeg, gif, or webp)", name, mime)
	}
	return llm.ImagePart{
		Name: asset.SanitizeFilename(name),
		URL:  "data:" + mime + ";base64," + base64.StdEncoding.EncodeToString(data),
	}, nil
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
// and returns one ImagePart per file, in the order the browser sent them. Every
// read is bounded, and a file over the cap fails the whole message rather than
// arriving silently cut short.
func readUploadedImages(r *http.Request) ([]llm.ImagePart, error) {
	if r.MultipartForm == nil {
		return nil, nil
	}
	files := r.MultipartForm.File["images"]
	if len(files) > maxImagesPerMessage {
		return nil, fmt.Errorf("too many images: %d attached, the limit is %d", len(files), maxImagesPerMessage)
	}
	images := make([]llm.ImagePart, 0, len(files))
	for _, fh := range files {
		f, err := fh.Open()
		if err != nil {
			return nil, fmt.Errorf("read image %s: %w", fh.Filename, err)
		}
		// One byte past the cap, so readImagePart can tell "exactly at the
		// limit" from "over it" without reading the rest of a huge file.
		data, err := io.ReadAll(io.LimitReader(f, maxImageBytes+1))
		f.Close()
		if err != nil {
			return nil, fmt.Errorf("read image %s: %w", fh.Filename, err)
		}
		img, err := readImagePart(fh.Filename, data)
		if err != nil {
			return nil, err
		}
		images = append(images, img)
	}
	return images, nil
}

// checkImageDataURL validates an image sent as JSON rather than as a file
// upload: it must be a data URL carrying an allowed image type. The bytes
// themselves are left to the provider to check; this only keeps a stray or
// nonsensical value from being stored and forwarded.
func checkImageDataURL(url string) error {
	if !strings.HasPrefix(url, "data:") {
		return fmt.Errorf("image url must be a data: URL (the bytes travel inline; the porter server is not reachable from the model provider)")
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
