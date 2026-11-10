package server

import (
	"errors"
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
	// maxImageBytes is the size that reaches the model. It lives in the asset
	// package, because read_image meets the same cap.
	maxImageBytes = asset.ImageMaxBytes
	// maxUploadBytes caps what the composer may send at all. A PNG or JPEG
	// between the two caps is shrunk before it is stored, so a retina
	// screenshot goes through instead of being refused.
	maxUploadBytes = 25 << 20 // 25 MiB
	// maxImagesPerMessage caps how many images one message may attach.
	maxImagesPerMessage = 4
)

// oversizeError explains a refused image the way a reader needs it: the file,
// how big it is, and the cap. The composer shows this text in a toast, so it
// reads as a sentence rather than a pair of byte counts.
func oversizeError(name string, size int64) error {
	return fmt.Errorf("%s is %s; the limit is %s per image", name, asset.HumanBytes(size), asset.HumanBytes(maxImageBytes))
}

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
//
// A PNG or JPEG over the byte cap is shrunk to fit, because the shortest path
// from a retina screenshot to a sent message should not be "crop it yourself".
// A GIF or WebP over the cap is refused: it may be animated, and a silent
// flattening is worse than a clear refusal.
func readImageUpload(name string, data []byte) (imageUpload, error) {
	if len(data) == 0 {
		return imageUpload{}, fmt.Errorf("image %s is empty", name)
	}
	mime := asset.ImageMimeType(data)
	if mime == "" {
		return imageUpload{}, fmt.Errorf("image %s is not an accepted image type (png, jpeg, gif, or webp)", name)
	}
	if int64(len(data)) > maxImageBytes {
		size := asset.HumanBytes(int64(len(data)))
		fitted, err := asset.FitImage(data)
		switch {
		case errors.Is(err, asset.ErrFramesOverCap):
			kind := strings.TrimPrefix(mime, "image/")
			return imageUpload{}, fmt.Errorf("%s is %s; the limit is %s per image, and a %s cannot be shrunk here — attach a PNG or JPEG", name, size, asset.HumanBytes(maxImageBytes), kind)
		case err != nil:
			return imageUpload{}, fmt.Errorf("%s is %s and could not be shrunk to the %s limit: %v", name, size, asset.HumanBytes(maxImageBytes), err)
		}
		data = fitted
	}
	return imageUpload{Name: asset.SanitizeFilename(name), Data: data}, nil
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
		return nil, fmt.Errorf("one message can carry %d images; this one has %d", maxImagesPerMessage, len(files))
	}
	uploads := make([]imageUpload, 0, len(files))
	for _, fh := range files {
		// fh.Size is the whole file, where the bounded read below stops one
		// byte past the cap, so the refusal names the size the user sees. The
		// upload cap is the larger one: a file between the two is read whole so
		// readImageUpload can shrink it.
		if fh.Size > maxUploadBytes {
			return nil, oversizeError(fh.Filename, fh.Size)
		}
		f, err := fh.Open()
		if err != nil {
			return nil, fmt.Errorf("read image %s: %w", fh.Filename, err)
		}
		// One byte past the cap, so readImageUpload can tell "exactly at the
		// limit" from "over it" without reading the rest of a huge file.
		data, err := io.ReadAll(io.LimitReader(f, maxUploadBytes+1))
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
	if !asset.IsImageMimeType(mime) {
		return fmt.Errorf("image type %s is not an accepted image type (png, jpeg, gif, or webp)", mime)
	}
	if encoding != "base64" {
		return fmt.Errorf("image data URL must be base64-encoded")
	}
	return nil
}
