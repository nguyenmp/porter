package asset

import (
	"bytes"
	"errors"
	"fmt"
	"image"
	"image/jpeg"
	"image/png"
	"math"
	"strings"

	"golang.org/x/image/draw"
)

// ImageMaxBytes caps one picture sent to the model, measured before base64
// encoding (base64 makes the bytes on the wire about a third larger). Every
// provider caps an image near 5 MB, and a picture over the cap fails the whole
// request, so this is the size a picture has to meet before it goes out — as a
// composer upload or through read_image.
const ImageMaxBytes = 5 << 20

// The shape limits a shrunk picture also has to meet. They come from the
// model providers, which refuse a picture longer than 8000 pixels on a side.
const (
	maxImageSide   = 8000
	maxImagePixels = 40 << 20
)

// ErrFramesOverCap explains a picture over the byte cap whose format cannot be
// re-encoded. PNG and JPEG hold one frame, so re-encoding them loses nothing;
// a GIF or WebP may be animated, and re-encoding would keep only the first
// frame, so those are refused instead of quietly flattened.
var ErrFramesOverCap = errors.New("its format can hold more than one frame, so shrinking it would drop the rest")

// ErrShrinkFailed is the reason a picture stays over the cap after every pass.
var errShrinkFailed = errors.New("it stays over the limit even after shrinking")

// Shrinkable reports whether a picture over the byte cap can be re-encoded
// smaller, which depends on the format rather than the file.
func Shrinkable(mime string) bool {
	return mime == "image/png" || mime == "image/jpeg"
}

// FitImage returns a picture unchanged when it already fits the model's cap,
// and a smaller re-encoded copy when it does not. The error says why a picture
// cannot be shrunk, so the caller can put that reason in a message.
//
// Callers that accept uploads check the media type first (ImageMimeType), so a
// file that is not a picture is refused with the type it claims, not here.
func FitImage(data []byte) ([]byte, error) {
	if len(data) <= ImageMaxBytes {
		return data, nil
	}
	mime := ImageMimeType(data)
	if mime == "" {
		return nil, errors.New("it is not a png, jpeg, gif, or webp picture")
	}
	if !Shrinkable(mime) {
		return nil, ErrFramesOverCap
	}
	return shrinkImage(data)
}

// shrinkImage re-encodes a picture over the byte cap so it fits. It keeps the
// format (PNG stays PNG, JPEG stays JPEG at quality 90) and steps the size down
// until the bytes fit, starting from the provider's pixel limit. A 12 MiB
// screenshot lands as a copy of a few MiB that still reads clearly.
//
// The pixel count is checked before the picture is decoded: a small file can
// claim to be enormous, and decoding one would cost far more memory than the
// upload itself.
func shrinkImage(data []byte) ([]byte, error) {
	cfg, format, err := image.DecodeConfig(bytes.NewReader(data))
	if err != nil {
		return nil, err
	}
	if cfg.Width <= 0 || cfg.Height <= 0 {
		return nil, errors.New("it reports no size")
	}
	if cfg.Width*cfg.Height > maxImagePixels {
		return nil, fmt.Errorf("it is %d×%d pixels, too large to shrink", cfg.Width, cfg.Height)
	}
	src, _, err := image.Decode(bytes.NewReader(data))
	if err != nil {
		return nil, err
	}
	long := cfg.Width
	if cfg.Height > long {
		long = cfg.Height
	}
	scale := 1.0
	if long > maxImageSide {
		scale = float64(maxImageSide) / float64(long)
	}
	for i := 0; i < 10; i++ {
		img := src
		if scale < 1 {
			img = scaled(src, scale)
		}
		out, err := encodeImage(img, format)
		if err != nil {
			return nil, err
		}
		if len(out) <= ImageMaxBytes {
			return out, nil
		}
		b := img.Bounds()
		if b.Dx() <= 64 || b.Dy() <= 64 {
			break
		}
		scale *= 0.7
	}
	return nil, errShrinkFailed
}

// scaled returns src resized by scale, keeping the picture's own aspect ratio.
func scaled(src image.Image, scale float64) image.Image {
	b := src.Bounds()
	w := int(math.Round(float64(b.Dx()) * scale))
	h := int(math.Round(float64(b.Dy()) * scale))
	if w < 1 {
		w = 1
	}
	if h < 1 {
		h = 1
	}
	dst := image.NewRGBA(image.Rect(0, 0, w, h))
	draw.CatmullRom.Scale(dst, dst.Bounds(), src, b, draw.Src, nil)
	return dst
}

// encodeImage writes img back out in the format it arrived in.
func encodeImage(img image.Image, format string) ([]byte, error) {
	var buf bytes.Buffer
	switch format {
	case "png":
		if err := png.Encode(&buf, img); err != nil {
			return nil, err
		}
	case "jpeg":
		if err := jpeg.Encode(&buf, img, &jpeg.Options{Quality: 90}); err != nil {
			return nil, err
		}
	default:
		return nil, fmt.Errorf("cannot re-encode a %s picture", format)
	}
	return buf.Bytes(), nil
}

// HumanBytes formats a byte count for a message a person reads: "512 KiB",
// "1.5 MiB", "12 MiB". Above a kibibyte it keeps one decimal place below 10,
// which is the precision that matters when someone is picking a smaller file.
func HumanBytes(n int64) string {
	const step = 1024
	if n < step {
		return fmt.Sprintf("%d B", n)
	}
	units := []string{"KiB", "MiB", "GiB", "TiB"}
	v := float64(n)
	i := -1
	for v >= step && i < len(units)-1 {
		v /= step
		i++
	}
	if v >= 10 || v == float64(int64(v)) {
		return fmt.Sprintf("%.0f %s", v, units[i])
	}
	// One decimal below 10, with a trailing ".0" dropped: "1.5 MiB", "5 MiB".
	return strings.TrimSuffix(fmt.Sprintf("%.1f", v), ".0") + " " + units[i]
}
