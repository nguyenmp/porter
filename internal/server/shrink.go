package server

import (
	"bytes"
	"errors"
	"fmt"
	"image"
	"image/jpeg"
	"image/png"
	"math"

	"golang.org/x/image/draw"
)

// The two shape limits a shrunk picture has to meet. The byte cap is
// maxImageBytes; these come from the model providers, which refuse a picture
// longer than 8000 pixels on a side and are no use for one this large anyway.
const (
	maxImageSide   = 8000
	maxImagePixels = 40 << 20
)

// shrinkable reports whether a picture over the byte cap can be re-encoded
// smaller. PNG and JPEG hold one frame, so re-encoding loses nothing. A GIF or
// WebP may be animated, and re-encoding would keep only the first frame, so
// those are refused instead of quietly flattened.
func shrinkable(mime string) bool {
	return mime == "image/png" || mime == "image/jpeg"
}

// errShrinkFailed is the reason a picture stays over the cap after every pass.
// The caller puts it in the message the user reads.
var errShrinkFailed = errors.New("it stays over the limit even after shrinking")

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
		if len(out) <= maxImageBytes {
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
