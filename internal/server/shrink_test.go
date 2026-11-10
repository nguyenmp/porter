package server

import (
	"bytes"
	"image"
	"image/color"
	"image/jpeg"
	"image/png"
	"math/rand"
	"testing"

	"porter/internal/asset"
)

// noisyPNG encodes a w×h picture of random colours, which compresses about as
// badly as a real screenshot: a big one is reliably over the byte cap.
func noisyPNG(t *testing.T, w, h int) []byte {
	t.Helper()
	img := image.NewRGBA(image.Rect(0, 0, w, h))
	rnd := rand.New(rand.NewSource(1))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			img.Set(x, y, color.RGBA{R: uint8(rnd.Intn(256)), G: uint8(rnd.Intn(256)), B: uint8(rnd.Intn(256)), A: 255})
		}
	}
	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		t.Fatalf("encode png: %v", err)
	}
	return buf.Bytes()
}

// TestShrinkImageFitsTheCap is the whole point of shrinking: a screenshot over
// the cap arrives as a picture under it, same format, still readable.
func TestShrinkImageFitsTheCap(t *testing.T) {
	src := noisyPNG(t, 1600, 1600)
	if int64(len(src)) <= maxImageBytes {
		t.Fatalf("test image is %s, want it over the %s cap", humanBytes(int64(len(src))), humanBytes(maxImageBytes))
	}
	out, err := shrinkImage(src)
	if err != nil {
		t.Fatalf("shrinkImage: %v", err)
	}
	if int64(len(out)) > maxImageBytes {
		t.Fatalf("shrunk to %s, want at most %s", humanBytes(int64(len(out))), humanBytes(maxImageBytes))
	}
	img, format, err := image.Decode(bytes.NewReader(out))
	if err != nil {
		t.Fatalf("decode shrunk picture: %v", err)
	}
	if format != "png" {
		t.Errorf("format = %q, want png", format)
	}
	if b := img.Bounds(); b.Dx() < 200 || b.Dy() < 200 {
		t.Errorf("shrunk to %dx%d, want a picture that still reads", b.Dx(), b.Dy())
	}
}

// TestShrinkImageCapsTheSide checks the provider's pixel limit: a picture
// wider than 8000 pixels comes back at 8000, keeping its aspect ratio.
func TestShrinkImageCapsTheSide(t *testing.T) {
	img := image.NewRGBA(image.Rect(0, 0, 9000, 100))
	for x := 0; x < 9000; x++ {
		img.Set(x, 50, color.RGBA{R: uint8(x % 256), A: 255})
	}
	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		t.Fatalf("encode png: %v", err)
	}
	out, err := shrinkImage(buf.Bytes())
	if err != nil {
		t.Fatalf("shrinkImage: %v", err)
	}
	got, _, err := image.Decode(bytes.NewReader(out))
	if err != nil {
		t.Fatalf("decode shrunk picture: %v", err)
	}
	b := got.Bounds()
	if b.Dx() != maxImageSide {
		t.Errorf("width = %d, want %d", b.Dx(), maxImageSide)
	}
	if b.Dy() < 88 || b.Dy() > 90 {
		t.Errorf("height = %d, want about 89 (the aspect ratio kept)", b.Dy())
	}
}

// TestShrinkImageKeepsJPEG checks that a JPEG stays a JPEG, so the asset keeps
// its type and the browser renders it.
func TestShrinkImageKeepsJPEG(t *testing.T) {
	src := noisyPNG(t, 1600, 1600)
	img, err := png.Decode(bytes.NewReader(src))
	if err != nil {
		t.Fatalf("decode png: %v", err)
	}
	var buf bytes.Buffer
	if err := jpeg.Encode(&buf, img, &jpeg.Options{Quality: 100}); err != nil {
		t.Fatalf("encode jpeg: %v", err)
	}
	out, err := shrinkImage(buf.Bytes())
	if err != nil {
		t.Fatalf("shrinkImage: %v", err)
	}
	if int64(len(out)) > maxImageBytes {
		t.Errorf("shrunk to %s, want at most %s", humanBytes(int64(len(out))), humanBytes(maxImageBytes))
	}
	if asset.ImageMimeType(out) != "image/jpeg" {
		t.Errorf("shrunk bytes are %q, want a jpeg", asset.ImageMimeType(out))
	}
}

// TestShrinkImageRefusesHugePixels checks the guard against a small file that
// claims an enormous picture: it is refused before it is decoded, so a bomb
// cannot cost the server hundreds of megabytes.
func TestShrinkImageRefusesHugePixels(t *testing.T) {
	img := image.NewGray(image.Rect(0, 0, 9000, 9000)) // 81 megapixels, over the cap
	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		t.Fatalf("encode png: %v", err)
	}
	if _, err := shrinkImage(buf.Bytes()); err == nil {
		t.Fatal("shrinkImage decoded a picture over the pixel cap")
	}
}
