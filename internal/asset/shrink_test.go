package asset

import (
	"bytes"
	"errors"
	"image"
	"image/color"
	"image/gif"
	"image/jpeg"
	"image/png"
	"math/rand"
	"testing"
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

// TestFitImageLeavesASmallPictureAlone checks the common case: nothing is
// re-encoded, so nothing is lost.
func TestFitImageLeavesASmallPictureAlone(t *testing.T) {
	small := noisyPNG(t, 40, 40)
	out, err := FitImage(small)
	if err != nil {
		t.Fatalf("FitImage: %v", err)
	}
	if !bytes.Equal(out, small) {
		t.Error("FitImage re-encoded a picture that already fit")
	}
}

// TestFitImageShrinksTheCap is the whole point: a screenshot over the cap
// arrives as a picture under it, same format, still readable.
func TestFitImageShrinksTheCap(t *testing.T) {
	src := noisyPNG(t, 1600, 1600)
	if int64(len(src)) <= ImageMaxBytes {
		t.Fatalf("test image is %s, want it over the %s cap", HumanBytes(int64(len(src))), HumanBytes(ImageMaxBytes))
	}
	out, err := FitImage(src)
	if err != nil {
		t.Fatalf("FitImage: %v", err)
	}
	if int64(len(out)) > ImageMaxBytes {
		t.Fatalf("shrunk to %s, want at most %s", HumanBytes(int64(len(out))), HumanBytes(ImageMaxBytes))
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
	if int64(len(out)) > ImageMaxBytes {
		t.Errorf("shrunk to %s, want at most %s", HumanBytes(int64(len(out))), HumanBytes(ImageMaxBytes))
	}
	if ImageMimeType(out) != "image/jpeg" {
		t.Errorf("shrunk bytes are %q, want a jpeg", ImageMimeType(out))
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

// TestFitImageRefusesAnimatedOverCap checks the refusal a GIF or WebP over the
// cap gets: re-encoding would keep only the first frame, so it is reported
// rather than flattened.
func TestFitImageRefusesAnimatedOverCap(t *testing.T) {
	var buf bytes.Buffer
	if err := gif.Encode(&buf, image.NewPaletted(image.Rect(0, 0, 4, 4), color.Palette{color.Black}), nil); err != nil {
		t.Fatalf("encode gif: %v", err)
	}
	big := append(buf.Bytes(), make([]byte, ImageMaxBytes+1)...)
	if _, err := FitImage(big); !errors.Is(err, ErrFramesOverCap) {
		t.Fatalf("FitImage err = %v, want ErrFramesOverCap", err)
	}
}

// TestHumanBytes checks the wording the user sees: units, no long decimals,
// and a whole number when the value is whole.
func TestHumanBytes(t *testing.T) {
	cases := []struct {
		in   int64
		want string
	}{
		{0, "0 B"},
		{999, "999 B"},
		{1024, "1 KiB"},
		{1536, "1.5 KiB"},
		{5 << 20, "5 MiB"},
		{9437184, "9 MiB"},
		{10 << 20, "10 MiB"},
		{3670016, "3.5 MiB"},
		{13002342, "12 MiB"},
		{1 << 30, "1 GiB"},
	}
	for _, c := range cases {
		if got := HumanBytes(c.in); got != c.want {
			t.Errorf("HumanBytes(%d) = %q, want %q", c.in, got, c.want)
		}
	}
}
