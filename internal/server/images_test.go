package server

import (
	"bytes"
	"context"
	"image"
	"image/color"
	"image/png"
	"io"
	"mime/multipart"
	"net/http"
	"strings"
	"testing"

	"porter/internal/api"
	"porter/internal/client"
)

// tinyPNG encodes a 1x1 PNG, so tests exercise the real byte sniffing
// (http.DetectContentType) instead of a hand-written header.
func tinyPNG(t *testing.T) []byte {
	t.Helper()
	img := image.NewRGBA(image.Rect(0, 0, 1, 1))
	img.Set(0, 0, color.RGBA{R: 255, A: 255})
	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		t.Fatalf("encode png: %v", err)
	}
	return buf.Bytes()
}

// postMultipartImage posts one message the way the web UI's composer does: a
// multipart form with an optional text field and count copies of one PNG.
func postMultipartImage(t *testing.T, srvURL, sessionID, content string, count int) *http.Response {
	t.Helper()
	var body bytes.Buffer
	mw := multipart.NewWriter(&body)
	if content != "" {
		if err := mw.WriteField("content", content); err != nil {
			t.Fatalf("write content field: %v", err)
		}
	}
	for i := 0; i < count; i++ {
		fw, err := mw.CreateFormFile("images", "shot.png")
		if err != nil {
			t.Fatalf("create form file: %v", err)
		}
		if _, err := fw.Write(tinyPNG(t)); err != nil {
			t.Fatalf("write image: %v", err)
		}
	}
	if err := mw.Close(); err != nil {
		t.Fatalf("close writer: %v", err)
	}
	req, err := http.NewRequest(http.MethodPost, srvURL+"/api/sessions/"+sessionID+"/messages", &body)
	if err != nil {
		t.Fatalf("build request: %v", err)
	}
	req.Header.Set("Content-Type", mw.FormDataContentType())
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("POST multipart: %v", err)
	}
	return resp
}

// waitForTurn blocks until the session's next turn finishes, so the user
// message it carried is committed.
func waitForTurn(t *testing.T, c *client.Client, id string, fromSeq uint64) {
	t.Helper()
	var done bool
	if err := c.Subscribe(context.Background(), id, fromSeq, nil, func(env api.Envelope) bool {
		if env.Kind == api.KindTurnDone {
			done = true
			return true
		}
		return false
	}); err != nil {
		t.Fatalf("Subscribe: %v", err)
	}
	if !done {
		t.Fatal("no turn_completed observed")
	}
}

func TestReadImagePartAcceptsPNG(t *testing.T) {
	part, err := readImagePart("shot.png", tinyPNG(t))
	if err != nil {
		t.Fatalf("readImagePart: %v", err)
	}
	if !strings.HasPrefix(part.URL, "data:image/png;base64,") {
		t.Errorf("URL = %q, want a png data URL", part.URL)
	}
	if part.Name != "shot.png" {
		t.Errorf("Name = %q, want shot.png", part.Name)
	}
}

// TestReadImagePartRejectsNonImage covers the rename trick: the media type
// comes from the bytes, so a text file called .png is refused.
func TestReadImagePartRejectsNonImage(t *testing.T) {
	if _, err := readImagePart("notreally.png", []byte("plain text, not a picture")); err == nil {
		t.Fatal("readImagePart accepted a text file named .png")
	}
}

func TestReadImagePartRejectsOversize(t *testing.T) {
	big := make([]byte, maxImageBytes+1)
	copy(big, tinyPNG(t))
	if _, err := readImagePart("big.png", big); err == nil {
		t.Fatal("readImagePart accepted an image over the cap")
	}
}

func TestReadImagePartRejectsEmpty(t *testing.T) {
	if _, err := readImagePart("empty.png", nil); err == nil {
		t.Fatal("readImagePart accepted an empty file")
	}
}

// TestAppendMultipartImage is the whole path: a multipart post from the web
// UI's composer, with one image, lands in history as an image attached to the
// user message, with the bytes inline as a data URL.
func TestAppendMultipartImage(t *testing.T) {
	srv := newTestServer(t, plainLLM())
	c := client.New(srv.URL)
	ctx := context.Background()
	info, err := c.Create(ctx)
	if err != nil {
		t.Fatalf("Create: %v", err)
	}
	resp := postMultipartImage(t, srv.URL, info.ID, "look at this", 1)
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusAccepted {
		t.Fatalf("multipart append status = %d, want 202", resp.StatusCode)
	}
	waitForTurn(t, c, info.ID, info.Seq)

	h, err := c.History(ctx, info.ID)
	if err != nil {
		t.Fatalf("History: %v", err)
	}
	if len(h.History) == 0 {
		t.Fatal("history is empty")
	}
	first := h.History[0]
	if first.Content != "look at this" {
		t.Errorf("content = %q, want %q", first.Content, "look at this")
	}
	if len(first.Images) != 1 {
		t.Fatalf("images = %d, want 1", len(first.Images))
	}
	if !strings.HasPrefix(first.Images[0].URL, "data:image/png;base64,") {
		t.Errorf("image URL = %q, want a png data URL", first.Images[0].URL)
	}
}

// TestAppendMultipartImageOnlyAccepted checks that an image with no text is a
// valid message.
func TestAppendMultipartImageOnlyAccepted(t *testing.T) {
	srv := newTestServer(t, plainLLM())
	c := client.New(srv.URL)
	info, err := c.Create(context.Background())
	if err != nil {
		t.Fatalf("Create: %v", err)
	}
	resp := postMultipartImage(t, srv.URL, info.ID, "", 1)
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusAccepted {
		t.Errorf("image-only append status = %d, want 202", resp.StatusCode)
	}
}

// TestAppendMultipartTooManyImages checks that the per-message cap is enforced
// on the server, not only in the browser.
func TestAppendMultipartTooManyImages(t *testing.T) {
	srv := newTestServer(t, plainLLM())
	c := client.New(srv.URL)
	info, err := c.Create(context.Background())
	if err != nil {
		t.Fatalf("Create: %v", err)
	}
	resp := postMultipartImage(t, srv.URL, info.ID, "too many", maxImagesPerMessage+1)
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusBadRequest {
		t.Errorf("over-limit append status = %d, want 400", resp.StatusCode)
	}
}

// TestViewRendersAttachedImage checks that a committed image renders as an
// <img> with its data URL intact. html/template strips `data:` URLs from src
// attributes by default (replacing them with #ZgotmplZ), so this guards the
// opt-out that makes an attached picture show up in the page.
func TestViewRendersAttachedImage(t *testing.T) {
	srv := newTestServer(t, plainLLM())
	c := client.New(srv.URL)
	info, err := c.Create(context.Background())
	if err != nil {
		t.Fatalf("Create: %v", err)
	}
	resp := postMultipartImage(t, srv.URL, info.ID, "look at this", 1)
	resp.Body.Close()
	if resp.StatusCode != http.StatusAccepted {
		t.Fatalf("multipart append status = %d, want 202", resp.StatusCode)
	}
	waitForTurn(t, c, info.ID, info.Seq)

	view, err := http.Get(srv.URL + "/api/sessions/" + info.ID + "/view")
	if err != nil {
		t.Fatalf("GET view: %v", err)
	}
	defer view.Body.Close()
	body, _ := io.ReadAll(view.Body)
	s := string(body)
	if !strings.Contains(s, `class="msg-image"`) {
		t.Errorf("view does not contain an attached image; got: %s", s)
	}
	if !strings.Contains(s, `src="data:image/png;base64,`) {
		t.Errorf("view does not carry the image's data URL; got: %s", s)
	}
	if strings.Contains(s, "ZgotmplZ") {
		t.Errorf("image src was stripped by the template escaping; got: %s", s)
	}
}
