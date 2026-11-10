package server

import (
	"bytes"
	"context"
	"image"
	"image/color"
	gifenc "image/gif"
	"image/png"
	"io"
	"mime/multipart"
	"net/http"
	"strings"
	"testing"

	"porter/internal/api"
	"porter/internal/asset"
	"porter/internal/client"
	mdr "porter/internal/render"
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
	png := tinyPNG(t)
	images := make([][]byte, 0, count)
	for i := 0; i < count; i++ {
		images = append(images, png)
	}
	return postMultipartImages(t, srvURL, sessionID, content, images)
}

// postMultipartImages posts one message carrying the given image bytes, one
// "images" file part each, so a test can choose the size of what it sends.
func postMultipartImages(t *testing.T, srvURL, sessionID, content string, images [][]byte) *http.Response {
	t.Helper()
	var body bytes.Buffer
	mw := multipart.NewWriter(&body)
	if content != "" {
		if err := mw.WriteField("content", content); err != nil {
			t.Fatalf("write content field: %v", err)
		}
	}
	for _, img := range images {
		fw, err := mw.CreateFormFile("images", "shot.png")
		if err != nil {
			t.Fatalf("create form file: %v", err)
		}
		if _, err := fw.Write(img); err != nil {
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

func TestReadImageUploadAcceptsPNG(t *testing.T) {
	up, err := readImageUpload("shot.png", tinyPNG(t))
	if err != nil {
		t.Fatalf("readImageUpload: %v", err)
	}
	if up.Name != "shot.png" {
		t.Errorf("Name = %q, want shot.png", up.Name)
	}
	if len(up.Data) == 0 {
		t.Error("Data is empty, want the file's bytes")
	}
}

func TestReadImageUploadRejectsNonImage(t *testing.T) {
	if _, err := readImageUpload("notreally.png", []byte("plain text, not a picture")); err == nil {
		t.Fatal("readImageUpload accepted a text file named .png")
	}
}

// TestReadImageUploadShrinksOversize checks that a PNG over the cap still goes:
// the bytes handed on are a re-encoded copy under the cap, not a refusal.
func TestReadImageUploadShrinksOversize(t *testing.T) {
	big := make([]byte, maxImageBytes+1)
	copy(big, tinyPNG(t))
	up, err := readImageUpload("big.png", big)
	if err != nil {
		t.Fatalf("readImageUpload refused an image it could shrink: %v", err)
	}
	if int64(len(up.Data)) > maxImageBytes {
		t.Errorf("shrunk to %s, want at most %s", asset.HumanBytes(int64(len(up.Data))), asset.HumanBytes(maxImageBytes))
	}
	if asset.ImageMimeType(up.Data) != "image/png" {
		t.Errorf("shrunk bytes are %q, want a png", asset.ImageMimeType(up.Data))
	}
}

// TestOversizeErrorNamesSizeAndLimit checks the refusal a user reads in the
// toast: the file, its size, and the cap, with no byte counts.
func TestOversizeErrorNamesSizeAndLimit(t *testing.T) {
	got := oversizeError("shot.png", 12<<20).Error()
	want := "shot.png is 12 MiB; the limit is 5 MiB per image"
	if got != want {
		t.Errorf("oversizeError = %q, want %q", got, want)
	}
}

// TestAppendOversizeImageSaysWhy posts an image the server cannot shrink over
// the cap, the way the composer does, and checks the 400 body names the file,
// the limit, and what to do instead. The web UI shows that body in a toast, so
// it has to read as a sentence.
func TestAppendOversizeImageSaysWhy(t *testing.T) {
	srv := newTestServer(t, plainLLM())
	c := client.New(srv.URL)
	info, err := c.Create(context.Background())
	if err != nil {
		t.Fatalf("Create: %v", err)
	}
	// A padded GIF: a format that may be animated, so the server refuses it
	// rather than flattening it to one frame.
	var gif bytes.Buffer
	if err := gifenc.Encode(&gif, image.NewPaletted(image.Rect(0, 0, 4, 4), color.Palette{color.Black}), nil); err != nil {
		t.Fatalf("encode gif: %v", err)
	}
	big := append(gif.Bytes(), make([]byte, 6<<20)...)
	resp := postMultipartImages(t, srv.URL, info.ID, "look at this", [][]byte{big})
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("oversize append status = %d, want 400", resp.StatusCode)
	}
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("read body: %v", err)
	}
	text := string(body)
	for _, want := range []string{"shot.png", "6 MiB", "limit is 5 MiB", "attach a PNG or JPEG"} {
		if !strings.Contains(text, want) {
			t.Errorf("400 body = %q, want it to contain %q", text, want)
		}
	}
}

func TestReadImageUploadRejectsEmpty(t *testing.T) {
	if _, err := readImageUpload("empty.png", nil); err == nil {
		t.Fatal("readImageUpload accepted an empty file")
	}
}

// TestAppendMultipartImage is the whole path: a multipart post from the web
// UI's composer, with one image, lands in history as an image attached to the
// user message, naming a published asset rather than carrying the bytes.
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
	if !strings.HasPrefix(first.Images[0].URL, "/assets/sessions/") {
		t.Errorf("image URL = %q, want a published asset URL", first.Images[0].URL)
	}
	if first.Images[0].Name != "shot.png" {
		t.Errorf("image name = %q, want shot.png", first.Images[0].Name)
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
// <img> pointing at its published asset URL.
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
	if !strings.Contains(s, `src="/assets/sessions/`) {
		t.Errorf("view does not point at the image's published asset; got: %s", s)
	}
	if strings.Contains(s, "ZgotmplZ") {
		t.Errorf("image src was stripped by the template escaping; got: %s", s)
	}
}

// TestAppendImageWithoutAssetStore checks that a server with no asset store
// refuses an upload with a clear reason, instead of accepting the message and
// dropping the picture. An in-memory database has no directory, so it has no
// asset root (see localAssetRoot).
func TestAppendImageWithoutAssetStore(t *testing.T) {
	_, srv := startServerDB(t, ":memory:", plainLLM())
	c := client.New(srv.URL)
	info, err := c.Create(context.Background())
	if err != nil {
		t.Fatalf("Create: %v", err)
	}
	resp := postMultipartImage(t, srv.URL, info.ID, "look at this", 1)
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("upload without an asset store: status = %d, want 400", resp.StatusCode)
	}
	body, _ := io.ReadAll(resp.Body)
	if !strings.Contains(string(body), "assets are not configured") {
		t.Errorf("refusal does not explain the missing asset store; got: %s", body)
	}
}

// TestReplyImageIsWidthBounded checks that an image in an assistant reply stays
// inside the message column. The markdown renderer gives a reply image no
// class, so the page must bound every <img> under .msg-content. Without that
// rule the browser draws a wide image at its native pixel size, and the image
// runs off the edge of the screen.
func TestReplyImageIsWidthBounded(t *testing.T) {
	// First, confirm the rule's assumption: a reply image is a bare <img>, with
	// no class for a class-based rule to target.
	html := mdr.Markdown("![shot](/assets/sessions/s/shot.png)")
	if !strings.Contains(html, "<img") {
		t.Fatalf("markdown image did not render as an <img>; got: %s", html)
	}
	if strings.Contains(html, "class=") {
		t.Fatalf("reply image gained a class, so a class-based rule could target it; got: %s", html)
	}

	// Then, confirm the page ships the rule that bounds it.
	srv := newTestServer(t, plainLLM())
	resp, err := http.Get(srv.URL + "/")
	if err != nil {
		t.Fatalf("GET /: %v", err)
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	if !strings.Contains(string(body), ".msg-content img { max-width: 100%; height: auto; max-height: 360px; }") {
		t.Errorf("chat page does not bound reply images to the message column; got: %s", body)
	}
}
