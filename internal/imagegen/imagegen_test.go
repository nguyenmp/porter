package imagegen

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// imageResponse is the smallest successful reply: one picture, base64.
func imageResponse(payload string) string {
	b := base64.StdEncoding.EncodeToString([]byte(payload))
	return fmt.Sprintf(`{"created":1,"data":[{"b64_json":%q,"media_type":"image/png"}]}`, b)
}

// TestGenerateWithoutInputsPostsJSON checks the text-only shape: JSON to
// /images/generations carrying the model and the prompt.
func TestGenerateWithoutInputsPostsJSON(t *testing.T) {
	var path, contentType string
	var body map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		path = r.URL.Path
		contentType = r.Header.Get("Content-Type")
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Errorf("decode request: %v", err)
		}
		fmt.Fprint(w, imageResponse("png-bytes"))
	}))
	defer srv.Close()

	c := NewClient(srv.URL, "k", "img-model", srv.Client())
	res, err := c.Generate(context.Background(), "a red apple", nil)
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if path != "/images/generations" {
		t.Errorf("path = %q, want /images/generations", path)
	}
	if !strings.HasPrefix(contentType, "application/json") {
		t.Errorf("content type = %q, want JSON", contentType)
	}
	if body["model"] != "img-model" || body["prompt"] != "a red apple" {
		t.Errorf("body = %v, want the model and prompt", body)
	}
	if string(res.Data) != "png-bytes" {
		t.Errorf("data = %q, want png-bytes", res.Data)
	}
	if res.MediaType != "image/png" {
		t.Errorf("media type = %q, want image/png", res.MediaType)
	}
}

// TestGenerateWithInputsPostsMultipart checks the shape that carries pictures:
// a form body to /images/edits with the model, the prompt, and the file. The
// split matters: a gateway that transforms the JSON path silently drops input
// pictures, so the multipart path is the one that works.
func TestGenerateWithInputsPostsMultipart(t *testing.T) {
	var path, contentType, model, prompt, file string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		path = r.URL.Path
		contentType = r.Header.Get("Content-Type")
		if err := r.ParseMultipartForm(1 << 20); err != nil {
			t.Errorf("parse multipart: %v", err)
			return
		}
		model = r.FormValue("model")
		prompt = r.FormValue("prompt")
		f, _, err := r.FormFile("image")
		if err != nil {
			t.Errorf("no image field: %v", err)
			return
		}
		b, _ := io.ReadAll(f)
		f.Close()
		file = string(b)
		fmt.Fprint(w, imageResponse("edited"))
	}))
	defer srv.Close()

	c := NewClient(srv.URL, "k", "img-model", srv.Client())
	inputs := []Input{{Name: "base.png", Data: []byte("base-bytes")}}
	res, err := c.Generate(context.Background(), "make it blue", inputs)
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if path != "/images/edits" {
		t.Errorf("path = %q, want /images/edits", path)
	}
	if !strings.HasPrefix(contentType, "multipart/form-data") {
		t.Errorf("content type = %q, want multipart", contentType)
	}
	if model != "img-model" || prompt != "make it blue" {
		t.Errorf("model/prompt = %q/%q, want the model and prompt", model, prompt)
	}
	if file != "base-bytes" {
		t.Errorf("uploaded file = %q, want base-bytes", file)
	}
	if string(res.Data) != "edited" {
		t.Errorf("data = %q, want edited", res.Data)
	}
}

// TestGenerateReportsEmptyResult checks the shape a gateway returns when the
// model refuses: HTTP 200 with no picture. That must read as a failure, not an
// empty success, or a call would look like it worked and store nothing.
func TestGenerateReportsEmptyResult(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprint(w, `{"created":1,"data":[],"usage":{"input_tokens":10,"output_tokens":0}}`)
	}))
	defer srv.Close()

	c := NewClient(srv.URL, "k", "img-model", srv.Client())
	_, err := c.Generate(context.Background(), "a red apple", nil)
	if err == nil {
		t.Fatal("Generate accepted an empty result")
	}
	if !strings.Contains(err.Error(), "no picture") {
		t.Errorf("error = %q, want it to say no picture came back", err)
	}
}

// TestGenerateReportsProviderError checks that a non-200 status carries the
// provider's own message, so the model and the user see why nothing came back.
func TestGenerateReportsProviderError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusNotFound)
		fmt.Fprint(w, `{"error":{"message":"that model is not available"}}`)
	}))
	defer srv.Close()

	c := NewClient(srv.URL, "k", "img-model", srv.Client())
	_, err := c.Generate(context.Background(), "a red apple", nil)
	if err == nil {
		t.Fatal("Generate accepted a 404")
	}
	if !strings.Contains(err.Error(), "that model is not available") {
		t.Errorf("error = %q, want the provider's message", err)
	}
}

// TestGenerateRejectsEmptyPrompt checks that an empty prompt never reaches the
// network: it is a caller mistake, and the API would bill nothing but confuse.
func TestGenerateRejectsEmptyPrompt(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		t.Error("a request was sent for an empty prompt")
	}))
	defer srv.Close()

	c := NewClient(srv.URL, "k", "img-model", srv.Client())
	if _, err := c.Generate(context.Background(), "   ", nil); err == nil {
		t.Fatal("Generate accepted a blank prompt")
	}
}
