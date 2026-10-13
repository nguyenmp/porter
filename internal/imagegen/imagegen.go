// Package imagegen calls an image model: it makes a picture from a prompt and,
// when given input pictures, guides or edits one with them. It speaks the
// OpenAI image shape, which most gateways and providers accept.
//
// Two request shapes are needed, because that API splits them: a prompt alone
// is JSON posted to /images/generations, while input pictures go as a
// multipart form to /images/edits. The split is not cosmetic — a gateway built
// on this shape (LiteLLM in front of OpenRouter, for one) only carries input
// pictures on the multipart path, and silently drops them on the other.
package imagegen

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"strings"
)

// Input is one picture sent to the model along with the prompt. Name seeds the
// uploaded file name; the model mostly ignores it, but a stable name keeps a
// gateway's logs readable.
type Input struct {
	Name string
	Data []byte
}

// Result is the picture the model returned.
type Result struct {
	Data []byte
	// MediaType is the picture's type when the provider reports it, and ""
	// when it does not (many gateways leave it out). Callers name the stored
	// picture from this when it is set.
	MediaType string
}

// Client calls an image model at one endpoint.
type Client struct {
	BaseURL string
	APIKey  string
	Model   string
	HTTP    *http.Client
}

// NewClient returns a client for the given endpoint. A nil hc means
// http.DefaultClient.
func NewClient(baseURL, apiKey, model string, hc *http.Client) *Client {
	if hc == nil {
		hc = http.DefaultClient
	}
	return &Client{BaseURL: strings.TrimSuffix(baseURL, "/"), APIKey: apiKey, Model: model, HTTP: hc}
}

// Generate makes a picture from prompt and, when inputs are given, from those
// pictures as references.
func (c *Client) Generate(ctx context.Context, prompt string, inputs []Input) (Result, error) {
	if strings.TrimSpace(prompt) == "" {
		return Result{}, errors.New("image prompt is empty")
	}
	if len(inputs) == 0 {
		return c.generateJSON(ctx, prompt)
	}
	return c.editMultipart(ctx, prompt, inputs)
}

// generateJSON posts a text-only request. It is the shape that carries a prompt
// and nothing else.
func (c *Client) generateJSON(ctx context.Context, prompt string) (Result, error) {
	body, err := json.Marshal(struct {
		Model  string `json:"model"`
		Prompt string `json:"prompt"`
		N      int    `json:"n"`
	}{Model: c.Model, Prompt: prompt, N: 1})
	if err != nil {
		return Result{}, fmt.Errorf("encode image request: %w", err)
	}
	resp, err := c.do(ctx, "/images/generations", "application/json", body)
	if err != nil {
		return Result{}, err
	}
	defer resp.Close()
	return decode(resp)
}

// editMultipart posts a request that carries input pictures. It is the shape
// that carries them; see the package comment.
func (c *Client) editMultipart(ctx context.Context, prompt string, inputs []Input) (Result, error) {
	var buf bytes.Buffer
	mw := multipart.NewWriter(&buf)
	if err := mw.WriteField("model", c.Model); err != nil {
		return Result{}, fmt.Errorf("encode image request: %w", err)
	}
	if err := mw.WriteField("prompt", prompt); err != nil {
		return Result{}, fmt.Errorf("encode image request: %w", err)
	}
	for i, in := range inputs {
		name := in.Name
		if name == "" {
			name = fmt.Sprintf("input-%d.png", i+1)
		}
		fw, err := mw.CreateFormFile("image", name)
		if err != nil {
			return Result{}, fmt.Errorf("encode image request: %w", err)
		}
		if _, err := fw.Write(in.Data); err != nil {
			return Result{}, fmt.Errorf("encode image request: %w", err)
		}
	}
	if err := mw.Close(); err != nil {
		return Result{}, fmt.Errorf("encode image request: %w", err)
	}
	resp, err := c.do(ctx, "/images/edits", mw.FormDataContentType(), buf.Bytes())
	if err != nil {
		return Result{}, err
	}
	defer resp.Close()
	return decode(resp)
}

// do sends one request and returns the response body of a successful call. A
// non-200 status is an error carrying the provider's own message, so the model
// (and the user) sees why a picture did not come back.
func (c *Client) do(ctx context.Context, path, contentType string, body []byte) (io.ReadCloser, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.BaseURL+path, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("build image request: %w", err)
	}
	req.Header.Set("Content-Type", contentType)
	if c.APIKey != "" {
		req.Header.Set("Authorization", "Bearer "+c.APIKey)
	}
	resp, err := c.HTTP.Do(req)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	if resp.StatusCode != http.StatusOK {
		msg, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		resp.Body.Close()
		return nil, fmt.Errorf("%s: unexpected status %s: %s", path, resp.Status, strings.TrimSpace(string(msg)))
	}
	return resp.Body, nil
}

// decode reads one image out of a response.
//
// A 200 with an empty data list is treated as a failure, not an empty success:
// some gateways answer that way when the model refuses the prompt (asked for a
// picture it will not make, or for something that does not match the rules).
// Reporting it beats storing nothing and saying the call worked.
func decode(body io.Reader) (Result, error) {
	var out struct {
		Data []struct {
			B64JSON   string `json:"b64_json"`
			MediaType string `json:"media_type"`
		} `json:"data"`
		Error *struct {
			Message string `json:"message"`
		} `json:"error"`
	}
	if err := json.NewDecoder(body).Decode(&out); err != nil {
		return Result{}, fmt.Errorf("decode image response: %w", err)
	}
	if out.Error != nil && out.Error.Message != "" {
		return Result{}, errors.New(out.Error.Message)
	}
	if len(out.Data) == 0 || out.Data[0].B64JSON == "" {
		return Result{}, errors.New("the image model returned no picture (it may have refused the prompt)")
	}
	data, err := base64.StdEncoding.DecodeString(out.Data[0].B64JSON)
	if err != nil {
		return Result{}, fmt.Errorf("decode image bytes: %w", err)
	}
	return Result{Data: data, MediaType: out.Data[0].MediaType}, nil
}
