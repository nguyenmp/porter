package client

import (
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestNewSendsBasicAuth(t *testing.T) {
	var gotAuth string
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotAuth = r.Header.Get("Authorization")
		w.WriteHeader(http.StatusOK)
	}))
	defer ts.Close()

	c := New(ts.URL, BasicAuth{Username: "porter", Password: "s3cret"})
	req, _ := http.NewRequest(http.MethodGet, ts.URL+"/anything", nil)
	resp, err := c.http.Do(req)
	if err != nil {
		t.Fatalf("Do: %v", err)
	}
	resp.Body.Close()
	want := "Basic cG9ydGVyOnMzY3JldA==" // base64("porter:s3cret")
	if gotAuth != want {
		t.Fatalf("Authorization = %q, want %q", gotAuth, want)
	}
}

func TestNewWithoutAuthSendsNoHeader(t *testing.T) {
	var gotAuth string
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotAuth = r.Header.Get("Authorization")
		w.WriteHeader(http.StatusOK)
	}))
	defer ts.Close()

	c := New(ts.URL)
	req, _ := http.NewRequest(http.MethodGet, ts.URL+"/anything", nil)
	resp, err := c.http.Do(req)
	if err != nil {
		t.Fatalf("Do: %v", err)
	}
	resp.Body.Close()
	if gotAuth != "" {
		t.Fatalf("Authorization = %q, want empty", gotAuth)
	}
}
