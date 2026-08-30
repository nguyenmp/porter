package config

import (
	"testing"
)

func TestValidate(t *testing.T) {
	tests := []struct {
		name    string
		cfg     Config
		wantErr bool
	}{
		{"default endpoint missing key", Config{BaseURL: DefaultBaseURL, Model: "m", APIKey: ""}, true},
		{"default endpoint with key", Config{BaseURL: DefaultBaseURL, Model: "m", APIKey: "k"}, false},
		{"custom endpoint no key ok", Config{BaseURL: "http://localhost:4000/v1", Model: "m", APIKey: ""}, false},
		{"empty base url", Config{BaseURL: "", Model: "m"}, true},
		{"empty model", Config{BaseURL: DefaultBaseURL, Model: ""}, true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.cfg.Validate()
			if (err != nil) != tt.wantErr {
				t.Fatalf("Validate() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestClientEnvReadsAuth(t *testing.T) {
	t.Setenv("PORTER_SERVER_URL", "http://example:8787")
	t.Setenv("PORTER_AUTH_USERNAME", "porter")
	t.Setenv("PORTER_AUTH_PASSWORD", "hunter2")
	c := ClientEnv()
	if c.Username != "porter" || c.Password != "hunter2" {
		t.Fatalf("ClientEnv auth = %q/%q, want porter/hunter2", c.Username, c.Password)
	}
}
