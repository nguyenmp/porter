package config

import (
	"fmt"
	"os"
)

const (
	DefaultBaseURL = "https://api.openai.com/v1"
	DefaultModel   = "gpt-4o-mini"
)

// Config holds the resolved runtime settings for a single invocation.
type Config struct {
	BaseURL string
	Model   string
	APIKey  string
}

// Env reads configuration from environment variables without validating
// presence; callers are expected to pass overrides via CLI flags afterward.
func Env() Config {
	return Config{
		BaseURL: getenv("PORTER_BASE_URL", DefaultBaseURL),
		Model:   getenv("PORTER_MODEL", DefaultModel),
		APIKey:  os.Getenv("PORTER_API_KEY"),
	}
}

func getenv(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

// Validate checks that required fields are present and meaningful.
func (c Config) Validate() error {
	if c.BaseURL == "" {
		return fmt.Errorf("base URL is required (set PORTER_BASE_URL)")
	}
	if c.Model == "" {
		return fmt.Errorf("model is required (set PORTER_MODEL)")
	}
	if c.APIKey == "" && c.BaseURL == DefaultBaseURL {
		return fmt.Errorf("API key is required for the default endpoint (set PORTER_API_KEY)")
	}
	return nil
}
