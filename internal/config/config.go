package config

import (
	"fmt"
	"os"
)

const (
	DefaultBaseURL     = "https://api.openai.com/v1"
	DefaultModel       = "gpt-4o-mini"
	DefaultAddr        = "127.0.0.1:8787"
	DefaultServerURL   = "http://127.0.0.1:8787"
)

// Config holds the settings for the server process, which owns the LLM
// connection and tool execution.
type Config struct {
	// Addr is the address the server listens on.
	Addr string
	BaseURL string
	Model   string
	APIKey  string
}

// Env reads server configuration from environment variables. It does not
// validate; callers should validate before use.
func Env() Config {
	return Config{
		Addr:    getenv("PORTER_ADDR", DefaultAddr),
		BaseURL: getenv("PORTER_BASE_URL", DefaultBaseURL),
		Model:   getenv("PORTER_MODEL", DefaultModel),
		APIKey:  os.Getenv("PORTER_API_KEY"),
	}
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

// ClientConfig holds the settings for a thin client (REPL or one-shot) that
// talks to the server over HTTP and owns no conversation state.
type ClientConfig struct {
	// ServerURL is the base URL of the porter server.
	ServerURL string
	// LogFile, when set, sends the event stream and progress lines to this file
	// instead of stderr, to keep the REPL terminal quiet (e.g. inside a
	// container).
	LogFile string
}

// ClientEnv reads client configuration from environment variables.
func ClientEnv() ClientConfig {
	return ClientConfig{
		ServerURL: getenv("PORTER_SERVER_URL", DefaultServerURL),
		LogFile:   os.Getenv("PORTER_LOG"),
	}
}

func getenv(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}