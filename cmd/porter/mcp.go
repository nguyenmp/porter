package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"

	"porter/internal/mcp"
)

// runMCP handles `porter mcp <login|logout> <server-name>`. Both resolve the
// server from the MCP config (./porter.mcp.json first, then the host's
// ~/.porter/porter.mcp.json) and act on its URL.
func runMCP(args []string, stdout io.Writer) error {
	if len(args) == 0 {
		return errors.New("usage: porter mcp <login|logout> <server-name>")
	}
	switch args[0] {
	case "login":
		if len(args) < 2 {
			return errors.New("usage: porter mcp login <server-name>")
		}
		return mcpLogin(context.Background(), args[1], stdout)
	case "logout":
		if len(args) < 2 {
			return errors.New("usage: porter mcp logout <server-name>")
		}
		return mcpLogout(context.Background(), args[1], stdout)
	default:
		return fmt.Errorf("unknown mcp subcommand %q (supported: login, logout)", args[0])
	}
}

// mcpLogin runs the interactive OAuth flow for the named server.
func mcpLogin(ctx context.Context, name string, stdout io.Writer) error {
	serverURL, scope, err := findServerConfig(name)
	if err != nil {
		return err
	}
	return mcp.Login(ctx, http.DefaultClient, serverURL, name, scope, stdout, nil)
}

// mcpLogout revokes and clears the named server's stored token.
func mcpLogout(ctx context.Context, name string, stdout io.Writer) error {
	serverURL, _, err := findServerConfig(name)
	if err != nil {
		return err
	}
	if err := mcp.Logout(ctx, http.DefaultClient, serverURL); err != nil {
		return err
	}
	fmt.Fprintf(stdout, "Logged out of %s.\n", serverURL)
	return nil
}

// findServerConfig resolves a server name to its URL and OAuth scope from the
// MCP config: ./porter.mcp.json (the server's config) first, then
// ~/.porter/porter.mcp.json (the host's).
func findServerConfig(name string) (url, scope string, err error) {
	paths := []string{"porter.mcp.json"}
	if home, herr := os.UserHomeDir(); herr == nil {
		paths = append(paths, filepath.Join(home, ".porter", "porter.mcp.json"))
	}
	for _, path := range paths {
		u, sc, ok, err := serverInFile(path, name)
		if err != nil {
			return "", "", err
		}
		if ok {
			return u, sc, nil
		}
	}
	return "", "", fmt.Errorf("MCP server %q not found (looked in ./porter.mcp.json and ~/.porter/porter.mcp.json)", name)
}

// serverInFile looks a server name up in one MCP config file, returning its
// URL and auth scope. ok is false when the file is missing or the name is
// absent; a present-but-malformed file is an error.
func serverInFile(path, name string) (url, scope string, ok bool, err error) {
	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return "", "", false, nil
		}
		return "", "", false, err
	}
	var cfg struct {
		Servers []struct {
			Name string `json:"name"`
			URL  string `json:"url"`
			Auth struct {
				Scope string `json:"scope"`
			} `json:"auth"`
		} `json:"servers"`
	}
	if err := json.Unmarshal(data, &cfg); err != nil {
		return "", "", false, fmt.Errorf("parse %s: %w", path, err)
	}
	for _, s := range cfg.Servers {
		if s.Name == name {
			return s.URL, s.Auth.Scope, true, nil
		}
	}
	return "", "", false, nil
}
