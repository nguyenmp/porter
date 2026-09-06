package tools

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
)

// SpoolWriteTool is the private provider-side tool that backs the
// agent-served spool_output tool: it writes a tool result's full bytes to a
// file in the execution environment. The agent looks the result up in history
// and calls this tool on the active provider, so the file lands on the same
// filesystem shell and the file tools edit. It is never in Defs — the model
// cannot call it directly; only the agent can.
const SpoolWriteTool = "_porter_spool_write"

// runSpoolWriteDir writes spooled bytes to path (resolved against the
// execution directory, mirroring the file tools) and returns a short
// confirmation naming the file. The content is written verbatim — no
// exit-code line, no trimming — so the file is a raw mirror of the committed
// result and can never drift from what history holds. Any cleanup of the
// shell tool's trailing exit line is the caller's job when it uses the file.
func runSpoolWriteDir(args []byte, dir string) (io.ReadCloser, error) {
	var in struct {
		Path    string `json:"path"`
		Content string `json:"content"`
	}
	if err := json.Unmarshal(args, &in); err != nil {
		return nil, fmt.Errorf("parse spool write arguments: %w", err)
	}
	path, err := resolveEditPath(dir, in.Path)
	if err != nil {
		return nil, err
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return nil, fmt.Errorf("create spool directory %s: %w", filepath.Dir(path), err)
	}
	if err := writeEditFile(path, in.Content); err != nil {
		return nil, fmt.Errorf("write spooled output: %w", err)
	}
	confirm := fmt.Sprintf("wrote %d bytes to %s", len(in.Content), path)
	return &stringStream{strings.NewReader(confirm)}, nil
}
