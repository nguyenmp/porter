package tools

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"strings"
)

// AssetReadTool is the private provider-side tool that backs the
// agent-served publish_asset tool: it reads a file from the execution
// environment and returns its bytes to the agent, which stores them on the
// server under the session's asset root and hands the model back a hosted
// URL. The read runs on the active provider so publish_asset works for any
// execution context — local sandbox or a remote host — like spool_write does
// in reverse. It is never in Defs: the model cannot call it directly; only
// the agent can.
const AssetReadTool = "_porter_asset_read"

// AssetMaxBytes caps a single published asset. The cap is enforced on the
// provider (cheap stat before the read, so a huge file is never slurped) and
// again by the agent after it decodes the base64, keeping the two ends of the
// private channel honest.
const AssetMaxBytes = 10 << 20 // 10 MiB

// assetReadResponse is the JSON payload _porter_asset_read streams back: the
// resolved absolute path, the byte size, and the file contents base64-encoded
// (the exec channel is text, so binary bytes travel as base64).
type assetReadResponse struct {
	Path string `json:"path"`
	Size int    `json:"size"`
	Data string `json:"data"`
}

// runAssetReadDir serves an _porter_asset_read call: it resolves path against
// the execution directory (mirroring the file tools), reads the file, and
// returns the JSON payload above. Unlike the line-oriented file tools this is
// a raw byte read — publish_asset exists to move binary artifacts (PNGs, HTML
// files) to the server, so NUL bytes are content, not a refusal.
func runAssetReadDir(args []byte, dir string) (io.ReadCloser, error) {
	var in struct {
		Path string `json:"path"`
	}
	if err := json.Unmarshal(args, &in); err != nil {
		return nil, fmt.Errorf("parse asset read arguments: %w", err)
	}
	path, err := resolveEditPath(dir, in.Path)
	if err != nil {
		return nil, err
	}
	info, err := os.Stat(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, fmt.Errorf("publish_asset: %s: no such file (create it first, e.g. with shell or another rendering tool)", path)
		}
		return nil, fmt.Errorf("stat %s: %w", path, err)
	}
	if info.IsDir() {
		return nil, fmt.Errorf("publish_asset: %s is a directory, not a file", path)
	}
	if info.Size() > AssetMaxBytes {
		return nil, fmt.Errorf("publish_asset: %s is %d bytes; the limit is %d bytes (%d MiB)", path, info.Size(), AssetMaxBytes, AssetMaxBytes/(1<<20))
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read %s: %w", path, err)
	}
	payload, err := json.Marshal(assetReadResponse{Path: path, Size: len(data), Data: base64.StdEncoding.EncodeToString(data)})
	if err != nil {
		return nil, fmt.Errorf("encode %s: %w", path, err)
	}
	return &stringStream{strings.NewReader(string(payload))}, nil
}
