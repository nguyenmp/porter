// Package spool defines the spool_output tool: it saves a tool result's full
// bytes to a file on the active execution provider, so the model can process
// output with shell (grep, jq, sed) and the file tools instead of reading it
// in context. It is recall's counterpart in the other direction:
// recall_tool_output pages a stored result's bytes into the model's context;
// spool_output writes them to disk where the tools run.
//
// spool_output is served by the agent from the turn's history (like
// recall_tool_output), but the write itself is executed by the active
// provider through a private provider tool (tools.SpoolWriteTool), so the
// file lands on the same filesystem shell and the file tools edit — whatever
// provider is active, even when it is not the one that produced the output.
// The file is a raw mirror of the committed result: no trimming, no exit-code
// line removed, so it can never drift from what history holds.
package spool

import (
	"encoding/json"
	"errors"
	"fmt"
	"path/filepath"
	"strconv"

	"porter/internal/llm"
	"porter/internal/tools"
)

// OutputTool is the model-facing name of the spool tool.
const OutputTool = "spool_output"

// MinHintBytes is the smallest fully-shown tool result that earns a
// spool_output hint in the model view. Outputs above ~1KB are usually data
// (diffs, listings, JSON, logs) rather than prose to read, and data is what
// shell is for.
const MinHintBytes = 1024

// DefaultDir is the directory (under the provider's working directory) where
// spool_output writes the full output when the caller gives no path.
const DefaultDir = ".porter/out"

// WriteArgs is the payload the agent passes to the provider's private write
// tool: the resolved destination path plus the full output bytes.
type WriteArgs struct {
	Path    string `json:"path"`
	Content string `json:"content"`
}

// args is the parsed model-facing spool_output call.
type args struct {
	CallID string `json:"call_id"`
	Path   string `json:"path"`
}

// ParseArgs validates a spool_output call and resolves its destination path:
// the caller's path when given, else DefaultDir/<call_id>. It returns the
// source call id (the result to save) and the write path, which the provider
// resolves against its working directory.
func ParseArgs(raw string) (callID, path string, err error) {
	var in args
	if err := json.Unmarshal([]byte(raw), &in); err != nil {
		return "", "", fmt.Errorf("parse spool_output arguments: %w", err)
	}
	if in.CallID == "" {
		return "", "", errors.New("spool_output: call_id is required (the id of the tool result to save, shown in its truncation header)")
	}
	if in.Path == "" {
		in.Path = filepath.Join(DefaultDir, in.CallID)
	}
	return in.CallID, in.Path, nil
}

// Lookup returns the full stored content of the tool result with the given
// call id, mirroring recall_tool_output's lookup, so the file is written from
// the exact bytes the source tool produced. A recall window is not spoolable
// on its own: it is a slice of a larger stored result, not the result.
func Lookup(history []llm.ChatMessage, callID string) (string, error) {
	for _, m := range history {
		if m.Role == "tool" && m.ToolCallID == callID {
			if m.ToolOutput != nil && m.ToolOutput.Recall {
				return "", fmt.Errorf("spool_output: call_id %q is a recall_tool_output window, not a full tool result; spool the original result instead", callID)
			}
			return m.Content, nil
		}
	}
	return "", fmt.Errorf("spool_output: unknown call_id %q (it must be a tool result in this conversation's history)", callID)
}

// WritePayload renders the arguments for the provider's private write tool
// (tools.SpoolWriteTool): the resolved destination path plus the full output
// bytes. Marshaling cannot fail for these fields.
func WritePayload(path, content string) []byte {
	b, _ := json.Marshal(WriteArgs{Path: path, Content: content})
	return b
}

// Def is the model-facing definition of the spool_output tool. The agent
// declares it on every request alongside recall_tool_output, so the model can
// save any result it has seen.
func Def() llm.Tool {
	return llm.Tool{
		Type: "function",
		Function: llm.Function{
			Name: OutputTool,
			Description: "Save a tool result's full output to a file on the execution environment, so you can work with it using shell (grep, jq, sed, head) and the file tools instead of reading it in context. " +
				"call_id is the id of the tool result to save (shown in its truncation header). path is optional: omit it to write to " + DefaultDir + "/<call_id> under the working directory, " +
				"or give a path (relative to the working directory, or absolute). The result reports where the file was written.",
			Parameters: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"call_id": map[string]any{
						"type":        "string",
						"description": "The tool_call_id of the tool result to save (shown in its truncation header).",
					},
					"path": map[string]any{
						"type":        "string",
						"description": "Optional destination: relative to the working directory or absolute. Defaults to " + DefaultDir + "/<call_id>.",
					},
				},
				"required": []string{"call_id"},
			},
		},
	}
}

// FooterHint returns the model-view nudge appended to a fully-shown tool
// result large enough to be data: it tells the model it can save the bytes to
// a file instead of keeping them in context. recall appends it.
func FooterHint(callID string) string {
	return fmt.Sprintf("\n[If this output is data you'd rather process with shell than keep in context, save it: %s(call_id=%s) — path optional, default %s/<call_id>.]",
		OutputTool, strconv.Quote(callID), DefaultDir)
}

// ShouldHint reports whether a tool's output should carry the spool hint.
// Tools whose output is prose to read, or that already live on disk (files
// the file tools read or edited, skill bodies), get no hint: spooling them
// would duplicate something the model already has.
func ShouldHint(toolName string) bool {
	switch toolName {
	case tools.ReadLinesTool, tools.LineInsertTool, tools.LineReplaceTool, tools.StringReplace, tools.LoadSkillTool:
		return false
	}
	return true
}
