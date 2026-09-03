// File editing tools that give the model a cheaper, more reliable alternative
// to rewriting whole files or editing by line-number patches through the shell:
//
//   - read_with_line_numbers reads a file (or a numbered window of it) with
//     cat -n style line numbers, so the model can reason about where things sit
//     and drive the other two tools by those numbers.
//   - line_replace cuts a whole-line range out of a file and pastes text in,
//     keyed by the numbering read_with_line_numbers reports. Because the edit
//     is a line-range splice it never rewrites the rest of the file, so a
//     change costs tokens proportional to the change, not the file.
//   - string_replace finds an exact literal string (no regex) and swaps every
//     occurrence for new text in one step. The model must state exactly how
//     many times old_text appears (expected_count, default 1), so a mismatch
//     is an error listing every match with context instead of a silent wrong
//     edit — the "read before edit" guarantee enforced by the model's own
//     output rather than by provider state (which cannot survive execution
//     handoff).
//
// All three run in Go on the execution provider, next to the shell tool, so
// every environment that can run shell commands gets them. They share the
// shell's working directory and see the same files, and they write atomically
// (temp file in the target's directory, then rename over it) so a failed or
// crashed edit never leaves the file half-written.
//
// Line numbering is exactly cat -n's: a file's lines are the elements of
// strings.SplitAfter(content, "\n") — every element ends in "\n" except the
// last, and an empty file has zero lines. Blank lines count; a trailing
// newline does not add a phantom line. Reads and edits both report whether the
// file ends in a newline, because that is the one boundary where text can be
// glued: inserting before/after a file that lacks a trailing newline splices
// onto the neighboring line. The tools never add or remove newlines
// themselves — that stays the model's call — but the success echo shows the
// changed lines with their new numbers so a botched newline is visible and
// fixable with another edit.
package tools

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"porter/internal/llm"
)

// The model-facing names of the file editing tools. They are declared next to
// shell by every provider, so keep them short and unambiguous.
const (
	ReadLinesTool   = "read_with_line_numbers"
	LineReplaceTool = "line_replace"
	StringReplace   = "string_replace"
)

// fileToolDefs returns the schemas of the file editing tools, declared next to
// shell so any execution environment can serve them. The descriptions carry
// when-to-use-which guidance because the model sees them on every request.
func fileToolDefs() []llm.Tool {
	return []llm.Tool{
		{
			Type: "function",
			Function: llm.Function{
				Name: ReadLinesTool,
				Description: "Read a file with line numbers (cat -n style: a right-aligned 1-based number, a tab, then the line's " +
					"exact content — the number and tab are not part of the file). Paths are relative to the working directory, same as shell. " +
					"Returns a header 'file has N lines (M bytes), ends with newline: yes/no, showing lines A-B' then the numbered lines. " +
					"start (default 1) is the first line to show, 1-based inclusive; limit (default -1) is how many lines to show (-1 = through end of file). " +
					"Read a large file in windows (small start/limit) instead of all at once, then edit by those line numbers with line_replace. " +
					"The ends-with-newline state matters: a file that does not end with a newline has its last line glued to whatever is inserted after it, " +
					"so check the header before inserting or appending lines. Refuses binary files (use shell + xxd/strings for those).",
				Parameters: map[string]any{
					"type": "object",
					"properties": map[string]any{
						"path": map[string]any{
							"type":        "string",
							"description": "Path of the file to read, relative to the working directory.",
						},
						"start": map[string]any{
							"type":        "integer",
							"description": "First line to show, 1-based inclusive (default 1).",
						},
						"limit": map[string]any{
							"type":        "integer",
							"description": "Number of lines to show; -1 (default) reads through end of file.",
						},
					},
					"required": []string{"path"},
				},
			},
		},
		{
			Type: "function",
			Function: llm.Function{
				Name: LineReplaceTool,
				Description: "Cut a whole-line range out of an existing file and paste text in, keyed by the line numbers " +
					"read_with_line_numbers reports (1-based). start is the first line to replace (inclusive); end is one past the last " +
					"line to replace (exclusive), and may be one past the last line (replace through end of file). When start == end the tool " +
					"inserts new_text before that line number (the inserted text becomes that line). Set new_text to an empty string to delete " +
					"lines start through end-1. To replace the whole file use start=1 and end=N+1. new_text should end with a newline to keep " +
					"line separation except when inserting at end of file; the tool never adds or removes newlines itself. Applies atomically " +
					"(temp file + rename) and never rewrites untouched lines. Cannot create a file — make new files with the shell tool. The " +
					"success echo shows the changed lines with their new numbers so you can confirm separation, then re-read for current numbering. " +
					"Use this for a big block you don't want to retype; use string_replace for a distinctive small text change.",
				Parameters: map[string]any{
					"type": "object",
					"properties": map[string]any{
						"path": map[string]any{
							"type":        "string",
							"description": "Path of the file to edit, relative to the working directory. Must already exist.",
						},
						"start": map[string]any{
							"type":        "integer",
							"description": "First line to replace, 1-based inclusive.",
						},
						"end": map[string]any{
							"type":        "integer",
							"description": "One past the last line to replace (exclusive); equal to start to insert before that line, or N+1 to replace through end of file.",
						},
						"new_text": map[string]any{
							"type":        "string",
							"description": "Replacement text. Empty string deletes the range. Should end in a newline for line separation; no newline is added automatically.",
						},
					},
					"required": []string{"path", "start", "end", "new_text"},
				},
			},
		},
		{
			Type: "function",
			Function: llm.Function{
				Name: StringReplace,
				Description: "Replace an exact literal string in an existing file with new_text — no regex; spaces, tabs, and newlines must match " +
					"exactly (read the file first to get them right). old_text must not be empty. Every occurrence of old_text is replaced, so " +
					"old_text must appear exactly expected_count times (default 1) in the whole file; if the count differs the edit is rejected and " +
					"every match is listed with line-numbered context so you can disambiguate (lengthen old_text, or switch to line_replace). " +
					"Applies atomically. Cannot create a file — make new files with the shell tool. Prefer line_replace for whole lines or big " +
					"blocks. Success echoes before/after context with line numbers.",
				Parameters: map[string]any{
					"type": "object",
					"properties": map[string]any{
						"path": map[string]any{
							"type":        "string",
							"description": "Path of the file to edit, relative to the working directory. Must already exist.",
						},
						"old_text": map[string]any{
							"type":        "string",
							"description": "Exact text to find. Must not be empty and must appear exactly expected_count times.",
						},
						"new_text": map[string]any{
							"type":        "string",
							"description": "Replacement text; may be empty to delete occurrences.",
						},
						"expected_count": map[string]any{
							"type":        "integer",
							"description": "How many times old_text must appear in the file (default 1). Any other count is an error, not a partial edit.",
						},
					},
					"required": []string{"path", "old_text", "new_text"},
				},
			},
		},
	}
}

// editContext is how many unchanged lines of context surround each side of a
// changed region in the tools' success echoes and mismatch listings.
const editContext = 2

// runReadDir serves a read_with_line_numbers call: a header plus the requested
// window of numbered lines. The result is returned whole (the agent's recall /
// truncation handles oversized results), not streamed per line.
func runReadDir(args []byte, dir string) (io.ReadCloser, error) {
	var in struct {
		Path  string `json:"path"`
		Start *int   `json:"start"`
		Limit *int   `json:"limit"`
	}
	if err := json.Unmarshal(args, &in); err != nil {
		return nil, fmt.Errorf("parse %s arguments: %w", ReadLinesTool, err)
	}
	path, err := resolveEditPath(dir, in.Path)
	if err != nil {
		return nil, err
	}
	content, err := readEditFile(path)
	if err != nil {
		return nil, err
	}
	start := 1
	if in.Start != nil {
		start = *in.Start
	}
	limit := -1
	if in.Limit != nil {
		limit = *in.Limit
	}
	if start < 1 {
		return nil, fmt.Errorf("%s: start must be >= 1 (got %d)", ReadLinesTool, start)
	}
	if limit < -1 || limit == 0 {
		return nil, fmt.Errorf("%s: limit must be -1 (through end of file) or >= 1 (got %d)", ReadLinesTool, limit)
	}
	lines := splitLines(content)
	n := len(lines)
	if start > n+1 {
		return nil, fmt.Errorf("%s: %s has %s; start=%d is past the end (max %d)", ReadLinesTool, path, linesPhrase(n), start, n+1)
	}
	last := n // 1-based inclusive last line to show; n means through end of file
	if limit != -1 && start+limit-1 < last {
		last = start + limit - 1
	}
	var b strings.Builder
	fmt.Fprintf(&b, "[file: %s, %d bytes, ends with newline: %s, showing ", linesPhrase(n), len(content), yesNo(strings.HasSuffix(content, "\n")))
	if last < start {
		b.WriteString("no lines]\n")
	} else {
		fmt.Fprintf(&b, "lines %d-%d]\n", start, last)
		b.WriteString(renderNumbered(lines[start-1:last], start))
	}
	return &stringStream{strings.NewReader(b.String())}, nil
}

// runLineReplaceDir serves a line_replace call: splice new_text in over the
// half-open line range [start, end) and write the result back atomically. The
// tool never creates files (that is shell's job) and never touches lines
// outside the range. The success echo shows the exact old lines removed and
// the new file's lines that now cover the change, with current numbers, so the
// model can confirm the newline seams came out as intended.
func runLineReplaceDir(args []byte, dir string) (io.ReadCloser, error) {
	var in struct {
		Path    string `json:"path"`
		Start   int    `json:"start"`
		End     int    `json:"end"`
		NewText string `json:"new_text"`
	}
	if err := json.Unmarshal(args, &in); err != nil {
		return nil, fmt.Errorf("parse %s arguments: %w", LineReplaceTool, err)
	}
	path, err := resolveEditPath(dir, in.Path)
	if err != nil {
		return nil, err
	}
	content, err := readEditFile(path)
	if err != nil {
		return nil, err
	}
	lines := splitLines(content)
	n := len(lines)
	if in.Start < 1 {
		return nil, fmt.Errorf("%s: start must be >= 1 (got %d)", LineReplaceTool, in.Start)
	}
	if in.End < in.Start {
		return nil, fmt.Errorf("%s: end (%d) must be >= start (%d)", LineReplaceTool, in.End, in.Start)
	}
	if in.End > n+1 {
		return nil, fmt.Errorf("%s: %s has %s; end=%d is past one past the last line (max %d)", LineReplaceTool, path, linesPhrase(n), in.End, n+1)
	}
	head := strings.Join(lines[:in.Start-1], "") // byte offset 0..headLen
	headLen := len(head)
	out := head + in.NewText + strings.Join(lines[in.End-1:], "")

	// The header describes what was asked (pre-edit numbering), then the file's
	// resulting state.
	action := "changed"
	switch {
	case in.NewText == "" && in.Start == in.End:
		action = "inserted nothing"
	case in.NewText == "":
		if in.End == in.Start+1 {
			action = fmt.Sprintf("deleted line %d", in.Start)
		} else {
			action = fmt.Sprintf("deleted lines %d-%d", in.Start, in.End-1)
		}
	case in.Start == in.End:
		action = fmt.Sprintf("inserted text before line %d", in.Start)
	case in.End == n+1:
		action = fmt.Sprintf("replaced lines %d-%d (through end of file) with new text", in.Start, n)
	default:
		action = fmt.Sprintf("replaced lines %d-%d with new text", in.Start, in.End-1)
	}
	newLines := splitLines(out)

	if out == content {
		// The edit was a no-op (new_text equals what it replaced): report it
		// honestly and leave the file untouched.
		var b strings.Builder
		fmt.Fprintf(&b, "[%s: %s: %s produced no change; file still has %s (%d bytes), ends with newline: %s]\n",
			LineReplaceTool, path, action, linesPhrase(n), len(out), yesNo(strings.HasSuffix(out, "\n")))
		return &stringStream{strings.NewReader(b.String())}, nil
	}

	if err := writeEditFile(path, out); err != nil {
		return nil, err
	}

	nn := len(newLines)
	var b strings.Builder
	fmt.Fprintf(&b, "[%s: %s: %s; was %s, now %s (%d bytes), ends with newline: %s]\n",
		LineReplaceTool, path, action, linesPhrase(n), linesPhrase(nn), len(out), yesNo(strings.HasSuffix(out, "\n")))

	// Old lines removed, with pre-edit numbers. Exact geometry: old lines
	// start..end-1. Only present when a range (not a pure insertion) was cut.
	if in.Start < in.End {
		b.WriteString("removed (old line numbers):\n")
		b.WriteString(renderRange(lines, in.Start-1, in.End-2))
	}
	// New lines covering the change, with post-edit numbers. Found by byte
	// geometry: new_text occupies out[headLen : headLen+len(new_text)], and the
	// lines containing its first and last bytes are the affected ones. This is
	// ground truth on the new file, so a newline glued across a seam shows up
	// as a merged numbered line the model can see and fix.
	b.WriteString("now (new line numbers):\n")
	if in.NewText == "" {
		// Pure deletion: nothing occupies the range; show the seam between the
		// last kept line and the first line after the deletion.
		b.WriteString(renderBand(newLines, in.Start-1, in.Start-2))
	} else {
		first := lineOfOffset(out, headLen) - 1
		last := lineOfOffset(out, headLen+len(in.NewText)-1) - 1
		b.WriteString(renderBand(newLines, first, last))
	}
	b.WriteString("\nRe-read the file (read_with_line_numbers) for current line numbers.\n")
	return &stringStream{strings.NewReader(b.String())}, nil
}

// runStringReplaceDir serves a string_replace call: find old_text as a literal
// substring everywhere in the file, require it to appear exactly
// expected_count times, and swap every occurrence for new_text in one atomic
// write. A count mismatch rejects the whole edit and lists every match with
// line-numbered context instead of guessing.
func runStringReplaceDir(args []byte, dir string) (io.ReadCloser, error) {
	var in struct {
		Path          string `json:"path"`
		OldText       string `json:"old_text"`
		NewText       string `json:"new_text"`
		ExpectedCount int    `json:"expected_count"`
	}
	if err := json.Unmarshal(args, &in); err != nil {
		return nil, fmt.Errorf("parse %s arguments: %w", StringReplace, err)
	}
	if in.ExpectedCount == 0 {
		in.ExpectedCount = 1 // JSON omitted
	}
	if in.OldText == "" {
		return nil, fmt.Errorf("%s: old_text must not be empty (an empty string matches everywhere)", StringReplace)
	}
	if in.ExpectedCount < 1 {
		return nil, fmt.Errorf("%s: expected_count must be >= 1 (got %d)", StringReplace, in.ExpectedCount)
	}
	path, err := resolveEditPath(dir, in.Path)
	if err != nil {
		return nil, err
	}
	content, err := readEditFile(path)
	if err != nil {
		return nil, err
	}
	lines := splitLines(content)

	// All non-overlapping literal matches, as byte offsets.
	var occ []int
	for from := 0; ; {
		i := strings.Index(content[from:], in.OldText)
		if i < 0 {
			break
		}
		i += from
		occ = append(occ, i)
		from = i + len(in.OldText)
	}
	if len(occ) != in.ExpectedCount {
		return nil, matchMismatchError(StringReplace, path, in.OldText, len(occ), in.ExpectedCount, lines, occ, content)
	}

	// Splice replacements back to front so earlier offsets stay valid.
	out := content
	for i := len(occ) - 1; i >= 0; i-- {
		o := occ[i]
		out = out[:o] + in.NewText + out[o+len(in.OldText):]
	}
	newLines := splitLines(out)

	if out == content {
		var b strings.Builder
		fmt.Fprintf(&b, "[%s: %s: replacing %q with identical text produced no change; file still has %s]\n",
			StringReplace, path, in.OldText, linesPhrase(len(lines)))
		return &stringStream{strings.NewReader(b.String())}, nil
	}

	if err := writeEditFile(path, out); err != nil {
		return nil, err
	}

	var b strings.Builder
	fmt.Fprintf(&b, "[%s: %s: replaced %q %s; was %s, now %s (%d bytes), ends with newline: %s]\n",
		StringReplace, path, in.OldText, timesPhrase(len(occ)), linesPhrase(len(lines)), linesPhrase(len(newLines)), len(out), yesNo(strings.HasSuffix(out, "\n")))

	// Echo the changed band in the old file and the new file, each numbered in
	// its own file, with ±editContext lines around it. A single occurrence is
	// the common case and renders exactly; multiple far-apart occurrences widen
	// the band and renderBand caps its middle (the header already states the
	// count replaced).
	p := 0
	for p < len(lines) && p < len(newLines) && lines[p] == newLines[p] {
		p++
	}
	s := 0
	for s < len(lines)-p && s < len(newLines)-p && lines[len(lines)-1-s] == newLines[len(newLines)-1-s] {
		s++
	}
	lastOld := len(lines) - s
	lastNew := len(newLines) - s
	if p < lastOld {
		b.WriteString("before (old line numbers):\n")
		b.WriteString(renderBand(lines, p, lastOld-1))
	}
	b.WriteString("after (new line numbers):\n")
	b.WriteString(renderBand(newLines, p, lastNew-1))
	b.WriteString("\nRe-read the file (read_with_line_numbers) for current line numbers.\n")
	return &stringStream{strings.NewReader(b.String())}, nil
}

// matchMismatchError builds the rejection for a count mismatch: it names the
// mismatch and lists every match with ±editContext lines of numbered context,
// so the model can lengthen old_text or switch to line_replace. The result is
// an error, so the agent surfaces it as "error: ..." and feeds the repeat
// tracker.
func matchMismatchError(tool, path, old string, got, want int, lines []string, occ []int, content string) error {
	var b strings.Builder
	fmt.Fprintf(&b, "%s: expected %q to appear %s but found %d in %s",
		tool, old, timesPhrase(want), got, path)
	if len(occ) == 0 {
		return errors.New(b.String())
	}
	fmt.Fprintf(&b, ":\n")
	const maxList = 20
	for i, o := range occ {
		if i >= maxList {
			fmt.Fprintf(&b, "\n[... and %d more matches ...]\n", len(occ)-i)
			break
		}
		line := lineOfOffset(content, o)
		fmt.Fprintf(&b, "\nmatch %d at line %d:\n", i+1, line)
		b.WriteString(renderBand(lines, line-1, line-1))
	}
	return errors.New(b.String())
}

// splitLines splits content into its numbered lines matching cat -n. It uses
// strings.SplitAfter, then drops the phantom empty element that split appends
// when content ends in "\n" (SplitAfter("a\n") is ["a\n", ""], but cat -n
// counts one line). Empty content has zero lines, not the one empty element
// SplitAfter returns. After the drop, every element ends in "\n" except
// possibly the last. This is the single source of truth for what a "line" is
// across the read and edit tools.
func splitLines(s string) []string {
	if s == "" {
		return nil
	}
	lines := strings.SplitAfter(s, "\n")
	if lines[len(lines)-1] == "" {
		lines = lines[:len(lines)-1]
	}
	return lines
}

// renderNumbered renders lines in cat -n style, numbering from base (1-based):
// a right-aligned number, a tab, then the line's exact content. A source line
// that lacks a trailing newline gets one added so the tool output stays well
// formed; the ends-with-newline state is reported in the header, never hidden.
func renderNumbered(lines []string, base int) string {
	var b strings.Builder
	w := 6
	if len(lines) > 0 {
		if n := len(strconv.Itoa(base + len(lines) - 1)); n > w {
			w = n
		}
	}
	for i, ln := range lines {
		fmt.Fprintf(&b, "%*d\t%s", w, base+i, ln)
		if !strings.HasSuffix(ln, "\n") {
			b.WriteByte('\n')
		}
	}
	return b.String()
}

// renderBand renders the changed region lines[from:to+1] (0-based inclusive
// bounds) plus up to editContext unchanged lines on each side, all with their
// real line numbers. A region longer than the cap is shown as its first and
// last few lines with an omission marker, so echoing a big inserted block never
// re-pays its full token cost; the seams — where newline mistakes would show —
// are always visible. from may be one past to (an empty band), in which case
// the context is centred on the seam at index from.
func renderBand(lines []string, from, to int) string {
	const cap = 8
	if to < from {
		// Empty band (a deletion): centre context on the seam at index from.
		lo := from - editContext
		if lo < 0 {
			lo = 0
		}
		hi := from + editContext + 1
		if hi > len(lines) {
			hi = len(lines)
		}
		if lo < hi {
			return renderNumbered(lines[lo:hi], lo+1)
		}
		return ""
	}
	lo := from - editContext
	if lo < 0 {
		lo = 0
	}
	hi := to + editContext + 1
	if hi > len(lines) {
		hi = len(lines)
	}
	bandLen := to - from + 1
	if bandLen <= cap {
		return renderNumbered(lines[lo:hi], lo+1)
	}
	// Long band: leading context, first few changed lines, omission marker,
	// last few changed lines, trailing context.
	first := from
	last := to
	b := renderNumbered(lines[lo:first+cap/2], lo+1)
	omitted := (last - cap/2) - (first + cap/2) + 1
	b += fmt.Sprintf("[... %s omitted ...]\n", linesPhrase(omitted))
	b += renderNumbered(lines[last-cap/2:hi], last-cap/2+1)
	return b
}

// renderRange renders exactly lines[from:to+1] (0-based inclusive), with no
// added context, capping long ranges the way renderBand does. It is for blocks
// whose content is the whole point (the lines a line_replace removed) where
// surrounding context would be mistaken for part of the change.
func renderRange(lines []string, from, to int) string {
	const cap = 8
	bandLen := to - from + 1
	if bandLen <= cap {
		return renderNumbered(lines[from:to+1], from+1)
	}
	b := renderNumbered(lines[from:from+cap/2], from+1)
	omitted := (to - cap/2) - (from + cap/2) + 1
	b += fmt.Sprintf("[... %s omitted ...]\n", linesPhrase(omitted))
	b += renderNumbered(lines[to-cap/2+1:to+1], to-cap/2+2)
	return b
}

// resolveEditPath resolves a tool path against the execution directory (dir,
// or the process working directory when empty), mirroring how the shell tool
// resolves relative paths so both tools edit the same files.
func resolveEditPath(dir, p string) (string, error) {
	if strings.TrimSpace(p) == "" {
		return "", errors.New("path is required")
	}
	if dir != "" && !filepath.IsAbs(p) {
		p = filepath.Join(dir, p)
	}
	abs, err := filepath.Abs(p)
	if err != nil {
		return "", fmt.Errorf("resolve path %q: %w", p, err)
	}
	return abs, nil
}

// readEditFile reads a text file for the edit tools, refusing binaries (NUL
// bytes) and non-files with guidance toward the shell tool, which is the
// escape hatch for those cases.
func readEditFile(path string) (string, error) {
	info, err := os.Stat(path)
	if err != nil {
		if os.IsNotExist(err) {
			return "", fmt.Errorf("%s: no such file (create new files with the shell tool, e.g. 'cat > file')", path)
		}
		return "", fmt.Errorf("stat %s: %w", path, err)
	}
	if info.IsDir() {
		return "", fmt.Errorf("%s: is a directory, not a file", path)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return "", fmt.Errorf("read %s: %w", path, err)
	}
	if bytes.IndexByte(data, 0) != -1 {
		return "", fmt.Errorf("%s: looks like a binary file (contains NUL bytes); use the shell tool with xxd or strings instead", path)
	}
	return string(data), nil
}

// writeEditFile writes content to path atomically: a temp file in the target's
// directory (same filesystem, so rename cannot fail across devices) carrying
// the target's file mode, then rename over the target. A symlink target is
// resolved first so the edit goes through the link to the real file instead of
// replacing the link with a regular file.
func writeEditFile(path, content string) error {
	target := path
	if real, err := filepath.EvalSymlinks(path); err == nil {
		target = real
	}
	mode := os.FileMode(0o644)
	if info, err := os.Stat(target); err == nil {
		mode = info.Mode().Perm()
	}
	dir := filepath.Dir(target)
	tmp, err := os.CreateTemp(dir, ".porter-edit-*.tmp")
	if err != nil {
		return fmt.Errorf("create temp file in %s: %w", dir, err)
	}
	name := tmp.Name()
	fail := func(err error) error {
		_ = tmp.Close()
		_ = os.Remove(name)
		return err
	}
	if _, err := tmp.WriteString(content); err != nil {
		return fail(fmt.Errorf("write temp file: %w", err))
	}
	if err := tmp.Chmod(mode); err != nil {
		return fail(fmt.Errorf("chmod temp file: %w", err))
	}
	if err := tmp.Close(); err != nil {
		_ = os.Remove(name)
		return fmt.Errorf("close temp file: %w", err)
	}
	if err := os.Rename(name, target); err != nil {
		_ = os.Remove(name)
		return fmt.Errorf("rename temp over %s: %w", target, err)
	}
	return nil
}

// lineOfOffset returns the 1-based line number of the byte at offset in s.
func lineOfOffset(s string, off int) int {
	if off > len(s) {
		off = len(s)
	}
	return 1 + strings.Count(s[:off], "\n")
}

// linesPhrase renders a line count in prose: "0 lines", "1 line", "12 lines".
func linesPhrase(n int) string {
	if n == 1 {
		return "1 line"
	}
	return fmt.Sprintf("%d lines", n)
}

// timesPhrase renders an occurrence count in prose: "once", "3 times".
func timesPhrase(n int) string {
	switch n {
	case 1:
		return "once"
	default:
		return fmt.Sprintf("%d times", n)
	}
}

// yesNo renders a boolean as the header words "yes"/"no".
func yesNo(b bool) string {
	if b {
		return "yes"
	}
	return "no"
}
