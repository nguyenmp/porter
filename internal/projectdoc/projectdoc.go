// Package projectdoc finds the instruction files a repository keeps for coding
// agents — AGENTS.md and CLAUDE.md — for a file the agent opens. The file tools
// append them to a read's output, so the model reads a project's conventions
// together with the first file it reads in that part of the tree, instead of
// having to know to look for them.
//
// The docs come from the repository, not from the person the model works for,
// and Message labels them that way. A repo can state its build steps and its
// house style; it cannot pass for the operator.
package projectdoc

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// Names are the instruction file names, in the order one directory's docs are
// reported when it holds both.
var Names = []string{"AGENTS.md", "CLAUDE.md"}

// MaxDoc bounds the body kept from one instruction file, so a long doc cannot
// crowd out the file the model asked for. The cut lands on a line boundary and
// says where it happened.
const MaxDoc = 8 << 10

// Doc is one instruction file found on the way up from a read file.
type Doc struct {
	// Path is the absolute path of the instruction file.
	Path string
	// Body is its text, cut at MaxDoc.
	Body string
}

// Find returns the instruction files that govern path. It walks from the
// file's own directory up to root, searches only directories at or under root,
// and puts the outermost doc first, so the nearest directory's rules come last
// and read as the most specific. Missing, unreadable, and empty files are
// skipped: the docs are a bonus to the read, never a reason for it to fail.
func Find(path, root string) []Doc {
	if strings.TrimSpace(path) == "" || strings.TrimSpace(root) == "" {
		return nil
	}
	if abs, err := filepath.Abs(filepath.Clean(root)); err == nil {
		root = abs
	}
	dir, err := filepath.Abs(filepath.Clean(path))
	if err != nil {
		return nil
	}
	dir = filepath.Dir(dir)
	var dirs []string
	for at := dir; under(at, root); at = filepath.Dir(at) {
		dirs = append(dirs, at)
		if at == root {
			break
		}
	}
	// dirs runs nearest first. Walk it backwards so the widest scope's rules
	// come first and the nearest directory's read as the most specific.
	var docs []Doc
	for i := len(dirs) - 1; i >= 0; i-- {
		for _, name := range Names {
			p := filepath.Join(dirs[i], name)
			body, ok := read(p)
			if !ok {
				continue
			}
			docs = append(docs, Doc{Path: p, Body: body})
		}
	}
	return docs
}

// under reports whether dir is root itself or a directory below it.
func under(dir, root string) bool {
	return dir == root || strings.HasPrefix(dir, root+string(filepath.Separator))
}

// read returns a doc's body, or false when the path is missing, is not a
// regular file, or holds nothing but whitespace.
func read(path string) (string, bool) {
	info, err := os.Stat(path)
	if err != nil || !info.Mode().IsRegular() {
		return "", false
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return "", false
	}
	body := string(data)
	if strings.TrimSpace(body) == "" {
		return "", false
	}
	if len(body) > MaxDoc {
		cut := strings.LastIndexByte(body[:MaxDoc], '\n')
		if cut <= 0 {
			cut = MaxDoc
		}
		body = body[:cut] + fmt.Sprintf("\n... [truncated at %d bytes]", MaxDoc)
	}
	return strings.TrimRight(body, "\n"), true
}

// Message renders docs as the block a file tool appends to a read's output. It
// names each file, so the model knows which directory's rules it is reading,
// and closes the block so the file's own text stays easy to tell apart.
func Message(docs []Doc) string {
	if len(docs) == 0 {
		return ""
	}
	var b strings.Builder
	b.WriteString("\n\n[Project instructions for this file, from the repository rather than from the user. Follow them for work in their directory.]")
	for _, d := range docs {
		fmt.Fprintf(&b, "\n\n--- %s ---\n%s", d.Path, d.Body)
	}
	b.WriteString("\n\n[End of project instructions.]")
	return b.String()
}
