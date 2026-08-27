// Package exec discovers the environment an execution provider runs in: the
// system it's on, the working directory, files there, and the skills it can
// load. A connected client (the REPL today) reports this context to the server,
// which injects it into the model and exposes a load_skill tool backed by the
// reported skills, so the model knows where commands will run and what skills
// exist without guessing.
package exec

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"sort"
	"strings"

	"porter/internal/api"
)

// maxFiles bounds the file listing included in the context so it stays a small
// fixed-size hint (enough to push toward README.md and other relevant files)
// rather than a full directory dump.
const maxFiles = 100

// System describes the OS the execution provider runs on, in a form that helps
// the model pick the right tooling (curl vs wget, GNU vs BSD userland).
func System() string {
	s := runtime.GOOS + "/" + runtime.GOARCH
	switch runtime.GOOS {
	case "darwin":
		s += " (macOS — BSD userland: grep/sed/find/stat are BSD variants, use `curl`, `shasum`; NOT GNU coreutils)"
	case "linux":
		s += " (Linux — GNU userland; `curl` or `wget` may be present)"
	case "windows":
		s += " (Windows)"
	default:
		s += " (" + runtime.GOOS + ")"
	}
	return s
}

// ListFiles returns the sorted names of the entries in cwd (directories with a
// trailing slash), bounded to maxFiles.
func ListFiles(cwd string) ([]string, error) {
	entries, err := os.ReadDir(cwd)
	if err != nil {
		return nil, fmt.Errorf("list %s: %w", cwd, err)
	}
	names := make([]string, 0, len(entries))
	for _, e := range entries {
		name := e.Name()
		if e.IsDir() {
			name += "/"
		}
		names = append(names, name)
	}
	sort.Strings(names)
	if len(names) > maxFiles {
		names = names[:maxFiles]
	}
	return names, nil
}

// Discover returns the environment context of cwd (defaulting to the process
// working directory): the system, the absolute working directory, the files
// there, and the skills available both globally and in the repo.
func Discover(cwd string) (api.ExecContext, error) {
	if cwd == "" {
		var err error
		cwd, err = os.Getwd()
		if err != nil {
			return api.ExecContext{}, fmt.Errorf("get cwd: %w", err)
		}
	}
	if abs, err := filepath.Abs(cwd); err == nil {
		cwd = abs
	}
	files, err := ListFiles(cwd)
	if err != nil {
		return api.ExecContext{}, fmt.Errorf("discover cwd files: %w", err)
	}
	return api.ExecContext{
		System: System(),
		CWD:    cwd,
		Files:  files,
		Skills: FindSkills(cwd),
	}, nil
}

// SystemMessage renders the context as the system message injected into the
// model's prompt: where commands run, what's in the working directory, and the
// skills that can be loaded. Sections that are empty are omitted.
func SystemMessage(c api.ExecContext) string {
	var b strings.Builder
	if c.System != "" {
		fmt.Fprintf(&b, "You are running on: %s\n", c.System)
	}
	if c.CWD != "" {
		fmt.Fprintf(&b, "Working directory: %s\n", c.CWD)
	}
	if len(c.Files) > 0 {
		b.WriteString("\nFiles in the working directory:\n")
		for _, f := range c.Files {
			fmt.Fprintf(&b, "- %s\n", f)
		}
	}
	if len(c.Skills) > 0 {
		b.WriteString("\nAvailable skills (load one with the load_skill tool when relevant):\n")
		for _, s := range c.Skills {
			fmt.Fprintf(&b, "- %s: %s\n", s.Name, s.Description)
		}
	}
	return strings.TrimSpace(b.String())
}

// FindSkills discovers skills under the repo root (via `git rev-parse
// --show-toplevel`, falling back to cwd when not in a git repo) and the user's
// home directory. A skill is a directory containing a SKILL.md, found at
// `<root>/.agents/skills/*/SKILL.md` or `<root>/.*/skills/*/SKILL.md` (any
// hidden directory's skills subdir). Results are deduplicated by name,
// preferring repo skills over global ones.
func FindSkills(cwd string) []api.Skill {
	seen := make(map[string]bool)
	var out []api.Skill
	for _, root := range skillRoots(cwd) {
		for _, path := range skillPaths(root) {
			s, ok := parseSkill(path)
			if !ok || seen[s.Name] {
				continue
			}
			seen[s.Name] = true
			out = append(out, s)
		}
	}
	return out
}

// skillRoots returns the directories to search for skills, most specific
// first: the repo root (or cwd when not in a git repo), the user home, and
// cwd itself when it differs from both.
func skillRoots(cwd string) []string {
	var roots []string
	if repo, err := gitRoot(cwd); err == nil && repo != "" {
		roots = append(roots, repo)
	} else if cwd != "" {
		roots = append(roots, cwd)
	}
	if home, err := os.UserHomeDir(); err == nil {
		roots = append(roots, home)
	}
	if cwd != "" && !containsPath(roots, cwd) {
		roots = append(roots, cwd)
	}
	return roots
}

// gitRoot returns the top-level directory of the git repo containing cwd.
func gitRoot(cwd string) (string, error) {
	cmd := exec.Command("git", "rev-parse", "--show-toplevel")
	cmd.Dir = cwd
	out, err := cmd.Output()
	if err != nil {
		return "", err
	}
	return strings.TrimSpace(string(out)), nil
}

// skillPaths returns the SKILL.md files under root matching the two skill
// patterns, filtering out the "." and ".." the `.*` glob also matches.
func skillPaths(root string) []string {
	var out []string
	for _, pattern := range []string{
		filepath.Join(root, ".agents", "skills", "*", "SKILL.md"),
		filepath.Join(root, ".*", "skills", "*", "SKILL.md"),
	} {
		matches, _ := filepath.Glob(pattern)
		for _, m := range matches {
			if info, err := os.Stat(m); err != nil || info.IsDir() {
				continue
			}
			rel, err := filepath.Rel(root, m)
			if err != nil {
				continue
			}
			// The `.*` component also matches "." and ".."; those are the
			// root's own and parent directories, not skill roots.
			first := strings.SplitN(rel, string(os.PathSeparator), 2)[0]
			if first == "." || first == ".." || !strings.HasPrefix(first, ".") {
				continue
			}
			out = append(out, m)
		}
	}
	return out
}

// parseSkill reads a SKILL.md and returns its metadata. The name comes from
// the frontmatter `name:` when present, else the skill directory name; the
// description from frontmatter `description:`, else the first non-blank line
// of the body.
func parseSkill(path string) (api.Skill, bool) {
	data, err := os.ReadFile(path)
	if err != nil {
		return api.Skill{}, false
	}
	name := filepath.Base(filepath.Dir(path))
	desc := ""
	if fm, ok := parseFrontmatter(string(data)); ok {
		if n, ok := fm["name"]; ok && strings.TrimSpace(n) != "" {
			name = strings.TrimSpace(n)
		}
		if d, ok := fm["description"]; ok && strings.TrimSpace(d) != "" {
			desc = strings.TrimSpace(d)
		}
	}
	if name == "" {
		return api.Skill{}, false
	}
	if desc == "" {
		desc = firstLine(string(data))
	}
	return api.Skill{Name: name, Description: desc, Path: path}, true
}

// parseFrontmatter reads the YAML-ish frontmatter at the top of a markdown
// file: a leading `---` line, key: value pairs, closed by a `---` line. Values
// are trimmed of surrounding quotes. It returns the key/value map and whether
// frontmatter was present and well-formed.
func parseFrontmatter(body string) (map[string]string, bool) {
	lines := strings.Split(body, "\n")
	if len(lines) == 0 || strings.TrimSpace(lines[0]) != "---" {
		return nil, false
	}
	fm := make(map[string]string)
	for i := 1; i < len(lines); i++ {
		line := strings.TrimSpace(lines[i])
		if line == "---" {
			return fm, true
		}
		key, val, ok := strings.Cut(line, ":")
		if !ok {
			continue
		}
		fm[strings.TrimSpace(key)] = unquote(strings.TrimSpace(val))
	}
	return nil, false
}

// unquote strips a single pair of surrounding double or single quotes.
func unquote(s string) string {
	if len(s) >= 2 {
		if (s[0] == '"' && s[len(s)-1] == '"') || (s[0] == '\'' && s[len(s)-1] == '\'') {
			return s[1 : len(s)-1]
		}
	}
	return s
}

// firstLine returns the first non-blank line of body (skipping markdown
// headings), truncated to a short description length.
func firstLine(body string) string {
	for _, l := range strings.Split(body, "\n") {
		l = strings.TrimSpace(l)
		if l == "" || strings.HasPrefix(l, "#") {
			continue
		}
		if r := []rune(l); len(r) > 100 {
			return string(r[:100]) + "…"
		}
		return l
	}
	return ""
}

// containsPath reports whether abs is in paths (comparing cleaned absolute
// paths).
func containsPath(paths []string, abs string) bool {
	for _, p := range paths {
		if p == abs {
			return true
		}
	}
	return false
}

// NOTE (later feature): skills are discovered only from the fixed roots here,
// at connect time. As we add read/write tools that move into other files or
// folders mid-session, we can re-discover skills rooted at the current
// directory (or hook into those tools) so skills appear as we navigate. Today
// the discovery happens once per provider connect, so the list can go stale if
// skills are edited or added after connecting — which is fine for now.
