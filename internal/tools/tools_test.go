package tools

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
	"testing"
	"time"

	"porter/internal/api"
	"porter/internal/handoff"
	"porter/internal/humanize"
	"porter/internal/remoteedit"
)

func run(ctx context.Context, p Provider, name string, args []byte) (string, error) {
	stream, err := p.Run(ctx, name, args)
	if err != nil {
		return "", err
	}
	b, rerr := io.ReadAll(stream)
	_ = stream.Close()
	if rerr != nil {
		return "", rerr
	}
	return string(b), nil
}

func TestShellRunsCommand(t *testing.T) {
	res, err := run(context.Background(), NewDispatcher(), "shell", []byte(`{"command":"echo hi"}`))
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if !strings.Contains(res, "exit code: 0") || !strings.Contains(res, "hi") {
		t.Errorf("unexpected result: %q", res)
	}
}

func TestShellReportsExitCode(t *testing.T) {
	res, err := run(context.Background(), NewDispatcher(), "shell", []byte(`{"command":"exit 3"}`))
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if !strings.Contains(res, "exit code: 3") {
		t.Errorf("unexpected result: %q", res)
	}
}

func TestShellCombinesStdoutAndStderr(t *testing.T) {
	res, err := run(context.Background(), NewDispatcher(), "shell", []byte(`{"command":"echo out; echo err >&2"}`))
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if !strings.Contains(res, "out") || !strings.Contains(res, "err") {
		t.Errorf("result should include stdout and stderr; got: %q", res)
	}
}

func TestDispatcherUnknownTool(t *testing.T) {
	if _, err := run(context.Background(), NewDispatcher(), "nope", []byte("{}")); err == nil {
		t.Fatal("expected error for unknown tool")
	}
}

func TestShellEmptyCommand(t *testing.T) {
	if _, err := run(context.Background(), NewDispatcher(), "shell", []byte(`{"command":"  "}`)); err == nil {
		t.Fatal("expected error for empty command")
	}
}

func TestShellRespectsCancellation(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	stream, err := NewDispatcher().Run(ctx, "shell", []byte(`{"command":"sleep 5"}`))
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	go func() {
		_, _ = io.Copy(io.Discard, stream)
		_ = stream.Close()
	}()
	cancel()
	// Reading after cancellation must terminate rather than hang. Any of
	// "(interrupt)" / EOF / "exit code" end-of-stream satisfies correctness.
	b := make([]byte, 1)
	_, _ = stream.Read(b)
}

// TestShellBackgroundDescendantDoesNotBlock is the regression test for the
// hung-process bug. The command backgrounds a descendant that inherits the
// output descriptor; the direct child finishes immediately. With the old pipe,
// completion meant "every process that inherited the pipe has exited", so the
// backgrounded descendant kept EOF from ever arriving and the agent loop hung
// even after the command itself was done. With the file-backed stream,
// completion is decided by the direct child exiting, so the stream must end
// while the backgrounded process is still running.
func TestShellBackgroundDescendantDoesNotBlock(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	stream, err := NewDispatcher().Run(ctx, "shell", []byte(`{"command":"sleep 5 & echo done"}`))
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	defer stream.Close()
	defer exec.Command("pkill", "-f", "sleep 5").Run() // tidy any stray survivor

	done := make(chan error, 1)
	go func() {
		_, rerr := io.ReadAll(stream)
		done <- rerr
	}()

	// Must return long before the 5s backgrounded sleep exits. Under the pipe
	// implementation this blocked until the survivor died; under the file
	// implementation it returns when the direct child exits (~instantly).
	select {
	case err := <-done:
		if err != nil {
			t.Fatalf("read: %v", err)
		}
	case <-time.After(1500 * time.Millisecond):
		t.Fatal("stream did not terminate: a backgrounded descendant kept the output open")
	}
}

// TestShellBackgroundRecipeKeepsLateOutput follows the recipe in the shell tool
// description: start the job in a call that returns at once, with its output
// and exit code redirected to real files, then read those files from a later
// call. The job waits on a gate the test opens after the starting call has
// ended, so every byte it prints arrives later, where the tool's own temp file
// would have dropped it. Nothing here uses wait: the job is not a child of the
// polling shell.
func TestShellBackgroundRecipeKeepsLateOutput(t *testing.T) {
	dir := t.TempDir()
	gate := filepath.Join(dir, "gate")
	logPath := filepath.Join(dir, "job.log")
	statusPath := filepath.Join(dir, "job.status")
	pidPath := filepath.Join(dir, "job.pid")

	start := fmt.Sprintf(
		`nohup sh -c 'while [ ! -f %s ]; do sleep 0.05; done; sh -c "echo late output; exit 7"; echo $? > %s' `+
			`> %s 2>&1 & echo $! > %s`,
		gate, statusPath, logPath, pidPath)
	args, err := json.Marshal(map[string]string{"command": start})
	if err != nil {
		t.Fatalf("marshal start command: %v", err)
	}
	if _, err := run(context.Background(), NewDispatcher(), "shell", args); err != nil {
		t.Fatalf("start call: %v", err)
	}

	pid, err := os.ReadFile(pidPath)
	if err != nil {
		t.Fatalf("read pid file: %v", err)
	}
	pidNum, err := strconv.Atoi(strings.TrimSpace(string(pid)))
	if err != nil {
		t.Fatalf("pid file %q: %v", pid, err)
	}
	t.Cleanup(func() {
		if _, err := os.Stat(statusPath); err != nil {
			_ = exec.Command("kill", "-9", strconv.Itoa(pidNum)).Run()
		}
	})
	if err := syscall.Kill(pidNum, 0); err != nil {
		t.Fatalf("job %d not running after the start call returned: %v", pidNum, err)
	}

	if err := os.WriteFile(gate, nil, 0o644); err != nil {
		t.Fatalf("open gate: %v", err)
	}
	deadline := time.Now().Add(15 * time.Second)
	for {
		if _, err := os.Stat(statusPath); err == nil {
			break
		}
		if time.Now().After(deadline) {
			out, _ := os.ReadFile(logPath)
			t.Fatalf("job did not finish within 15s; log: %q", out)
		}
		time.Sleep(20 * time.Millisecond)
	}

	log, err := os.ReadFile(logPath)
	if err != nil {
		t.Fatalf("read log: %v", err)
	}
	if !strings.Contains(string(log), "late output") {
		t.Errorf("log = %q, want the output the job wrote after the call ended", log)
	}
	status, err := os.ReadFile(statusPath)
	if err != nil {
		t.Fatalf("read status: %v", err)
	}
	if got := strings.TrimSpace(string(status)); got != "7" {
		t.Errorf("status file = %q, want the job's exit code 7", got)
	}
}

// TestShellCancelUnblocksStream verifies cancellation still terminates the
// stream promptly, so a user can abort a hung or runaway command without the
// agent loop waiting forever.
func TestShellCancelUnblocksStream(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	stream, err := NewDispatcher().Run(ctx, "shell", []byte(`{"command":"sleep 5"}`))
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	defer stream.Close()

	done := make(chan error, 1)
	go func() {
		_, rerr := io.ReadAll(stream)
		done <- rerr
	}()
	time.Sleep(200 * time.Millisecond)
	cancel()

	select {
	case <-done:
		// Stream terminated on cancel.
	case <-time.After(3 * time.Second):
		t.Fatal("stream did not terminate after cancel")
	}
}

// TestShellStopKillsWholeTree verifies that cancelling a compound command stops
// the shell AND its forked children, not just the direct child. The command
// starts a foreground `sleep 5` as its real work; the process group must be
// gone (no survivor) after cancel.
func TestShellStopKillsWholeTree(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	stream, err := NewDispatcher().Run(ctx, "shell", []byte(`{"command":"sleep 5"}`))
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	defer stream.Close()

	done := make(chan error, 1)
	go func() {
		_, rerr := io.ReadAll(stream)
		done <- rerr
	}()
	time.Sleep(200 * time.Millisecond)
	cancel()

	select {
	case <-done:
		// Stream terminated: the whole tree was killed and the stream closed.
	case <-time.After(3 * time.Second):
		t.Fatal("stream did not terminate after cancel")
	}
	time.Sleep(200 * time.Millisecond) // allow reparenting to settle
	if out, err := exec.Command("pgrep", "-f", "^sleep 5$").Output(); err == nil {
		t.Fatalf("survivor still running after cancel: %q", strings.TrimSpace(string(out)))
	}
}

// TestLoadSkillReturnsSkillBody verifies the load_skill tool reads a discovered
// skill's SKILL.md and streams it back with the conventional exit-status line.
func TestLoadSkillReturnsSkillBody(t *testing.T) {
	dir := t.TempDir()
	path := dir + "/SKILL.md"
	if err := os.WriteFile(path, []byte("# my skill\n\ninstructions here\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	d := NewDispatcherWithSkills([]api.Skill{{Name: "my-skill", Description: "desc", Path: path}})

	// The tool is declared to the model when skills are present: shell + the
	// file editing tools + load_skill.
	defs := d.Defs()
	if len(defs) != 6 {
		t.Fatalf("Defs = %d tools, want shell + file tools + load_skill", len(defs))
	}
	if got := defs[len(defs)-1].Function.Name; got != "load_skill" {
		t.Errorf("last tool = %q, want load_skill", got)
	}
	if !strings.Contains(defs[len(defs)-1].Function.Description, "my-skill: desc") {
		t.Errorf("load_skill description missing skill metadata: %q", defs[len(defs)-1].Function.Description)
	}

	res, err := run(context.Background(), d, "load_skill", []byte(`{"name":"my-skill"}`))
	if err != nil {
		t.Fatalf("Run load_skill: %v", err)
	}
	if !strings.Contains(res, "# my skill") || !strings.Contains(res, "instructions here") {
		t.Errorf("load_skill result = %q, want the skill body", res)
	}
	if !strings.Contains(res, "exit code: 0") {
		t.Errorf("load_skill result missing exit-status line: %q", res)
	}
}

// TestLoadSkillReturnsBuiltinBody verifies a built-in skill (sentinel path,
// e.g. the plain-language prompt compiled into the binary) is served from
// memory with no file on disk.
func TestLoadSkillReturnsBuiltinBody(t *testing.T) {
	d := NewDispatcherWithSkills([]api.Skill{humanize.BuiltinSkill()})

	// The sentinel path is not a real file: resolution must come from memory.
	if _, err := os.Stat(humanize.BuiltinSkill().Path); !os.IsNotExist(err) {
		t.Fatalf("built-in sentinel path %q unexpectedly exists on disk", humanize.BuiltinSkill().Path)
	}

	res, err := run(context.Background(), d, "load_skill", []byte(`{"name":"`+humanize.SkillName+`"}`))
	if err != nil {
		t.Fatalf("Run load_skill (built-in): %v", err)
	}
	if res != humanize.Prompt()+"\nexit code: 0\n" {
		t.Errorf("built-in load_skill result mismatch:\n got %q\nwant prompt body with exit line", res)
	}
	if !strings.Contains(res, "Apply plain language") {
		t.Errorf("built-in load_skill result missing prompt body: %q", res)
	}
	if !strings.Contains(res, "exit code: 0") {
		t.Errorf("built-in load_skill result missing exit-status line: %q", res)
	}
}

// TestLoadSkillReturnsRemoteEditBuiltinBody verifies the built-in
// editing-remote-files skill (a second compiled-in skill, like the
// plain-language one) is served from memory with no file on disk.
func TestLoadSkillReturnsRemoteEditBuiltinBody(t *testing.T) {
	d := NewDispatcherWithSkills([]api.Skill{remoteedit.BuiltinSkill()})

	// The sentinel path is not a real file: resolution must come from memory.
	if _, err := os.Stat(remoteedit.BuiltinSkill().Path); !os.IsNotExist(err) {
		t.Fatalf("built-in sentinel path %q unexpectedly exists on disk", remoteedit.BuiltinSkill().Path)
	}

	res, err := run(context.Background(), d, "load_skill", []byte(`{"name":"`+remoteedit.SkillName+`"}`))
	if err != nil {
		t.Fatalf("Run load_skill (built-in): %v", err)
	}
	if res != remoteedit.Prompt()+"\nexit code: 0\n" {
		t.Errorf("built-in load_skill result mismatch:\n got %q\nwant prompt body with exit line", res)
	}
	if !strings.Contains(res, "How to edit files on a remote host") {
		t.Errorf("built-in load_skill result missing prompt body: %q", res)
	}
	if !strings.Contains(res, "exit code: 0") {
		t.Errorf("built-in load_skill result missing exit-status line: %q", res)
	}
}

// TestLoadSkillReturnsHandoffBuiltinBody verifies the built-in handoff skill
// (a third compiled-in skill, like plain-language and editing-remote-files) is
// served from memory with no file on disk.
func TestLoadSkillReturnsHandoffBuiltinBody(t *testing.T) {
	d := NewDispatcherWithSkills([]api.Skill{handoff.BuiltinSkill()})

	// The sentinel path is not a real file: resolution must come from memory.
	if _, err := os.Stat(handoff.BuiltinSkill().Path); !os.IsNotExist(err) {
		t.Fatalf("built-in sentinel path %q unexpectedly exists on disk", handoff.BuiltinSkill().Path)
	}

	res, err := run(context.Background(), d, "load_skill", []byte(`{"name":"`+handoff.SkillName+`"}`))
	if err != nil {
		t.Fatalf("Run load_skill (built-in): %v", err)
	}
	if res != handoff.Prompt()+"\nexit code: 0\n" {
		t.Errorf("built-in load_skill result mismatch:\n got %q\nwant prompt body with exit line", res)
	}
	if !strings.Contains(res, "Could you write a summary using plain-language?") {
		t.Errorf("built-in load_skill result missing prompt body: %q", res)
	}
	if !strings.Contains(res, "exit code: 0") {
		t.Errorf("built-in load_skill result missing exit-status line: %q", res)
	}
}

// TestLoadSkillUnknownSkill verifies an unknown skill name is reported as an
// error rather than silently returning nothing.
func TestLoadSkillUnknownSkill(t *testing.T) {
	d := NewDispatcher()
	if _, err := run(context.Background(), d, "load_skill", []byte(`{"name":"nope"}`)); err == nil {
		t.Fatal("expected error for unknown skill")
	}
}

// TestDispatcherWithoutSkillsExposesBaseTools verifies a dispatcher with no
// skills declares shell plus the file editing tools (every environment can
// serve them) but not load_skill (a provider can't serve it without skills).
func TestDispatcherWithoutSkillsExposesBaseTools(t *testing.T) {
	defs := NewDispatcher().Defs()
	want := []string{"shell", ReadLinesTool, LineInsertTool, LineReplaceTool, StringReplace}
	if len(defs) != len(want) {
		t.Fatalf("Defs without skills = %d tools %+v, want %d", len(defs), defs, len(want))
	}
	for i, w := range want {
		if defs[i].Function.Name != w {
			t.Errorf("tool %d = %q, want %q", i, defs[i].Function.Name, w)
		}
	}
}

// TestDispatcherEnvironmentEmpty verifies a local dispatcher reports no
// environment context (discovery is the connected client's job).
func TestDispatcherEnvironmentEmpty(t *testing.T) {
	if got := NewDispatcher().Environment(); got != "" {
		t.Errorf("Dispatcher.Environment() = %q, want empty", got)
	}
}

// TestShellDefSteersToDedicatedTools locks in the shell description's job
// boundaries: it must point at the file tools and the MCP-provided web_fetch
// instead of advertising edits and network calls as shell's job.
func TestShellDefSteersToDedicatedTools(t *testing.T) {
	defs := NewDispatcher().Defs()
	var shellDesc string
	for _, d := range defs {
		if d.Function.Name == "shell" {
			shellDesc = d.Function.Description
		}
	}
	if shellDesc == "" {
		t.Fatal("shell tool not declared")
	}
	for _, want := range []string{"read_with_line_numbers", "line_insert", "line_replace", "string_replace", "web_fetch", "curl"} {
		if !strings.Contains(shellDesc, want) {
			t.Errorf("shell description missing %q:\n%s", want, shellDesc)
		}
	}
	for _, stale := range []string{"inspect or edit files", "make network calls"} {
		if strings.Contains(shellDesc, stale) {
			t.Errorf("shell description still advertises %q:\n%s", stale, shellDesc)
		}
	}
}

// TestShellDefExplainsBackgrounding locks in the recipe the description gives
// for a long job: send output and exit code to real files, then check the job
// with kill -0 from a later call. It also keeps the reasons in the text: wait
// cannot reach a process a later shell did not start, timeout cannot run the
// wait builtin, and cancelling a call kills every process the call started.
func TestShellDefExplainsBackgrounding(t *testing.T) {
	var shellDesc string
	for _, d := range NewDispatcher().Defs() {
		if d.Function.Name == "shell" {
			shellDesc = d.Function.Description
		}
	}
	if shellDesc == "" {
		t.Fatal("shell tool not declared")
	}
	for _, want := range []string{"nohup", "kill -0", "shell builtin", "temporary file", "every process the call started"} {
		if !strings.Contains(shellDesc, want) {
			t.Errorf("shell description missing %q:\n%s", want, shellDesc)
		}
	}
}
