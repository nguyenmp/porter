#!/usr/bin/env python3
"""Run evals against porter's one-shot CLI.

For each provider it starts one `porter server` (or uses a running one), runs
every case a few times, scores each run, and appends one row per run to a
results file. Use report.py to turn the rows into a table.

The CLI's stdout is read line by line as the run happens, so token and timing
numbers reflect when the model actually produced output. Reading it after the
process exits would stamp every line with the same time.

Read providers.yaml for what a provider entry needs. No third-party packages.
"""

import argparse
import datetime
import json
import os
import platform
import re
import signal
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def load_config(path):
    """Read providers/case config as YAML if available, else JSON."""
    text = open(path).read()
    try:
        import yaml

        return yaml.safe_load(text)
    except ImportError:
        return json.loads(text)


# ---------------------------------------------------------------------------
# Objective checkers. Each name a case can set in its `check` field maps to two
# functions: one builds the ground truth once per provider, and one scores a
# single answer. The scorer takes (text, state) and returns (ok, detail).
# ---------------------------------------------------------------------------

def _time_state():
    return {"now": datetime.datetime.now()}


# _DATE_PATTERNS find a calendar date in an answer. Order matters: the most
# exact formats come first, so "September 10, 2026" is not read as a bare
# number.
_DATE_PATTERNS = [
    (r"\b(\d{4}-\d{1,2}-\d{1,2})\b", ("%Y-%m-%d",)),
    (r"\b([A-Z][a-z]+ \d{1,2},? \d{4})\b", ("%B %d, %Y", "%B %d %Y")),
    (r"\b(\d{1,2} [A-Z][a-z]+ \d{4})\b", ("%d %B %Y",)),
    (r"\b([A-Z][a-z]+ \d{1,2})\b", ("%B %d",)),
]

# _TIME_PATTERNS find a clock time. A 24-hour match has no AM/PM, so the
# 12-hour patterns cannot claim it first.
_TIME_PATTERNS = [
    (r"\b(\d{1,2}:\d{2}:\d{2}\s*[AaPp]\.?[Mm]\.?)", ("%I:%M:%S %p", "%I:%M:%S%p")),
    (r"\b(\d{1,2}:\d{2}\s*[AaPp]\.?[Mm]\.?)", ("%I:%M %p", "%I:%M%p")),
    (r"\b(\d{1,2}:\d{2}:\d{2})\b", ("%H:%M:%S",)),
    (r"\b(\d{1,2}:\d{2})\b", ("%H:%M",)),
]


def _parse_with(raw, formats):
    """Parse raw with the first format that fits, or return None."""
    cleaned = raw.replace(".", "").strip()
    for fmt in formats:
        for candidate in (raw.strip(), cleaned):
            try:
                return datetime.datetime.strptime(candidate, fmt), fmt
            except ValueError:
                continue
    return None, None


def _find_date(text, now):
    """The date an answer names, or today's date if it names none."""
    for regex, formats in _DATE_PATTERNS:
        for m in re.finditer(regex, text):
            parsed, fmt = _parse_with(m.group(1), formats)
            if parsed is None:
                continue
            if "%Y" not in fmt:
                parsed = parsed.replace(year=now.year)
            return parsed.date(), m.group(1)
    return now.date(), None


def _find_times(text):
    """Every clock time an answer names, as (hour, minute, second, raw)."""
    found = []
    for regex, formats in _TIME_PATTERNS:
        for m in re.finditer(regex, text):
            parsed, _ = _parse_with(m.group(1), formats)
            if parsed is not None:
                found.append((parsed.hour, parsed.minute, parsed.second, m.group(1)))
        if found:
            # The first pattern that matches wins, so a 24-hour time is not
            # also read as a 12-hour one.
            break
    return found


def _time_check(text, state):
    now = state["now"]
    tolerance_min = 6
    tolerance = datetime.timedelta(minutes=tolerance_min)

    day, date_raw = _find_date(text, now)
    times = _find_times(text)
    if not times:
        return False, "no clock time found in answer"

    # An answer may give UTC while the run happened in local time, or the
    # reverse. Read the clock time as local, as UTC, and as local with a UTC
    # offset, then accept whichever lands closest. A wider set of zones would
    # let almost any hour match one, which would make the check meaningless.
    utc_offset = now.astimezone().utcoffset() or datetime.timedelta(0)
    shifts = [datetime.timedelta(0), -utc_offset, utc_offset]
    best = None
    for hour, minute, second, raw in times:
        try:
            named = datetime.datetime.combine(
                day, datetime.time(hour, minute, second))
        except ValueError:
            continue
        for shift in shifts:
            candidate = named - shift
            diff = abs(candidate - now)
            if best is None or diff < best[0]:
                best = (diff, raw, candidate)

    if best is None:
        return False, "no usable clock time in answer"

    diff, raw, candidate = best
    when = "%s %s" % (date_raw, raw) if date_raw else raw
    if diff > tolerance:
        return False, ("answer says %s (%s), which is %.0f minutes from now"
                       % (when, candidate.strftime("%Y-%m-%d %H:%M:%S"),
                          diff.total_seconds() / 60))
    return True, "answer says %s, within %d min of now" % (when, tolerance_min)


def _uname_state():
    try:
        out = subprocess.run(["uname", "-srm"], capture_output=True, text=True,
                             timeout=10).stdout.strip()
    except Exception:  # non-POSIX host; fall back to Python's own report
        out = "%s|%s" % (platform.system(), platform.machine())
    parts = out.split()
    # Map each uname token to words we accept in the model's answer.
    allowed = []
    osname = parts[0] if parts else ""
    if osname == "Darwin":
        allowed += ["darwin", "macos", "mac os", "mac", "apple"]
    elif osname == "Linux":
        allowed += ["linux"]
    arch = ""
    for part in parts[1:]:
        if part.lower() in ("x86_64", "amd64", "arm64", "aarch64"):
            arch = part
            break
    arch_low = arch.lower()
    if "arm" in arch_low:
        allowed += ["arm64", "arm", "aarch64", "apple silicon", "m-series",
                    "m1", "m2", "m3", "m4"]
    elif "x86_64" in arch_low or "amd64" in arch_low:
        allowed += ["x86_64", "amd64", "intel"]
    return {"allowed": allowed, "host": out}


def _uname_check(text, state):
    low = text.lower()
    # Pick the first word that appears at a word boundary, so "mac" does not
    # match inside "machine".
    found = [a for a in state["allowed"]
             if re.search(r"\b%s\b" % re.escape(a), low)]
    if len(found) < 2:  # need the OS and an arch
        return False, ("answer lacks machine identity; want any of %s, host is %s"
                       % (state["allowed"], state["host"]))
    return True, "matched %s" % found


CHECKERS = {
    "time": (_time_state, _time_check),
    "uname": (_uname_state, _uname_check),
}


# ---------------------------------------------------------------------------
# JSONL parsing: turns the CLI's stdout, with the time each line arrived, into
# metrics.
# ---------------------------------------------------------------------------

def parse_lines(timed_lines, started):
    """timed_lines is a list of (monotonic_ts, json_obj). Started is the
    monotonic time the run began."""
    m = {
        "final_text": "",
        "tokens": {"input": 0, "output": 0,
                   "cached_input": 0, "uncached_input": 0},
        "tools": [],
        "first_token_s": None,
        "gen_seconds": 0.0,
    }
    req_start = None  # first delta of the current model request
    req_end = None    # its last token, or the usage event that closed it

    def close_req():
        nonlocal req_start, req_end
        if req_start is not None and req_end is not None and req_end > req_start:
            m["gen_seconds"] += req_end - req_start
        req_start = None
        req_end = None

    for ts, obj in timed_lines:
        typ = obj.get("type")
        kind = obj.get("kind")

        if typ in ("message_delta", "reasoning_delta"):
            if m["first_token_s"] is None:
                m["first_token_s"] = ts - started
            if req_start is None:
                req_start = ts
            req_end = ts
        elif typ == "message":
            m["final_text"] = obj.get("content") or ""
            if req_start is None:
                req_start = ts
            req_end = ts
        elif typ == "usage":
            m["tokens"]["input"] += obj.get("input_tokens") or 0
            m["tokens"]["output"] += obj.get("output_tokens") or 0
            m["tokens"]["cached_input"] += obj.get("cached_input_tokens") or 0
            m["tokens"]["uncached_input"] += obj.get("uncached_input_tokens") or 0
            # Usage closes out the request it belongs to.
            req_end = ts
            close_req()

        if kind == "tool_result":
            # Tool time is not model time; close the request before recording.
            close_req()
            result = obj.get("result") or ""
            elapsed = (obj.get("finished_at") or 0) - (obj.get("started_at") or 0)
            m["tools"].append({
                "name": obj.get("name"),
                "ok": _tool_ok(obj.get("name"), result, obj.get("timed_out")),
                "elapsed_ms": max(0, elapsed),
            })

    close_req()
    return m


def _tool_ok(name, result, timed_out):
    """Read whether one tool call worked.

    The shell tool appends an `exit code: N` line, so its result says outright
    whether the command succeeded. Every other tool has no exit code, so a
    non-empty result counts as success.
    """
    if timed_out:
        return False
    if name == "shell":
        codes = re.findall(r"[Ee]xit code:\s*(-?\d+)", result)
        if codes:
            return codes[-1] == "0"
        return bool(result.strip())
    return bool(result.strip())


# ---------------------------------------------------------------------------
# Running porter.
# ---------------------------------------------------------------------------

def find_porter(explicit):
    if explicit:
        return explicit
    local = os.path.join(HERE, "..", "porter-macos")
    if os.path.exists(local):
        return local
    return "porter"


# LOCAL_HOSTS are hosts that need no API key. A model server on this machine,
# such as a mock or a local proxy, checks its key however it wants, or not at
# all.
LOCAL_HOSTS = ("localhost", "127.0.0.1", "::1", "0.0.0.0", "host.docker.internal")


def base_url_of(env):
    """The gateway base URL this provider's server will use."""
    return env.get("PORTER_BASE_URL") or "https://api.openai.com/v1"


def needs_key(base_url):
    """Whether a gateway at this URL needs an API key."""
    from urllib.parse import urlparse

    host = (urlparse(base_url).hostname or "").lower()
    return host not in LOCAL_HOSTS


def auth_problem(name, env):
    """Why this provider cannot authenticate, or None if it can.

    A server started without a key still runs, but every request fails with a
    401. One config mistake would then fail a whole grid of runs while looking
    like bad model results, so say it up front instead.
    """
    base_url = base_url_of(env)
    if not needs_key(base_url):
        return None
    if env.get("PORTER_API_KEY") or os.environ.get("PORTER_API_KEY"):
        return None
    return ("provider %s: no PORTER_API_KEY in the environment, and %s is not a "
            "local address, so every request will fail with 401. Set the key "
            "(export PORTER_API_KEY=...) or put it in this entry's env block."
            % (name, base_url))


def env_of(provider):
    """A provider's env block, checked to hold plain strings.

    YAML turns an unquoted JSON value into a map, and a subprocess cannot use a
    map as an environment value. Failing here names the provider and the
    setting, instead of letting the server start with the wrong config.
    """
    out = {}
    for key, value in (provider.get("env") or {}).items():
        if not isinstance(value, str):
            sys.exit("provider %s: env.%s must be a string, got %s (%r)"
                     % (provider["name"], key, type(value).__name__, value))
        out[key] = value
    return out


def wait_ready(addr, timeout=60):
    import urllib.request

    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            req = urllib.request.Request("http://%s/" % addr, method="HEAD")
            with urllib.request.urlopen(req, timeout=2) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(0.25)
    return False


def start_server(porter, provider, work_dir):
    os.makedirs(work_dir, exist_ok=True)
    env = dict(os.environ)
    env.update(env_of(provider))
    env["PORTER_ADDR"] = provider["addr"]
    log = open(os.path.join(work_dir, "server.log"), "w")
    proc = subprocess.Popen([porter, "server"], cwd=work_dir, env=env,
                            stdout=log, stderr=subprocess.STDOUT,
                            start_new_session=True)
    if not wait_ready(provider["addr"]):
        stop_server(proc)
        raise RuntimeError("server did not start for %s (see %s/server.log)"
                           % (provider["name"], work_dir))
    return proc


def stop_server(proc):
    if proc is None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except Exception:
        try:
            proc.terminate()
        except Exception:
            return
    try:
        proc.wait(timeout=10)
    except Exception:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            pass


def run_one(porter, server_url, work_dir, prompt, timeout):
    """Run one prompt and return a metrics dict.

    Stdout is read while the run happens, so each JSONL line is stamped with
    the moment it arrived. That is what makes the tokens/second number mean
    anything.
    """
    env = dict(os.environ)
    env["PORTER_SERVER_URL"] = server_url
    started = time.monotonic()
    proc = subprocess.Popen([porter, prompt], cwd=work_dir, env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, bufsize=1, start_new_session=True)

    timed_lines = []
    timed_out = False

    def kill():
        nonlocal timed_out
        timed_out = True
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            pass

    timer = threading.Timer(timeout, kill)
    timer.start()
    try:
        for line in proc.stdout:
            ts = time.monotonic()
            line = line.strip()
            if not line:
                continue
            try:
                timed_lines.append((ts, json.loads(line)))
            except ValueError:
                continue
        stderr = proc.stderr.read()
        proc.wait()
    finally:
        timer.cancel()

    wall = time.monotonic() - started
    m = parse_lines(timed_lines, started)
    m["end_to_end_s"] = wall
    m["session_id"] = _session_from_stderr(stderr)
    if timed_out:
        m["error"] = "timed out after %ds" % timeout
    elif proc.returncode != 0:
        m["error"] = _clean_error(stderr)
    m["gen_tokens_per_sec"] = (
        (m["tokens"]["output"] / m["gen_seconds"])
        if m["gen_seconds"] > 0 and m["tokens"]["output"] else None)
    return m


def _session_from_stderr(stderr):
    for line in stderr.splitlines():
        if "session " in line:
            return line.split("session ", 1)[1].strip()
    return ""


def _clean_error(stderr):
    """Pull the failure reason from the CLI's stderr.

    The first stderr line is always the session banner ("porter: session X"),
    which is not an error. Skip it and prefer a line that names a failure.
    """
    lines = [l[len("porter:"):].strip() for l in stderr.splitlines()
             if l.startswith("porter:")]
    lines = [l for l in lines if not l.startswith("session ")]
    for line in lines:
        if "fail" in line.lower() or "error" in line.lower():
            return line
    return lines[-1] if lines else stderr.strip()


# ---------------------------------------------------------------------------
# Main loop.
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--providers", default=os.path.join(HERE, "providers.yaml"))
    ap.add_argument("--cases-dir", default=os.path.join(HERE, "cases"))
    ap.add_argument("--out", default=os.path.join(HERE, "results.jsonl"))
    ap.add_argument("--porter", default="", help="path to the porter binary")
    ap.add_argument("--trials", type=int, default=0,
                    help="override each provider's trial count")
    ap.add_argument("--timeout", type=int, default=300,
                    help="per-run timeout in seconds")
    ap.add_argument("--only", default="",
                    help="run only providers/cases matching this substring")
    ap.add_argument("--no-server", action="store_true",
                    help="use running servers instead of starting them")
    args = ap.parse_args()

    config = load_config(args.providers)
    providers = config.get("providers", [])
    if not providers:
        sys.exit("no providers found in %s" % args.providers)

    case_names = sorted(
        d for d in os.listdir(args.cases_dir)
        if os.path.isdir(os.path.join(args.cases_dir, d))
        and os.path.exists(os.path.join(args.cases_dir, d, "case.yaml")))
    if not case_names:
        sys.exit("no cases (folders with case.yaml) in %s" % args.cases_dir)

    porter = find_porter(args.porter)

    # Check every provider's auth before running anything. One missing key
    # would otherwise fail every row of the grid with the same 401, which reads
    # like a model result instead of a setup mistake.
    problems = []
    for provider in providers:
        if args.only and args.only not in provider["name"]:
            continue
        problem = auth_problem(provider["name"], env_of(provider))
        if problem:
            problems.append(problem)
    if problems:
        sys.exit("\n".join(problems))

    out_f = open(args.out, "a")

    for provider in providers:
        name = provider["name"]
        if args.only and args.only not in name:
            continue
        trials = args.trials or provider.get("trials", 3)
        state_cache = {}  # check name -> ground truth, built once per provider
        server_url = env_of(provider).get("PORTER_SERVER_URL")
        proc = None
        work_dir = os.path.join(HERE, ".eval-work", name)
        os.makedirs(work_dir, exist_ok=True)

        if provider.get("external") or args.no_server:
            if not server_url:
                sys.exit("provider %s: external mode needs env.PORTER_SERVER_URL"
                         % name)
            print("== %s  (using running server %s)" % (name, server_url))
        else:
            env = env_of(provider)
            auth = ("key from %s" % ("env block" if env.get("PORTER_API_KEY")
                                     else "PORTER_API_KEY"))
            if not needs_key(base_url_of(env)):
                auth = "no key needed (local address)"
            print("== %s  (starting server on %s; %s)"
                  % (name, provider["addr"], auth))
            proc = start_server(porter, provider, work_dir)
            server_url = "http://%s" % provider["addr"]

        try:
            for case in case_names:
                if args.only and args.only not in case:
                    continue
                case_cfg = load_config(
                    os.path.join(args.cases_dir, case, "case.yaml"))
                prompt = case_cfg["prompt"]
                check = case_cfg.get("check")
                checker = CHECKERS.get(check)
                if checker and check not in state_cache:
                    state_cache[check] = checker[0]()

                for trial in range(1, trials + 1):
                    metrics = run_one(porter, server_url, work_dir, prompt,
                                      args.timeout)
                    row = {
                        "provider": name,
                        "case": case,
                        "trial": trial,
                        "ts": datetime.datetime.now().isoformat(),
                    }
                    row.update(metrics)

                    if "error" in metrics:
                        row["verdict"] = "fail"
                    elif checker:
                        ok, detail = checker[1](metrics.get("final_text", ""),
                                                state_cache[check])
                        row["verdict"] = "pass" if ok else "fail"
                        row["detail"] = detail
                    else:
                        row["verdict"] = "pass"
                        row["detail"] = "no checker"

                    tools = metrics.get("tools", [])
                    row["tool_summary"] = {
                        "calls": len(tools),
                        "ok": sum(1 for t in tools if t["ok"]),
                    }
                    out_f.write(json.dumps(row) + "\n")
                    out_f.flush()
                    print("  %-22s t%-2d %-4s %7.1fs e2e  %6s tok/s  %d tool(s)"
                          % (case, trial, row["verdict"].upper(),
                             metrics.get("end_to_end_s") or 0,
                             _fmt(metrics.get("gen_tokens_per_sec")),
                             len(tools)))
        finally:
            stop_server(proc)

    out_f.close()
    print("done; rows in %s" % args.out)


def _fmt(v):
    return "-" if v is None else "%.0f" % v


if __name__ == "__main__":
    main()
