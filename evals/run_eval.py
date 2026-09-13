#!/usr/bin/env python3
"""Run evals against porter's one-shot CLI.

Every (provider, case, trial) is one task in a shared thread pool, so
slow runs drain across all threads instead of pinning a per-provider
chain. Each provider starts its own `porter server` (or uses a running
one); every task scores one run and writes one row to a per-provider
results file. When all tasks finish, the per-provider files are merged
into the main results file. Use report.py to turn the rows into a table.

The CLI's stdout is read line by line as the run happens, so token and
timing numbers reflect when the model actually produced output. Reading
it after the process exits would stamp every line with the same time.

Read providers.yaml for what a provider entry needs. No third-party
packages.
"""

import argparse
import datetime
import ipaddress
import json
import os
import platform
import re
import signal
import socket
import subprocess
import sys
import threading
import time
import concurrent.futures
import shutil
import tempfile
import traceback

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
# single answer. The scorer takes (text, state, metrics) and returns (ok,
# detail). Metrics is the parsed run; a checker needs it only when it scores
# the stream rather than the wording, such as the reasoning case.
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


def _time_check(text, state, metrics=None):
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


def _uname_check(text, state, metrics=None):
    low = text.lower()
    # Pick the first word that appears at a word boundary, so "mac" does not
    # match inside "machine".
    found = [a for a in state["allowed"]
             if re.search(r"\b%s\b" % re.escape(a), low)]
    if len(found) < 2:  # need the OS and an arch
        return False, ("answer lacks machine identity; want any of %s, host is %s"
                       % (state["allowed"], state["host"]))
    return True, "matched %s" % found


def _reasoning_state():
    # Nothing to look up: this case scores the stream, not the wording.
    return {}


def _reasoning_check(text, state, metrics=None):
    """Did the endpoint stream reasoning, and did an answer still land?

    The count comes from the JSONL, not from the reply: an endpoint that drops
    reasoning returns a normal-looking answer, so the text alone cannot show
    whether any reasoning came back.
    """
    info = (metrics or {}).get("reasoning") or {}
    chars = info.get("chars") or 0
    deltas = info.get("deltas") or 0
    if not chars:
        return False, ("no reasoning text in the stream (%d delta(s)); the "
                       "endpoint is dropping it or reasoning is off" % deltas)
    if not text.strip():
        return False, ("reasoning arrived (%d chars) but the answer is empty"
                       % chars)
    return True, ("%d reasoning chars in %d delta(s), then an answer"
                  % (chars, deltas))


# _IP_LOOKUPS name services that answer "what is my public address?". Two of
# them, one per address family, so an answer in IPv6 is checked as closely as
# an answer in IPv4.
_IP_LOOKUPS = ("https://api.ipify.org", "https://api64.ipify.org")


def _public_ips():
    """This machine's public addresses, or an empty set when a lookup fails."""
    import urllib.request

    found = set()
    for url in _IP_LOOKUPS:
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                text = r.read(64).decode("utf-8", "replace").strip()
        except Exception:
            continue
        try:
            found.add(str(ipaddress.ip_address(text)))
        except ValueError:
            continue
    return found


def _local_ips():
    """The addresses this machine holds, so a private-address answer is checked
    against something real instead of a guess."""
    out = ""
    for cmd in (["ifconfig"], ["ip", "-o", "addr"]):
        try:
            out = subprocess.run(cmd, capture_output=True, text=True,
                                 timeout=10).stdout
        except Exception:
            out = ""
        if out:
            break
    found = set()
    for raw in re.findall(r"\binet6?\s+(?:addr:)?([0-9A-Fa-f:.]+)", out):
        try:
            found.add(str(ipaddress.ip_address(raw.strip("."))))
        except ValueError:
            continue
    try:  # a machine whose interface dump said nothing still knows its name
        for info in socket.getaddrinfo(socket.gethostname(), None):
            found.add(info[4][0])
    except Exception:
        pass
    return found


def _my_ip_state():
    return {"public": _public_ips(), "local": _local_ips()}


# _IP_CANDIDATE matches the two shapes an address takes: a dotted quad, or a
# run of colon-separated groups. ipaddress decides whether a match really is an
# address, so a version number like 25.6.0 is dropped instead of counted.
_IP_CANDIDATE = re.compile(r"[0-9A-Fa-f]{1,4}(?::[0-9A-Fa-f]{0,4}){2,}"
                           r"|[0-9]{1,3}(?:\.[0-9]{1,3}){3}")


def _ips_in(text):
    """Every address the answer names, ignoring ones that mean "no address"."""
    out = []
    for raw in _IP_CANDIDATE.findall(text):
        try:
            addr = ipaddress.ip_address(raw)
        except ValueError:
            continue
        if addr.is_unspecified or addr.is_loopback:
            continue  # 0.0.0.0 and 127.0.0.1 answer a different question
        out.append(addr)
    return out


def _my_ip_check(text, state, metrics=None):
    """Score one answer against the addresses this machine really holds.

    A model can answer from memory, and a made-up address looks like a real
    one, so the answer has to match the machine and not just look right.
    """
    found = _ips_in(text)
    public = state.get("public") or set()
    local = state.get("local") or set()
    for addr in found:
        if str(addr) in public:
            return True, "answer names this machine's public address (%s)" % addr
        if str(addr) in local:
            return True, "answer names an address on this machine (%s)" % addr
    if not found:
        return False, "no IP address in the answer"
    said = ", ".join(str(a) for a in found)
    if not public:
        # The lookup failed, so there is nothing to compare against. Say so
        # rather than failing an answer that may well be right.
        if any(not (a.is_private or a.is_link_local) for a in found):
            return True, ("answer says %s; the public-address lookup failed, "
                          "so this is unverified" % said)
        return False, ("answer says %s, which this machine does not hold; its "
                       "own addresses are %s"
                       % (said, ", ".join(sorted(local)) or "unknown"))
    return False, ("answer says %s; this machine's addresses are %s"
                   % (said, ", ".join(sorted(public | local)) or "unknown"))


# tech-keywords: check that certain terms appear in the answer.
# The state holds the keywords to find and a minimum character count.
_TECH_KEYWORDS = {
    "must_have": ["docker", "ansible"],
    "min_chars": 50,
}


def _tech_keywords_state():
    return dict(_TECH_KEYWORDS)


def _tech_keywords_check(text, state, metrics=None):
    low = text.lower()
    missing = []
    for kw in state["must_have"]:
        if not re.search(r"\b" + re.escape(kw) + r"\b", low):
            missing.append(kw)
    if missing:
        return False, "missing keywords: %s" % ", ".join(missing)
    if len(text) < state["min_chars"]:
        return False, "answer too short (%d chars, need %d)" % (
            len(text), state["min_chars"])
    return True, "found all keywords (%s)" % ", ".join(state["must_have"])
CHECKERS = {
    "time": (_time_state, _time_check),
    "uname": (_uname_state, _uname_check),
    "reasoning": (_reasoning_state, _reasoning_check),
    "ip": (_my_ip_state, _my_ip_check),
    "tech-keywords": (_tech_keywords_state, _tech_keywords_check),
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
        # Reasoning arrives twice: streamed as reasoning_delta events, then
        # repeated on the assembled message. Counting both would double it, so
        # chars is whichever of the two is larger.
        "reasoning": {"deltas": 0, "chars": 0},
        "first_token_s": None,
        "gen_seconds": 0.0,
    }
    streamed_reasoning = 0  # chars seen as reasoning_delta events
    final_reasoning = 0     # chars on the assembled message event
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
            if typ == "reasoning_delta":
                m["reasoning"]["deltas"] += 1
                streamed_reasoning += len(obj.get("reasoning") or "")
            if m["first_token_s"] is None:
                m["first_token_s"] = ts - started
            if req_start is None:
                req_start = ts
            req_end = ts
        elif typ == "message":
            m["final_text"] = obj.get("content") or ""
            final_reasoning = len(obj.get("reasoning") or "")
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
    m["reasoning"]["chars"] = max(streamed_reasoning, final_reasoning)
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


def _start_provider(porter, provider, args):
    """Start (or reuse) one provider's server.

    Returns (name, server_url, proc); proc is the server process to stop
    later, or None when using a running server. Raises when the server
    cannot start.
    """
    name = provider["name"]
    work_dir = os.path.join(HERE, ".eval-work", name)
    os.makedirs(work_dir, exist_ok=True)
    if provider.get("external") or args.no_server:
        server_url = env_of(provider).get("PORTER_SERVER_URL")
        if not server_url:
            raise RuntimeError(
                "provider %s: external mode needs env.PORTER_SERVER_URL" % name)
        print("== %s  (using running server %s)" % (name, server_url))
        return name, server_url, None
    env = env_of(provider)
    auth = ("key from %s" % ("env block" if env.get("PORTER_API_KEY")
                             else "PORTER_API_KEY"))
    if not needs_key(base_url_of(env)):
        auth = "no key needed (local address)"
    print("== %s  (starting server on %s; %s)"
          % (name, provider["addr"], auth))
    proc = start_server(porter, provider, work_dir)
    return name, "http://%s" % provider["addr"], proc


def _run_task(porter, provider_name, server_url, work_dir, case_cfg, case,
              trial, timeout, check_states, out_f, lock):
    """Run one (case, trial) against one provider and write its row.

    A wrong answer, timeout, or tool error is a normal row, not an
    exception. Raises only on unexpected errors (server died, crash), so
    the caller can blame the provider without aborting other tasks.
    """
    prompt = case_cfg["prompt"]
    check = case_cfg.get("check")
    checker = CHECKERS.get(check)
    metrics = run_one(porter, server_url, work_dir, prompt, timeout)
    row = {
        "provider": provider_name,
        "case": case,
        "trial": trial,
        "ts": datetime.datetime.now().isoformat(),
    }
    row.update(metrics)

    if "error" in metrics:
        row["verdict"] = "fail"
    elif checker:
        ok, detail = checker[1](metrics.get("final_text", ""),
                                check_states[check], metrics)
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
    with lock:
        out_f.write(json.dumps(row) + "\n")
        out_f.flush()
    print("  %-22s t%-2d %-4s %7.1fs e2e  %6s tok/s  %d tool(s)"
          % (case, trial, row["verdict"].upper(),
             metrics.get("end_to_end_s") or 0,
             _fmt(metrics.get("gen_tokens_per_sec")),
             len(tools)))


def _run_parallel(porter, providers, case_names, args):
    """Run every (provider, case, trial) as its own task in a shared pool.

    One thread per provider was only as fast as its slowest provider,
    because a provider's runs ran strictly one after another: one looping
    run pinned the whole thread for minutes. Now each run is an independent
    task, so the pool drains slow runs across all threads and wall time
    approaches total work / workers instead of the slowest chain.
    """
    considered = [p for p in providers
                  if not args.only or args.only in p["name"]]

    # Start every provider's server up front, so tasks can run in any order.
    servers = []
    for provider in considered:
        try:
            servers.append(_start_provider(porter, provider, args))
        except Exception as e:
            print("  provider %s failed: %s" % (provider["name"], e),
                  file=sys.stderr)

    tmp_dir = tempfile.mkdtemp(prefix="porter-eval-")
    n_ok = 0
    n_fail = len(considered) - len(servers)  # providers that never started
    row_count = 0
    write_lock = threading.Lock()
    try:
        by_name = {p["name"]: p for p in providers}
        cases = {c: load_config(os.path.join(args.cases_dir, c, "case.yaml"))
                 for c in case_names}
        tasks = []
        out_files = {}
        for name, server_url, proc in servers:
            provider = by_name[name]
            trials = args.trials or provider.get("trials", 3)
            work_dir = os.path.join(HERE, ".eval-work", name)
            out_files[name] = open(
                os.path.join(tmp_dir, name + ".jsonl"), "w")
            # Ground truth per checker, built once per provider as before.
            check_states = {}
            for case in case_names:
                if args.only and args.only not in case:
                    continue
                check = cases[case].get("check")
                if check and check in CHECKERS and check not in check_states:
                    check_states[check] = CHECKERS[check][0]()
            for case in case_names:
                if args.only and args.only not in case:
                    continue
                for trial in range(1, trials + 1):
                    tasks.append((name, server_url, work_dir, cases[case],
                                  case, trial, check_states))

        workers = max(1, args.workers or len(servers))
        with concurrent.futures.ThreadPoolExecutor(
                max_workers=workers) as pool:
            futures = {}
            for name, server_url, work_dir, case_cfg, case, trial, states \
                    in tasks:
                future = pool.submit(
                    _run_task, porter, name, server_url, work_dir, case_cfg,
                    case, trial, args.timeout, states, out_files[name],
                    write_lock)
                futures[future] = name

            outcomes = {}  # provider name -> list of task results
            for future in concurrent.futures.as_completed(futures):
                name = futures[future]
                try:
                    future.result()
                    outcomes.setdefault(name, []).append(True)
                except Exception:
                    outcomes.setdefault(name, []).append(False)
                    print("  provider %s: %s" % (name, traceback.format_exc()),
                          file=sys.stderr)
            for name, results in outcomes.items():
                if all(results):
                    n_ok += 1
                else:
                    n_fail += 1

        for f in out_files.values():
            f.close()
        # Merge per-provider files into the main output file.
        with open(args.out, "w") as combined:
            for provider in considered:
                out_path = os.path.join(tmp_dir, provider["name"] + ".jsonl")
                if os.path.exists(out_path):
                    with open(out_path) as f:
                        for line in f:
                            combined.write(line)
                            row_count += 1
                    os.remove(out_path)
    finally:
        for name, server_url, proc in servers:
            stop_server(proc)
        shutil.rmtree(tmp_dir, ignore_errors=True)

    print("%d provider(s) ok, %d failed; %d row(s) in %s"
          % (n_ok, n_fail, row_count, args.out))
    if n_fail:
        sys.exit(1)

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
    ap.add_argument("--workers", type=int, default=2,
                    help="runs at once (default: 2)")
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

    _run_parallel(porter, providers, case_names, args)

    print("done; rows in %s" % args.out)


def _fmt(v):
    return "-" if v is None else "%.0f" % v


if __name__ == "__main__":
    main()
