#!/usr/bin/env python3
"""Tests for the objective checkers in run_eval.py.

Run with:  python3 test_checkers.py

These are the cases that decide pass or fail, so a checker that misreads a
correct answer would make every result wrong in the same direction. Each case
below is a real answer shape a model gives, plus the verdict it must get.
"""

import datetime
import sys

import run_eval as R

# A fixed "now", so the expected results stay put: 2026-09-10 11:00:57 local,
# which is 18:00:57 UTC.
NOW = datetime.datetime(2026, 9, 10, 11, 0, 57)

TIME_CASES = [
    # (answer, expected verdict, why)
    ("The current date and time is: **Thursday, September 10, 2026 at 11:00:57 AM PDT**",
     True, "long form with markdown bold, 12-hour clock and a zone"),
    ("It's **18:01:23 UTC** on **Thursday, September 10, 2026**.", True,
     "UTC clock with a separate date"),
    ("2026-09-10 11:00:57", True, "plain ISO date and time, no zone"),
    ("Today is 10 September 2026. The time is 11:02.", True,
     "day-first date, time in another sentence"),
    ("The date is September 10, 2026 and it's 11:04 AM.", True, "month-first date"),
    ("It is 11:00:57.", True, "time only, no date, means today"),
    ("It is 14:00 on 2026-09-10.", False, "wrong hour, under every zone we read"),
    ("It's 3:30 PM on September 10, 2026.", False, "wrong hour"),
    ("It was 2001-01-01 00:00:00.", False, "right format, far-off date"),
    ("No idea.", False, "no time at all"),
    ("The answer is 42.", False, "no time at all"),
]

UNAME_CASES = [
    ("I'm on macOS on Apple Silicon (arm64).", True),
    ("This runs on Darwin 25.6.0 arm64.", True),
    ("This machine is a computer.", False),
    ("Windows on x86_64.", False),
    ("I cannot tell you.", False),
]

# The machine in these cases holds 203.0.113.9 in public and a home-network
# address plus a link-local IPv6 address of its own. The public block is the
# one reserved for documentation, so a test can never name a real machine.
IP_STATE = {"public": {"203.0.113.9"},
            "local": {"192.168.1.24", "fe80::c8:1aff:fe6a:2d11"}}

IP_CASES = [
    ("Your public IP address is 203.0.113.9.", True, "the machine's public address"),
    ("The machine's IP address is 192.168.1.24.", True, "its own local address"),
    ("It looks like 203.0.113.9 (the router is 192.168.1.1).",
     True, "right address named alongside a wrong one"),
    ("I can't see your network, so I can't tell you.", False, "no address at all"),
    ("Your IP address is 192.168.1.1.", False, "a made-up gateway address"),
    ("Your public address is 198.51.100.7.", False, "plausible but not this machine"),
    ("127.0.0.1", False, "loopback is not an answer to this question"),
    ("0.0.0.0", False, "the unspecified address is not an answer either"),
    ("The time is 2026-09-10 11:00:57, on version 25.6.0.",
     False, "dotted numbers that are not addresses"),
]

# When the public lookup fails there is nothing to compare against, so a
# global address is taken at face value and a private one is still checked
# against the machine. The global address here is Google's public DNS, which
# no machine in these tests can hold.
IP_NO_LOOKUP_CASES = [
    ("Your IP address is 8.8.8.8.", True, "unverified, so accepted"),
    ("Your IP address is 192.168.1.1.", False, "still not an address it holds"),
]
TECH_KEYWORDS_CASES = [
    ("This project uses Docker containers and Ansible for provisioning.",
     True, "has both keywords"),
    ("The VPS runs docker for all its services.",
     False, "missing ansible"),
    ("Ansible handles the initial machine setup.",
     False, "missing docker"),
    ("I don't know.", False, "neither keyword"),
    ("Docker. Ansible. " * 20, True, "both keywords, long enough"),
    ("docker and ansible", False, "both but too short (under 50 chars)"),
]


# Reasoning counts are read from the JSONL, so pin the parse itself: a stream
# where reasoning arrives as deltas and again on the assembled message (it must
# not be counted twice), a stream where only the message carries it, and a
# stream with none.
def _streamed_reasoning():
    return [
        (0.0, {"type": "reasoning_delta", "reasoning": "think "}),
        (0.5, {"type": "reasoning_delta", "reasoning": "again"}),
        (0.7, {"type": "message_delta", "delta": "5 cents"}),
        (1.0, {"type": "message", "content": "5 cents", "reasoning": "think again"}),
        (1.2, {"type": "usage", "input_tokens": 10, "output_tokens": 2}),
    ]


def _message_only_reasoning():
    return [
        (0.0, {"type": "message", "content": "5 cents", "reasoning": "thought it"}),
        (0.2, {"type": "usage", "input_tokens": 8, "output_tokens": 2}),
    ]


def _no_reasoning():
    return [
        (0.0, {"type": "message", "content": "5 cents"}),
        (0.2, {"type": "usage", "input_tokens": 8, "output_tokens": 2}),
    ]


STREAM_CASES = [
    (_streamed_reasoning, {"deltas": 2, "chars": 11}, "deltas and message agree"),
    (_message_only_reasoning, {"deltas": 0, "chars": 10}, "no deltas, message only"),
    (_no_reasoning, {"deltas": 0, "chars": 0}, "nothing to count"),
]


def check_stream(name, cases):
    failures = 0
    for build, want, why in cases:
        got = R.parse_lines(build(), 0.0)["reasoning"]
        if got != want:
            failures += 1
            print("  FAIL  want=%s got=%s (%s)" % (want, got, why))
    print("%-8s %d/%d cases as expected"
          % (name, len(cases) - failures, len(cases)))
    return failures


def check(name, checker, cases, expected_state, metrics=None):
    failures = 0
    for answer, want, *rest in cases:
        state = expected_state()
        ok, detail = checker(answer, state, metrics or {})
        if ok != want:
            failures += 1
            why = rest[0] if rest else ""
            print("  FAIL  want=%-5s got=%-5s  %s" % (want, ok, answer[:64]))
            print("        (%s) -> %s" % (why, detail))
    total = len(cases)
    print("%-8s %d/%d cases as expected" % (name, total - failures, total))
    return failures


def main():
    bad = 0
    # Pin the clock so the time cases are deterministic.
    bad += check("time", R.CHECKERS["time"][1], TIME_CASES,
                 lambda: {"now": NOW})
    bad += check("uname", R.CHECKERS["uname"][1], UNAME_CASES,
                 lambda: {"allowed": ["darwin", "macos", "mac", "apple",
                                      "arm64", "arm", "aarch64",
                                      "apple silicon"],
                          "host": "Darwin 25.6.0 arm64"})
    bad += check("ip", R.CHECKERS["ip"][1], IP_CASES, lambda: IP_STATE)
    bad += check("ip-lookupdown", R.CHECKERS["ip"][1], IP_NO_LOOKUP_CASES,
                 lambda: {"public": set(), "local": IP_STATE["local"]})

    # The reasoning checker scores the stream, not the answer text, so each
    # case carries the counts the runner parsed out of the JSONL.
    reasoning_cases = [
        ("The ball costs 5 cents.", {"reasoning": {"deltas": 14, "chars": 612}},
         True, "reasoning streamed, then the answer"),
        ("The ball costs 5 cents.", {"reasoning": {"deltas": 0, "chars": 0}},
         False, "endpoint dropped the reasoning"),
        ("The ball costs 5 cents.", {},
         False, "row carries no reasoning counts"),
        ("", {"reasoning": {"deltas": 3, "chars": 40}},
         False, "reasoning arrived without an answer"),
    ]
    failures = 0
    for answer, metrics, want, why in reasoning_cases:
        ok, detail = R.CHECKERS["reasoning"][1](answer, {}, metrics)
        if ok != want:
            failures += 1
            print("  FAIL  want=%-5s got=%-5s  %s" % (want, ok, why))
            print("        -> %s" % detail)
    bad += failures
    print("%-8s %d/%d cases as expected"
          % ("reasoning", len(reasoning_cases) - failures, len(reasoning_cases)))
    bad += check("tech-keywords", R.CHECKERS["tech-keywords"][1],
                   TECH_KEYWORDS_CASES,
                   lambda: {"must_have": ["docker", "ansible"], "min_chars": 50})

    bad += check_stream("stream", STREAM_CASES)

    # Tool success is read from the shell tool's exit-code line.
    tool_cases = [
        ("shell", "out\n\nexit code: 0\n", False, True, "clean exit"),
        ("shell", "sh: nope: not found\n\nexit code: 127\n", False, False, "failure exit"),
        ("shell", "partial\n\nexit code: 0\n", True, False, "timed out"),
        ("shell", "", False, False, "no output"),
        ("read_with_line_numbers", "file has 3 lines", False, True, "other tool"),
        ("read_with_line_numbers", "", False, False, "other tool, empty"),
    ]
    failures = 0
    for name, result, timed_out, want, why in tool_cases:
        got = R._tool_ok(name, result, timed_out)
        if got != want:
            failures += 1
            print("  FAIL  %s timed_out=%s want=%s got=%s (%s)"
                  % (name, timed_out, want, got, why))
    bad += failures
    print("%-8s %d/%d cases as expected"
          % ("tools", len(tool_cases) - failures, len(tool_cases)))

    print()
    if bad:
        print("%d checker case(s) wrong" % bad)
        return 1
    print("all checker cases pass")
    return 0


if __name__ == "__main__":
    sys.exit(main())
