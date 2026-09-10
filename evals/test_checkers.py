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


def check(name, checker, cases, expected_state):
    failures = 0
    for answer, want, *rest in cases:
        state = expected_state()
        ok, detail = checker(answer, state)
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
    time_state, uname_state = R.CHECKERS["time"][0], R.CHECKERS["uname"][0]
    # Pin the clock so the time cases are deterministic.
    bad += check("time", R.CHECKERS["time"][1], TIME_CASES,
                 lambda: {"now": NOW})
    bad += check("uname", R.CHECKERS["uname"][1], UNAME_CASES,
                 lambda: {"allowed": ["darwin", "macos", "mac", "apple",
                                      "arm64", "arm", "aarch64",
                                      "apple silicon"],
                          "host": "Darwin 25.6.0 arm64"})

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
