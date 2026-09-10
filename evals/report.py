#!/usr/bin/env python3
"""Turn results.jsonl into a summary table.

One row per provider and case: how many trials passed, plus the average and
spread of the main numbers. Run `python3 report.py` after `python3 run_eval.py`.
"""

import argparse
import json
import statistics
import sys


def mean(items):
    items = [i for i in items if i is not None]
    return statistics.mean(items) if items else float("nan")


def sd(items):
    items = [i for i in items if i is not None]
    return statistics.stdev(items) if len(items) > 1 else 0.0


def fmt(v, digits=1):
    if v is None or (isinstance(v, float) and v != v):  # NaN
        return "-"
    return "%%.%df" % digits % v


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="infile", default="results.jsonl")
    ap.add_argument("--group-by", default="case",
                    help="table rows: 'case' (one row per case, cols per provider) or 'provider'")
    args = ap.parse_args()

    rows = []
    with open(args.infile) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    if not rows:
        sys.exit("no rows in %s" % args.infile)

    if args.group_by == "provider":
        _report_by_provider(rows)
    else:
        _report_by_case(rows)


def _pass_rate(rs):
    ok = sum(1 for r in rs if r.get("verdict") == "pass")
    return ok, len(rs)


def _print_group(title, groups):
    # groups: list of (key, rows)
    print(title)
    print("  %-28s %-7s %-11s %-11s %-9s %-9s" %
          ("case/provider", "pass", "e2e_s", "gen_tok/s", "tools", "out_tok"))
    print("  " + "-" * 78)
    for key, rs in groups:
        ok, n = _pass_rate(rs)
        e2e = mean([r.get("end_to_end_s") for r in rs])
        gts = mean([r.get("gen_tokens_per_sec") for r in rs])
        calls = mean([r.get("tool_summary", {}).get("calls") for r in rs])
        out = mean([r.get("tokens", {}).get("output") for r in rs])
        print("  %-28s %-7s %-11s %-11s %-9s %-9s" %
              ("%s/%d" % (key, n), "%d/%d" % (ok, n),
               fmt(e2e), fmt(gts), fmt(calls, 1), fmt(out, 0)))
    print()


def _report_by_case(rows):
    cases = sorted(set(r["case"] for r in rows))
    for case in cases:
        crs = [r for r in rows if r["case"] == case]
        groups = []
        for prov in sorted(set(r["provider"] for r in crs)):
            groups.append((prov, [r for r in crs if r["provider"] == prov]))
        _print_group("CASE: %s" % case, groups)


def _report_by_provider(rows):
    provs = sorted(set(r["provider"] for r in rows))
    for prov in provs:
        prs = [r for r in rows if r["provider"] == prov]
        groups = []
        for case in sorted(set(r["case"] for r in prs)):
            groups.append((case, [r for r in prs if r["case"] == case]))
        _print_group("PROVIDER: %s" % prov, groups)


if __name__ == "__main__":
    main()
