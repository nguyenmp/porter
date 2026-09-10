#!/usr/bin/env python3
"""Turn results.jsonl into one HTML page that compares endpoints.

The page holds a recommendation, a sortable table, bar charts for speed and
wait, a dot for every trial, and the raw numbers behind them. Open the file in
a browser: it needs no server, no network, and no build step.

    python3 report_html.py --in results.jsonl --out results_report.html

Three choices here differ from report.py:

* Trials the gateway refused count in the pass column but stay out of the speed
  numbers. A call that fails in 0.1 s does not make an endpoint fast.
* Speed columns show the median, which is the middle trial. Five trials is few
  enough that one slow call drags an average a long way off.
* The ranking covers only the measures the endpoint itself controls: how fast
  it writes, how long it waits before the first token, and whether it reuses
  the prompt from cache. Total time also carries the tool calls, the local
  startup, and the wait before any later turn the model chooses to take, so it
  cannot say which endpoint is quicker.
"""

import argparse
import datetime
import html
import json
import random
import statistics


def median(values):
    values = [v for v in values if v is not None]
    return statistics.median(values) if values else None


def number_words(count):
    """Spell out a small count, so a sentence never opens with a digit."""
    return {0: "No", 1: "One", 2: "Two", 3: "Three", 4: "Four",
            5: "Five", 6: "Six", 7: "Seven", 8: "Eight", 9: "Nine"}.get(count, str(count))


def format_number(value):
    """Write a token count the way a person reads it, with thousands groups."""
    return "" if value is None else "{:,.0f}".format(value)


def spread(values):
    values = [v for v in values if v is not None]
    if not values:
        return {"med": None, "min": None, "max": None}
    return {"med": statistics.median(values), "min": min(values), "max": max(values)}


def short_name(name):
    """Drop the shared model prefix so the provider slug stands out."""
    for prefix in ("deepseek-v4-flash-", "deepseek-v4-"):
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def precision_of(name):
    for tag in ("fp8", "bf16", "fp4", "fp16"):
        if name.endswith("-" + tag):
            return tag
    return "unstated"


def shared_model(names):
    """Name the model every endpoint serves, or '' when they differ.

    All provider names in one run start with the model name, so the common
    prefix names the model the page is about.
    """
    if not names:
        return ""
    prefix = names[0]
    for name in names[1:]:
        while prefix and not name.startswith(prefix):
            prefix = prefix[:-1]
        if not prefix:
            return ""
    return prefix.rstrip("-")


def split_cell(rows, provider, case):
    """Turn one provider's trials for one case into the numbers the page shows."""
    group = [r for r in rows if r["provider"] == provider and r["case"] == case]
    good = [r for r in group if r.get("verdict") == "pass"]
    bad = [r for r in group if r.get("verdict") != "pass"]

    cached = sum(r["tokens"].get("cached_input", 0) for r in group)
    sent = sum(r["tokens"].get("input", 0) for r in group)

    errors = []
    for r in bad:
        message = (r.get("error") or "no error text").strip()
        if message not in errors:
            errors.append(message)

    return {
        "n_all": len(group),
        "n_pass": len(good),
        "n_error": len(bad),
        "errors": errors,
        "first_token": spread([r.get("first_token_s") for r in good]),
        "end_to_end": spread([r.get("end_to_end_s") for r in good]),
        "gen_tps": spread([r.get("gen_tokens_per_sec") for r in good]),
        "out_tokens": spread([r["tokens"].get("output") for r in good]),
        "in_tokens": median([r["tokens"].get("input") for r in good]),
        "cached_pct": (100.0 * cached / sent) if sent else None,
        "tool_calls": sum(r["tool_summary"].get("calls", 0) for r in good),
        "tool_ok": sum(r["tool_summary"].get("ok", 0) for r in good),
        "trials_with_tools": sum(1 for r in good if r["tool_summary"].get("calls", 0)),
    }


def trial_records(rows):
    out = []
    for r in sorted(rows, key=lambda r: (r["provider"], r["case"], r["trial"])):
        good = r.get("verdict") == "pass"
        out.append({
            "provider": r["provider"],
            "short": short_name(r["provider"]),
            "case": r["case"],
            "trial": r["trial"],
            "verdict": r.get("verdict"),
            "first_token_s": r.get("first_token_s"),
            "end_to_end_s": r.get("end_to_end_s"),
            "gen_tps": r.get("gen_tokens_per_sec"),
            "in_tokens": r["tokens"].get("input"),
            "out_tokens": r["tokens"].get("output"),
            "cached_tokens": r["tokens"].get("cached_input"),
            "calls": r["tool_summary"].get("calls", 0),
            "ok": r["tool_summary"].get("ok", 0),
            "note": "" if good else (r.get("error") or ""),
            "detail": r.get("detail") or "",
            "ts": r.get("ts"),
        })
    return out


def build_findings(rows, providers):
    """Write the short summary that opens the page."""
    findings = []

    answered = [r for r in rows if r.get("verdict") == "pass"]
    refused = [r for r in rows if r.get("verdict") != "pass"]
    findings.append({
        "head": "Every endpoint that answered got both cases right.",
        "body": "%d of %d trials passed. The other %d never reached a model: the gateway "
                "refused them. Speed is the only real difference this run shows."
                % (len(answered), len(rows), len(refused)),
    })

    def middle(provider, field):
        """The median for one endpoint across the trials that answered."""
        return median([r.get(field) for r in rows
                       if r["provider"] == provider and r.get("verdict") == "pass"])

    def middle_input(provider):
        """The median prompt size for one endpoint, which counts tokens, not seconds."""
        return median([r["tokens"].get("input") for r in rows
                       if r["provider"] == provider and r.get("verdict") == "pass"])

    speed = [(middle(p, "gen_tokens_per_sec"), p) for p in providers]
    speed = [s for s in speed if s[0] is not None]
    if len(speed) > 1:
        speed.sort(reverse=True)
        fastest, slowest = speed[0], speed[-1]
        findings.append({
            "head": "The fastest endpoint writes about %d times faster than the slowest."
                    % round(fastest[0] / slowest[0]),
            "body": "%s writes %.0f output tokens a second and finishes a task in %.1f s. "
                    "%s writes %.0f a second and takes %.1f s. Both counts are medians, "
                    "which means the middle trial of the five. One slow call moves a "
                    "median less than it moves an average."
                    % (short_name(fastest[1]), fastest[0], middle(fastest[1], "end_to_end_s"),
                       short_name(slowest[1]), slowest[0], middle(slowest[1], "end_to_end_s")),
        })

    waits = [r for r in rows if r.get("verdict") == "pass" and (r.get("first_token_s") or 0) > 10]
    if waits:
        worst = max(waits, key=lambda r: r["first_token_s"])
        peers = [r["end_to_end_s"] for r in rows
                 if r["provider"] == worst["provider"]
                 and r.get("verdict") == "pass"
                 and r is not worst]
        body = ("The worst is %s on the %s case: %.1f s to the first token, %.1f s in total."
                % (short_name(worst["provider"]), worst["case"],
                   worst["first_token_s"], worst["end_to_end_s"]))
        if peers:
            body += (" Its other trials all finished inside %.1f s, so count that one as a "
                     "bad call rather than a slow endpoint." % max(peers))
        findings.append({
            "head": "%s %s took more than 10 s to start answering."
                    % (number_words(len(waits)),
                       "trial" if len(waits) == 1 else "trials"),
            "body": body,
        })

    no_cache = [p for p in providers
                if any(r["provider"] == p and r.get("verdict") == "pass" for r in rows)
                and all(r["tokens"].get("cached_input", 0) == 0
                        for r in rows if r["provider"] == p and r.get("verdict") == "pass")]
    if no_cache:
        names = ", ".join(short_name(p) for p in no_cache)
        size = middle_input(no_cache[0])
        body = ("Every other endpoint read most of the long prompt from its cache. %s read "
                "none of it, so every turn sends the whole prompt again and pays for it in "
                "full." % names)
        if size:
            body += " That is %s input tokens a turn." % format_number(size)
        findings.append({
            "head": "%s reported no cached input." % names,
            "body": body,
        })

    return findings


def trials_of(rows, provider):
    """The trials that answered, which are the only ones worth ranking."""
    return [r for r in rows if r["provider"] == provider and r.get("verdict") == "pass"]


def measure(trials, key):
    """The median for one measure across one endpoint's trials."""
    if key == "cached_pct":
        sent = sum(r["tokens"].get("input", 0) for r in trials)
        hit = sum(r["tokens"].get("cached_input", 0) for r in trials)
        return (100.0 * hit / sent) if sent else 0.0
    return median([r.get(key) for r in trials])


# Only these three measures reflect the endpoint itself. Each one is read from
# the stream the model sends, so it holds whatever else the run was doing.
MEASURES = [
    {"key": "gen_tokens_per_sec", "label": "Tokens/s",
     "hint": "higher is better", "higher": True, "digits": 1},
    {"key": "first_token_s", "label": "Wait (s)",
     "hint": "lower is better", "higher": False, "digits": 2},
    {"key": "cached_pct", "label": "From cache",
     "hint": "higher is better", "higher": True, "digits": 0},
]


def build_picks(rows, providers, rounds=2000):
    """Rank the endpoints on the measures the endpoint itself controls.

    An endpoint lands in the better half when its median beats the middle
    endpoint's median. With nine endpoints there is no exact half, so that line
    falls between fourth and fifth place and sits close to the middle of the
    crowd. Each mark therefore carries how often it survives a resample.
    """
    live = [p for p in providers if trials_of(rows, p)]
    cut = len(live) // 2

    met = {m["key"]: {p: measure(trials_of(rows, p), m["key"]) for p in live}
           for m in MEASURES}
    order = {m["key"]: sorted(live, key=lambda p: met[m["key"]][p],
                              reverse=m["higher"]) for m in MEASURES}

    # Resample the trials, recompute the medians, re-rank. This separates the
    # part of a mark that is signal from the part that is luck in five calls.
    verdicts = {(p, m["key"]): 0 for p in live for m in MEASURES}
    if cut:
        for _ in range(rounds):
            for m in MEASURES:
                sample = {}
                for p in live:
                    group = trials_of(rows, p)
                    drawn = [random.choice(group) for _ in group]
                    sample[p] = measure(drawn, m["key"])
                for p in sorted(live, key=lambda p: sample[p],
                                reverse=m["higher"])[:cut]:
                    verdicts[(p, m["key"])] += 1

    out_rows = []
    for p in live:
        cells = []
        for m in MEASURES:
            share = verdicts[(p, m["key"])] / rounds if rounds else 0.0
            rank = order[m["key"]].index(p) + 1
            best = rank <= cut
            cells.append({
                "label": m["label"],
                "text": ("{:,.%df}" % m["digits"]).format(met[m["key"]][p]),
                "rank": rank,
                "best": best,
                # A mark worth trusting stays in the better half in nine rounds
                # out of ten.
                "firm": best and share >= 0.9,
                "share": share,
            })
        out_rows.append({
            "name": p, "short": short_name(p), "cells": cells,
            "marks": sum(1 for c in cells if c["best"]),
            "firm": sum(1 for c in cells if c["firm"]),
            "of": len(MEASURES),
        })
    out_rows.sort(key=lambda r: (-r["marks"], -r["firm"]))

    headlines, notes = [], []
    if out_rows and out_rows[0]["marks"]:
        most = out_rows[0]["marks"]
        leaders = [r for r in out_rows if r["marks"] == most]
        names = [r["short"] for r in leaders]
        if len(names) <= 3:
            headlines.append({
                "label": "Use these" if len(names) > 1 else "Use this",
                "text": " and ".join(names),
                "why": "in the better half of %d of the %d measures" % (most, len(MEASURES)),
            })
        else:
            headlines.append({
                "label": "No clear winner",
                "text": "%d endpoints tie at %d of %d measures" % (len(names), most, len(MEASURES)),
                "why": "read the table below for which measures each one clears",
            })
        if 1 < len(leaders) <= 3:
            missed = [(r["short"], [MEASURES[i]["label"] for i, c in enumerate(r["cells"])
                                    if not c["best"]]) for r in leaders]
            if all(m for _, m in missed):
                notes.append("They part company here: "
                             + "; ".join("%s misses %s" % (n, ", ".join(m)) for n, m in missed) + ".")
        elif len(names) == 1:
            leader = leaders[0]
            gaps = [MEASURES[i]["label"] for i, c in enumerate(leader["cells"]) if not c["best"]]
            if gaps:
                notes.append("%s still misses %s." % (leader["short"], ", ".join(gaps)))
    # The numbers behind these two notes, measured from this run.
    one_turn = [r["end_to_end_s"] - (r.get("first_token_s") or 0) - (r.get("gen_seconds") or 0)
                for r in rows if r.get("verdict") == "pass" and r["tokens"].get("input", 0) < 7000]
    more_turns = [r["end_to_end_s"] - (r.get("first_token_s") or 0) - (r.get("gen_seconds") or 0)
                  for r in rows if r.get("verdict") == "pass" and r["tokens"].get("input", 0) >= 7000]
    notes.append("Only these three measures belong to the endpoint. It sets how fast it "
                 "writes, how long it takes to start, and whether it can reuse the prompt "
                 "from its cache.")
    if one_turn:
        tail = ((" When the model answers in one turn, that leftover time comes to %.2f s; "
                 "when it takes a second turn it reaches %.2f s, because the wait before "
                 "that turn starts writing is not covered by the first-token clock.")
                % (median(one_turn), median(more_turns)) if more_turns else "")
        notes.append("Total time is left out. It also carries the tool calls, the local "
                     "startup, the wait before any later turn the model decides to take, "
                     "and it rewards an endpoint that writes a shorter answer." + tail)

    skipped = [r["short"] for r in out_rows if r["marks"] == 0]
    if skipped:
        headlines.append({
            "label": "Skip",
            "text": ", ".join(skipped),
            "why": "in the better half of no measure at all",
        })
    return {"rows": out_rows, "metrics": MEASURES, "cut": cut, "rounds": rounds,
            "headlines": headlines, "notes": notes, "count": len(live)}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="infile", default="results.jsonl",
                    help="the JSONL file run_eval.py wrote")
    ap.add_argument("--out", dest="outfile", default="results_report.html",
                    help="where to write the HTML page")
    args = ap.parse_args()

    rows = []
    with open(args.infile) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise SystemExit("no rows in %s" % args.infile)

    providers = sorted({r["provider"] for r in rows})
    cases = sorted({r["case"] for r in rows})

    def pooled(provider, field):
        """The median for one endpoint across both cases, used to rank the charts."""
        good = [r for r in rows if r["provider"] == provider and r.get("verdict") == "pass"]
        return median([r.get(field) for r in good])

    cells = {}
    for p in providers:
        for c in cases:
            cells["%s|%s" % (p, c)] = split_cell(rows, p, c)

    stamps = sorted(r["ts"] for r in rows if r.get("ts"))
    window = "%s to %s" % (stamps[0][:16].replace("T", " "), stamps[-1][:16].replace("T", " ")) if stamps else "unknown"

    data = {
        # Both stamps read from the same clock, so a reader can compare them.
        "generated": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "window": window,
        "rows_total": len(rows),
        "model": shared_model(providers),
        "providers": [{"name": p, "short": short_name(p), "precision": precision_of(p),
                       "pooled": {"gen_tps": pooled(p, "gen_tokens_per_sec"),
                                  "first_token": pooled(p, "first_token_s"),
                                  "end_to_end": pooled(p, "end_to_end_s")}}
                      for p in providers],
        "cases": cases,
        "cells": cells,
        "trials": trial_records(rows),
        "findings": build_findings(rows, providers),
        "picks": build_picks(rows, providers),
    }

    with open(args.outfile, "w") as f:
        f.write(PAGE.replace("__DATA__", json.dumps(data, allow_nan=False))
                    .replace("__TITLE__", html.escape(args.infile)))
    print("wrote %s (%d endpoints, %d cases, %d trials)"
          % (args.outfile, len(providers), len(cases), len(rows)))


PAGE = r"""<!doctype html>
<html lang="en">
<meta charset="utf-8">
<title>Endpoint comparison</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  :root {
    --ink: #14202e;
    --muted: #5d6b7a;
    --line: #dde4ec;
    --bg: #f6f8fb;
    --card: #ffffff;
    --machine: #2f6fd0;
    --whattime: #0f9b8e;
    --bad: #c2352b;
    --warn: #a86213;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; padding: 32px 28px 72px; background: var(--bg); color: var(--ink);
    font: 15px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  }
  .wrap { max-width: 1180px; margin: 0 auto; }
  h1 { font-size: 26px; margin: 0 0 6px; letter-spacing: -0.01em; }
  h2 { font-size: 19px; margin: 40px 0 4px; letter-spacing: -0.01em; }
  h2 + .sub { margin: 0 0 16px; color: var(--muted); font-size: 13.5px; }
  .meta { color: var(--muted); font-size: 13.5px; margin: 0 0 24px; }
  .card { background: var(--card); border: 1px solid var(--line); border-radius: 10px; padding: 20px 22px; }
  .findings { display: grid; gap: 16px; }

  .headline { display: grid; gap: 9px; margin: 0 0 20px; }
  .headline div { display: flex; gap: 12px; align-items: baseline; flex-wrap: wrap; }
  .headline em {
    font-style: normal; font-size: 11.5px; text-transform: uppercase; letter-spacing: 0.05em;
    color: var(--muted); min-width: 116px;
  }
  .headline strong { font-size: 15px; }
  .headline span { color: var(--muted); font-size: 13.5px; }
  td.mark, th.mark { background: #eaf1fb; }
  td.mark.firm, th.mark.firm { background: #d3e3fa; }
  td.mark .val { font-weight: 600; }
  .val { display: block; }
  .rank { display: block; font-size: 11px; color: var(--muted); text-transform: none; letter-spacing: 0; }
  #picknotes { margin-top: 16px; display: grid; gap: 7px; }
  #picknotes div { font-size: 13.5px; color: var(--muted); border-top: 1px solid var(--line); padding-top: 9px; }
  .finding { border-left: 3px solid var(--machine); padding-left: 14px; }
  .finding b { display: block; font-size: 15px; }
  .finding span { color: var(--muted); font-size: 14px; }
  table { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }
  th, td { text-align: right; padding: 7px 9px; border-bottom: 1px solid var(--line); white-space: nowrap; }
  th:first-child, td:first-child { text-align: left; }
  thead th { font-size: 12px; text-transform: uppercase; letter-spacing: 0.04em; color: var(--muted); cursor: pointer; user-select: none; }
  thead th:hover { color: var(--ink); }
  thead th.sorted::after { content: " \25BE"; }
  thead th.sorted.asc::after { content: " \25B4"; }
  tbody tr:hover { background: #f2f6fc; }
  tr.bad td { color: var(--muted); }
  .name { font-weight: 600; }
  .badge { display: inline-block; font-size: 11px; padding: 1px 6px; border-radius: 999px; background: #eef2f8; color: var(--muted); margin-left: 7px; vertical-align: 1px; }
  .badge.fp8 { background: #e7f0ff; color: #1f5ab5; }
  .pass { color: #17693f; font-weight: 600; }
  .err { color: var(--bad); font-weight: 600; }
  .chart { background: var(--card); border: 1px solid var(--line); border-radius: 10px; padding: 18px 20px 12px; margin-bottom: 18px; }
  .chart h3 { margin: 0 0 10px; font-size: 16px; }
  .legend { display: flex; gap: 18px; margin: 0 0 10px; font-size: 13px; color: var(--muted); }
  .legend i { display: inline-block; width: 11px; height: 11px; border-radius: 3px; margin-right: 6px; vertical-align: -1px; }
  svg { display: block; width: 100%; height: auto; overflow: visible; }
  .trialwrap { max-height: 460px; overflow: auto; }
  .tablewrap { overflow-x: auto; }
  .note { font-size: 13.5px; color: var(--muted); }
  .note li { margin-bottom: 7px; }
  .pill { font-size: 11.5px; padding: 1px 7px; border-radius: 999px; background: #fdeceb; color: var(--bad); }
  footer { margin-top: 40px; color: var(--muted); font-size: 13px; }
</style>
<body>
<div class="wrap">
  <h1 id="title"></h1>
  <p class="meta" id="meta"></p>

  <div class="card findings" id="findings"></div>

  <h2>Which endpoint should you use?</h2>
  <p class="sub">Each endpoint is ranked on the three measures it controls. A tinted cell means it lands in the better half of the nine. A bold number means it stays there in nine resamples out of ten, so treat the plain ones as too close to call.</p>
  <div class="card">
    <div class="headline" id="headline"></div>
    <div id="picks" class="tablewrap"></div>
    <div id="picknotes"></div>
  </div>
  <div class="card" style="margin-top:14px">
    <ul class="note">
      <li>An endpoint lands in the better half when its median beats the middle endpoint's median. With nine endpoints that line falls between fourth and fifth place, and five endpoints change sides when it moves one place. Count how many measures an endpoint clears instead of trusting one mark.</li>
      <li>To test each mark I reran the ranking two thousand times. Each round drew five trials at random from the ones that endpoint ran, repeats allowed, then recomputed the medians and the order. A bold number held its place in at least nine rounds out of ten.</li>
      <li>Five trials per endpoint, all inside one hour, is a small sample. Read these picks as a shortlist to test, not a verdict.</li>
    </ul>
  </div>

  <h2>Every endpoint side by side</h2>
  <p class="sub">One row per endpoint and case, quickest first. Click a heading to sort by it. Speed numbers come from the trials that answered, so a refused call never counts as a fast time. Each speed number is a median: the middle trial, not the average.</p>
  <div class="card"><div id="scoreboard" class="tablewrap"></div></div>

  <h2>How fast each endpoint writes</h2>
  <p class="sub">Output tokens a second for the middle trial. Longer is faster. The thin line marks the slowest and fastest trial. Time spent waiting for a tool call is not counted here.</p>
  <div class="chart" id="chart-tps"></div>

  <h2>How long before the answer starts</h2>
  <p class="sub">Seconds from sending the prompt to the first output token, for the middle trial. Shorter is better. This is the wait you notice while the answer is still blank.</p>
  <div class="chart" id="chart-ft"></div>

  <h2>How long a whole task takes</h2>
  <p class="sub">Seconds from sending the prompt to the last token, for the middle trial. Tool calls are included. Shorter is better.</p>
  <div class="chart" id="chart-e2e"></div>

  <h2>How steady each endpoint is</h2>
  <p class="sub">One dot for each trial that answered, on a log scale so the slow ones stay on the page. Further left is faster. A wide group means you cannot tell what the next call will do. Only the trials that answered appear here.</p>
  <div class="chart" id="chart-dots"></div>

  <h2>Every trial in full</h2>
  <p class="sub">The numbers behind the charts, one row per trial, in the order they ran.</p>
  <div class="card trialwrap" id="trials"></div>

  <h2>What this run cannot tell you</h2>
  <div class="card">
    <ul class="note">
      <li>Each endpoint ran five trials per case. That is enough to spot a slow endpoint, and too few to separate two endpoints that land close together.</li>
      <li>The endpoints ran one after another over about an hour, not at the same time. Load on the provider and on the gateway changed between runs, and some of the spread in these numbers is that.</li>
      <li>Tokens a second counts only the time the model spent writing. Total time adds the wait for the tool calls, so an endpoint can look quick on one chart and slow on the other.</li>
      <li>Each endpoint pinned one provider with no fallback, so every row measures that endpoint alone. No provider stood in for another.</li>
      <li>Correctness comes down to two small cases. Every endpoint that answered passed both, so this run cannot say which one reasons better.</li>
      <li>Tokens a second is a noisy number for short answers. One endpoint wrote 24 to 58 output tokens on the machine case, and its trials ran from 14 to 170 tokens a second. A fraction of a second of scheduling delay moves a tiny answer a long way.</li>
      <li>The gateway refused digitalocean ten times between 10:23 and 10:30. The same endpoint answered at 11:13. Something changed in between, so those refusals say nothing about how fast the endpoint is.</li>
    </ul>
  </div>

  <footer id="footer"></footer>
</div>

<script>
const DATA = __DATA__;

const MACHINE = getComputedStyle(document.documentElement).getPropertyValue('--machine').trim();
const WHATTIME = getComputedStyle(document.documentElement).getPropertyValue('--whattime').trim();
const CASE_COLOR = {};
DATA.cases.forEach((c, i) => CASE_COLOR[c] = i === 0 ? MACHINE : WHATTIME);

const fmt = (v, d = 2) => (v === null || v === undefined || Number.isNaN(v)) ? "-" : v.toFixed(d);
const cap = s => s.replace(/[-_/]/g, " ").replace(/\b\w/g, ch => ch.toUpperCase());

document.getElementById('title').textContent = DATA.model
  ? `${DATA.providers.length} ${DATA.model} endpoints, compared`
  : 'Endpoints compared';

document.getElementById('meta').textContent =
  `${DATA.rows_total} trials · ${DATA.cases.length} cases · run ${DATA.window} · `
  + `page built ${DATA.generated} from __TITLE__`;

document.getElementById('findings').innerHTML = DATA.findings.map(f =>
  `<div class="finding"><b>${f.head}</b><span>${f.body}</span></div>`).join('');

// ---------- the recommendation ----------
(function renderPicks() {
  const P = DATA.picks;
  document.getElementById('headline').innerHTML = P.headlines.map(h =>
    `<div><em>${h.label}</em><strong>${h.text}</strong><span>${h.why}</span></div>`).join('');

  const head = '<tr><th>Endpoint</th>'
    + P.metrics.map(m => `<th>${m.label}<span class="rank">${m.hint}</span></th>`).join('')
    + '<th>Better half</th></tr>';
  const body = P.rows.map(r => `<tr>
      <td>${r.short}</td>`
    + r.cells.map(c => {
        const cls = [(c.best ? 'mark' : ''), (c.firm ? 'firm' : '')].join(' ').trim();
        return `<td class="${cls}"><span class="val">${c.text}</span>`
             + `<span class="rank">#${c.rank}</span></td>`;
      }).join('')
    + `<td class="count">${r.marks} of ${r.of}</td></tr>`).join('');
  document.getElementById('picks').innerHTML =
    `<table><thead>${head}</thead><tbody>${body}</tbody></table>`;
  document.getElementById('picknotes').innerHTML =
    P.notes.map(n => `<div>${n}</div>`).join('');
})();

// ---------- scoreboard ----------
// Each column carries the direction that puts the better number first.
const COLS = [
  {key: 'label',   label: 'Endpoint',        num: false, dir: 1},
  {key: 'pass',    label: 'Passed',          num: true,  dir: -1},
  {key: 'tps',     label: 'Tokens/s',        num: true,  dir: -1},
  {key: 'ft',      label: 'Wait (s)',        num: true,  dir: 1},
  {key: 'e2e',     label: 'Total (s)',       num: true,  dir: 1},
  {key: 'out',     label: 'Out tokens',      num: true,  dir: -1},
  {key: 'inp',     label: 'In tokens',       num: true,  dir: -1},
  {key: 'cached',  label: 'From cache',      num: true,  dir: -1},
  {key: 'calls',   label: 'Tools per trial', num: true,  dir: -1},
  {key: 'toolpct', label: 'Trials using a tool', num: true, dir: -1},
];

function scoreRows() {
  const out = [];
  DATA.providers.forEach(p => DATA.cases.forEach(c => {
    const cell = DATA.cells[p.name + '|' + c];
    if (!cell || cell.n_all === 0) return;
    out.push({
      provider: p, case: c, cell,
      label: p.short, caseLabel: cap(c),
      pass: cell.n_all ? cell.n_pass / cell.n_all : 0,
      passText: `${cell.n_pass}/${cell.n_all}`,
      tps: cell.gen_tps.med, ft: cell.first_token.med, e2e: cell.end_to_end.med,
      out: cell.out_tokens.med, inp: cell.in_tokens,
      cached: cell.cached_pct,
      calls: cell.n_pass ? cell.tool_calls / cell.n_pass : null,
      toolpct: cell.n_pass ? 100 * cell.trials_with_tools / cell.n_pass : null,
    });
  }));
  return out;
}

// Open on the fastest endpoint, so the table leads with the best option.
let sortKey = 'tps', sortDir = -1;

function renderScoreboard() {
  const rows = scoreRows().sort((a, b) => {
    const x = a[sortKey], y = b[sortKey];
    if (typeof x === 'string' || typeof y === 'string') return sortDir * String(x).localeCompare(String(y));
    if (x === null) return 1;
    if (y === null) return -1;
    return sortDir * (x - y);
  });
  const head = COLS.map(c => `<th data-key="${c.key}" class="${c.num ? 'num' : ''} ${c.key === sortKey ? 'sorted' + (sortDir === 1 ? ' asc' : '') : ''}">${c.label}</th>`).join('');
  const body = rows.map(r => {
    const c = r.cell;
    const failed = c.n_pass === 0;
    const tip = c.errors.length ? ` title="${c.errors.join(' / ').replace(/"/g, '')}"` : '';
    return `<tr class="${failed ? 'bad' : ''}"${tip}>
      <td><span class="name">${r.provider.short}</span><span class="badge ${r.provider.precision}">${r.provider.precision}</span>
          <span class="badge">${r.caseLabel}</span></td>
      <td class="${failed ? 'err' : 'pass'}">${r.passText}${c.n_error ? ` <span class="pill">${c.n_error} refused</span>` : ''}</td>
      <td>${fmt(r.tps, 1)}</td>
      <td>${fmt(r.ft)}</td>
      <td>${fmt(r.e2e)}</td>
      <td>${r.out === null ? '-' : Math.round(r.out)}</td>
      <td>${r.inp === null ? '-' : Math.round(r.inp).toLocaleString()}</td>
      <td>${r.cached === null ? '-' : Math.round(r.cached) + '%'}</td>
      <td>${fmt(r.calls, 1)}</td>
      <td>${r.toolpct === null ? '-' : Math.round(r.toolpct) + '%'}</td>
    </tr>`;
  }).join('');
  document.getElementById('scoreboard').innerHTML = `<table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table>`;
  document.querySelectorAll('#scoreboard thead th').forEach(th => th.onclick = () => {
    const key = th.dataset.key;
    if (key === sortKey) sortDir = -sortDir;
    else { sortKey = key; sortDir = COLS.find(c => c.key === key).dir; }
    renderScoreboard();
  });
}
renderScoreboard();

// ---------- charts ----------
function barChart(el, rows, opts) {
  const g = 158, w = 880, barH = 13, rowH = 30, series = DATA.cases;
  // Scale to the medians, with a little room. One slow trial can sit far past
  // that, so whiskers and value labels stop at the plot edge and say so.
  const plotRight = w - 62;
  const max = opts.max || Math.max(...rows.flatMap(r => series.map(s => r.vals[s].med || 0))) * 1.15;
  const height = 26 + rows.length * rowH + 8;
  const x = v => Math.min(g + (v / max) * (plotRight - g), plotRight);
  let svg = `<svg viewBox="0 0 ${w} ${height}" role="img" aria-label="${opts.title}">`;
  for (let t = 0; t <= max; t += opts.tick) {
    svg += `<line x1="${x(t)}" y1="14" x2="${x(t)}" y2="${height - 6}" stroke="#eef2f7"/>
            <text x="${x(t)}" y="12" font-size="10" fill="#8c99a8" text-anchor="middle">${opts.axis(t)}</text>`;
  }
  rows.forEach((r, i) => {
    const top = 22 + i * rowH;
    svg += `<text x="${g - 12}" y="${top + 17}" font-size="12.5" fill="#14202e" text-anchor="end">${r.label}</text>`;
    series.forEach((s, si) => {
      const d = r.vals[s];
      const y = top + si * barH + si * 1.5;
      if (d.med === null) {
        svg += `<rect x="${g}" y="${y}" width="3" height="${barH}" fill="#c9d2dc"/>`;
        return;
      }
      const color = CASE_COLOR[s];
      const barEnd = Math.max(x(d.med), g + 2);
      svg += `<rect x="${g}" y="${y}" width="${barEnd - g}" height="${barH}" fill="${color}" rx="2"/>`;
      if (d.min !== null && d.max > d.min) {
        const lo = x(d.min), hi = x(d.max);
        svg += `<line x1="${lo}" y1="${y + barH / 2}" x2="${hi}" y2="${y + barH / 2}" stroke="#14202e" stroke-width="1" opacity="0.45"/>`;
        if (d.min <= max) {
          svg += `<line x1="${lo}" y1="${y + 2}" x2="${lo}" y2="${y + barH - 2}" stroke="#14202e" stroke-width="1" opacity="0.45"/>`;
        }
        if (d.max > max) {
          // The range runs past the edge: mark that instead of hiding it.
          svg += `<path d="M ${plotRight - 7} ${y} L ${plotRight} ${y + barH / 2} L ${plotRight - 7} ${y + barH} Z" fill="#14202e" opacity="0.5"/>`;
        } else {
          svg += `<line x1="${hi}" y1="${y + 2}" x2="${hi}" y2="${y + barH - 2}" stroke="#14202e" stroke-width="1" opacity="0.45"/>`;
        }
      }
      const label = opts.value(d.med);
      if (barEnd + label.length * 6.6 + 10 < w) {
        svg += `<text x="${barEnd + 5}" y="${y + barH - 1}" font-size="11" fill="#5d6b7a">${label}</text>`;
      } else {
        // No room after the bar: put the number inside it.
        svg += `<text x="${barEnd - 5}" y="${y + barH - 1}" font-size="11" fill="#fff" text-anchor="end">${label}</text>`;
      }
    });
  });
  svg += `</svg>`;
  // The section heading above the card already explains the chart, so the card
  // shows its title, the legend, and the plot.
  el.innerHTML = `<h3>${opts.title}</h3>
    <div class="legend">${series.map(s => `<span><i style="background:${CASE_COLOR[s]}"></i>${cap(s)}</span>`).join('')}
    <span><i style="background:#14202e;opacity:.45;width:14px;height:2px"></i>slowest to fastest trial</span>
    <span>&#9656; range runs past the edge of the chart</span></div>` + svg;
}

function chartRows(metric, lowerIsBetter) {
  return DATA.providers.map(p => {
    const vals = {};
    DATA.cases.forEach(c => vals[c] = DATA.cells[p.name + '|' + c][metric]);
    // Rank on the endpoint's median across both cases, so the chart leads with
    // the best option. An endpoint with no answered trial sorts last.
    return {label: p.short, vals, score: p.pooled[metric]};
  }).sort((a, b) => {
    if (a.score === null) return 1;
    if (b.score === null) return -1;
    return (lowerIsBetter ? 1 : -1) * (a.score - b.score);
  });
}

// ---------- draw the charts ----------
// Each chart puts its best option at the top. Writing fast is best when the
// number is high; waiting and finishing are best when the number is low.
barChart(document.getElementById('chart-tps'), chartRows('gen_tps', false),
  {title: 'Output tokens a second', tick: 25, axis: t => t, value: v => v.toFixed(0)});

barChart(document.getElementById('chart-ft'), chartRows('first_token', true),
  {title: 'Seconds to the first token', tick: 1, axis: t => t, value: v => v.toFixed(2)});

barChart(document.getElementById('chart-e2e'), chartRows('end_to_end', true),
  {title: 'Seconds for a whole task', tick: 2, axis: t => t, value: v => v.toFixed(2)});

// ---------- dot plot ----------
function dotChart(el) {
  const g = 158, w = 880, rowH = 30;
  // Quickest overall first, matching the charts above.
  const rows = DATA.providers.slice().sort((a, b) => {
    if (a.pooled.end_to_end === null) return 1;
    if (b.pooled.end_to_end === null) return -1;
    return a.pooled.end_to_end - b.pooled.end_to_end;
  }).map(p => ({
    label: p.short,
    trials: DATA.trials.filter(t => t.provider === p.name && t.verdict === 'pass')
  }));
  const height = 26 + rows.length * rowH + 22;
  const lo = Math.log10(1), hi = Math.log10(100);
  const x = v => g + ((Math.log10(Math.max(v, 1)) - lo) / (hi - lo)) * (w - g - 66);
  let svg = `<svg viewBox="0 0 ${w} ${height}" role="img" aria-label="Trial times">`;
  [1, 2, 5, 10, 20, 50, 100].forEach(t => {
    svg += `<line x1="${x(t)}" y1="14" x2="${x(t)}" y2="${height - 20}" stroke="#eef2f7"/>
            <text x="${x(t)}" y="12" font-size="10" fill="#8c99a8" text-anchor="middle">${t}s</text>`;
  });
  rows.forEach((r, i) => {
    const y = 22 + i * rowH + 14;
    svg += `<text x="${g - 12}" y="${y + 4}" font-size="12.5" fill="#14202e" text-anchor="end">${r.label}</text>`;
    const passes = r.trials.filter(t => t.end_to_end_s !== null);
    if (passes.length) {
      svg += `<line x1="${x(Math.min(...passes.map(t => t.end_to_end_s)))}" y1="${y}" x2="${x(Math.max(...passes.map(t => t.end_to_end_s)))}" y2="${y}" stroke="#c9d2dc" stroke-width="2"/>`;
    }
    r.trials.forEach(t => {
      svg += `<circle cx="${x(t.end_to_end_s)}" cy="${y}" r="4.5" fill="${CASE_COLOR[t.case]}" fill-opacity="0.72" stroke="#fff"/>`;
    });
  });
  svg += `<text x="${g}" y="${height - 4}" font-size="11" fill="#8c99a8">each dot is one trial · log scale · further left is faster</text></svg>`;
  el.innerHTML = `<h3>Seconds for a whole task, trial by trial</h3>
    <div class="legend">${DATA.cases.map(c => `<span><i style="background:${CASE_COLOR[c]}"></i>${cap(c)}</span>`).join('')}</div>` + svg;
}
dotChart(document.getElementById('chart-dots'));

// ---------- every trial ----------
(function renderTrials() {
  const cols = ['Endpoint', 'Case', 'Trial', 'Result', 'Wait (s)', 'Total (s)', 'Tokens/s',
                'In tokens', 'Out tokens', 'From cache', 'Tool calls', 'Note'];
  const rows = DATA.trials.map(t => `<tr class="${t.verdict === 'pass' ? '' : 'bad'}">
    <td>${t.short}</td><td>${cap(t.case)}</td><td>${t.trial}</td>
    <td class="${t.verdict === 'pass' ? 'pass' : 'err'}">${t.verdict}</td>
    <td>${fmt(t.first_token_s)}</td><td>${fmt(t.end_to_end_s)}</td><td>${fmt(t.gen_tps, 1)}</td>
    <td>${(t.in_tokens || 0).toLocaleString()}</td><td>${t.out_tokens}</td>
    <td>${(t.cached_tokens || 0).toLocaleString()}</td>
    <td>${t.calls} called, ${t.ok} worked</td>
    <td style="text-align:left;max-width:340px;white-space:normal" class="note">${t.note || t.detail}</td>
  </tr>`).join('');
  document.getElementById('trials').innerHTML =
    `<table><thead><tr>${cols.map(c => `<th style="cursor:default">${c}</th>`).join('')}</tr></thead><tbody>${rows}</tbody></table>`;
})();

document.getElementById('footer').textContent =
  'Built by evals/report_html.py from __TITLE__. The speed numbers cover the trials that '
  + 'answered; the calls the gateway refused still show in the Passed column.';
</script>
</body>
</html>
"""

if __name__ == "__main__":
    main()
