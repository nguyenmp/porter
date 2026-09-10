# How the endpoint report works

`report_html.py` reads `results.jsonl` and writes one HTML page,
`results_report.html`, that compares endpoints and recommends which to use.
This file explains what the report measures, why, and how to rebuild it after
a new run.

One naming note: `results.jsonl` calls each endpoint a `provider`. The page
calls it an endpoint. Same thing.

## The three measures we rank on

We rank each endpoint on three measures. They are the only ones the endpoint
itself controls.

1. **Tokens a second** — how fast the endpoint writes. Higher is better.
   The runner reads this from the stream: output tokens divided by the time
   the model spent generating. Tool time is not counted, so this is the
   model's speed, not the agent's.
2. **Wait (s)** — how long before the first token arrives. Lower is better.
   This is the blank wait you notice while the answer is still empty. It
   covers the first turn only, because that is all the runner records.
3. **From cache** — the share of input tokens the endpoint served from its
   prompt cache. Higher is better, because cached tokens cost less and arrive
   faster. One endpoint (mancer-fp8) reported none; it pays full price for
   about 10,900 input tokens on every turn.

Every number on the page is a **median**: the middle trial of the five, not
the average. One slow call moves a median less than it moves an average.

## What we do not rank on, and why

**Total time** stays on the page as context but is not ranked. It includes the
tool calls, the local startup, and the wait before any later turn the model
chooses to take. It also rewards an endpoint that writes a shorter answer.
None of those are the endpoint's speed.

We tried a "not writing" measure once. It failed: the leftover time turned out
to be the wait before a second turn starts writing, which exists only when the
model calls a tool. That measure was really "did the model answer in one
turn." Do not bring it back.

**Correctness** gives no signal in this run: every endpoint that answered
passed both cases. If a future run fails some cases, correctness becomes a
hard filter before any ranking.

## How the recommendation works

1. For each endpoint and each measure, take the median of the trials that
   answered. Calls the gateway refused stay out of the speed numbers.
2. An endpoint is in the **better half** when its median beats the middle
   endpoint's median. With nine endpoints, that means rank 1 to 4.
3. To check that a mark is not luck, the report reruns the ranking 2,000
   times. Each round draws five trials at random, repeats allowed, then
   recomputes the medians and re-ranks. A mark that stays in the better half
   in at least 9 of 10 rounds is shown bold.
4. The headline recommends the endpoints in the better half of all three
   measures and names the ones in none.

Five trials per endpoint, all inside one hour, is a small sample. Treat the
recommendation as a shortlist to test, not a verdict.

## How to rebuild the report

1. Get a `results.jsonl`. Run the eval as `README.md` describes:
   `python3 run_eval.py`.
2. Build the page:

   ```
   python3 report_html.py --in results.jsonl --out results_report.html
   ```

3. Open `results_report.html` in a browser. The page is one file: no server,
   no network, no build step.

That is all. If a provider came back refused (like digitalocean in the first
run), the page counts the refusals but keeps them out of the speed numbers. A
refused call says nothing about how fast the endpoint is.
