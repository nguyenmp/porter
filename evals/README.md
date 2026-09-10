# Evals for porter

Run the same prompts against several endpoints and compare them. An endpoint
here means one provider's server for a model, at one precision. Porter does the
agent work; this folder scores the results. It is separate from the porter
binary, so you can test any endpoint without a code change.

## Point porter at your gateway, not at OpenRouter

Porter talks to whatever `PORTER_BASE_URL` names. Point it at your LiteLLM
proxy so the proxy holds the credentials, and porter never needs an OpenRouter
key of its own.

```
PORTER_BASE_URL=https://litellm.href.cat/v1
PORTER_MODEL=openrouter/deepseek/deepseek-v4-flash
PORTER_API_KEY=<your LiteLLM key>
```

Two things that cost me real debugging time:

- **The model name needs the gateway's routing prefix.** LiteLLM's convention
  for routing a model to OpenRouter is `openrouter/<model>`. Without the
  prefix, the same-looking name can point somewhere else entirely. On this
  gateway `deepseek/deepseek-v4-flash` goes straight to DeepSeek's own API.
- **A backend with no providers ignores your pin.** Send a `PORTER_PROVIDER`
  pin with the non-prefixed name and you get a 200 that ignored the pin. With
  the prefixed name, the same pin either works or fails with an error. So check
  which backend the name routes to before trusting a pin.

The eval driver checks a few things up front for this reason: it refuses to
start if a non-local address has no key, and it prints where each provider's
key came from.

## What one test entry covers

A model is not one thing to test. Many providers serve one model, and a provider
often serves it at several precisions (a *quantization*: fp8, bf16, fp4). Those
are different endpoints with different price, speed, and quality, even though
the model name is the same. So what you test here is the triple:

- **model** — for example `deepseek/deepseek-v4-flash`
- **provider slug** — for example `deepinfra/fp8`
- **precision** — fp8, bf16, fp4, fp16, or unknown

One `providers.yaml` entry pins one such triple. Run a model's entries, see
which endpoints answer well and quickly, then repeat for the next model.

## What it measures

Each run scores five things, all read from the JSONL the one-shot CLI prints:

- **Right answer**: a check per case. Two exist today (time, machine).
- **Tools work**: the shell tool adds an `exit code: N` line, so the runner
  reads success without guessing. It also counts how many tools the agent
  called.
- **No language mixing**: not yet checked. Add a checker that flags an answer
  using words outside the prompt's language.
- **Tokens per second**: output tokens divided by the time the model spent
  generating. The runner works this out from the stream, so tool time does not
  count.
- **End-to-end time**: wall clock around the whole run, plus time to first
  token.

## How to run it

1. Build porter (macOS):
   ```sh
   cd .. && go build -o porter-macos ./cmd/porter
   ```
2. Copy the provider config and fill in your keys:
   ```sh
   cp providers.yaml.example providers.yaml
   ```
3. Run the evals:
   ```sh
   python3 run_eval.py
   ```
4. Read the table:
   ```sh
   python3 report.py
   ```

## Finding the endpoints to test

List the endpoints that serve a model, with price, precision, and uptime:

```sh
python3 list_providers.py deepseek/deepseek-v4-flash
```

```
slug                       provider         precision      ctx  tools    up30m    in $/M   out $/M
--------------------------------------------------------------------------------------------------
digitalocean               DigitalOcean     unknown    1048576    yes    99.8%     0.068     0.168
streamlake/fp8             StreamLake       fp8        1024000    yes    76.2%     0.087     0.174
baidu/fp8                  Baidu            fp8        1048576    yes    97.3%     0.088     0.177
deepinfra/fp8              DeepInfra        fp8        1048576    yes    99.7%     0.090     0.180
...
15 endpoints shown (1 repeated record, 5 without tool support hidden)
```

By default the table hides endpoints that cannot call tools, because every case
here needs tools. Useful options:

```sh
# only one precision, and only endpoints that are up
python3 list_providers.py openai/gpt-oss-120b --quantization fp8 --min-uptime 99

# write one eval entry per endpoint, ready to run
python3 list_providers.py deepseek/deepseek-v4-flash --quantization fp8 \
    --limit 4 --write providers.generated.yaml
```

### Probe before you run: the catalog is not your account

OpenRouter's list shows what *could* serve a model, not what *your account* may
use. Guardrails, a data policy, or a zero-data-retention setting can remove
endpoints, and a pinned request to one comes back 404. In a results file that
reads as the model failing the case, when the endpoint was never available at
all.

`--probe` sends one tiny pinned request per endpoint and drops the ones that
cannot answer:

```
$ python3 list_providers.py deepseek/deepseek-v4-flash --probe --write providers.generated.yaml
probing 6 endpoints against https://litellm.href.cat/v1 ...
  ok       digitalocean               DigitalOcean
  BLOCKED  streamlake/fp8             excluded by your zero-data-retention setting
  BLOCKED  baidu/fp8                  excluded by your zero-data-retention setting
  ok       deepinfra/fp8              DeepInfra
  BLOCKED  gmicloud/fp8               excluded by your zero-data-retention setting
  ok       siliconflow/fp8            SiliconFlow

3 of 6 endpoints answer this account
```

On one run of this model, three of six fp8 endpoints were blocked for that
reason. Without the probe, half the grid would have failed with 404s.

The generated file needs no key of its own: export `PORTER_API_KEY` and the
driver passes it through.

Two warnings about the table:

- `throughput_30m` and `latency_30m` are usually null, so do not sort by them.
  Measure speed with the eval instead. `up30m` does carry a value.
- A provider slug can cover more than one precision (`deepinfra/bf16` and
  `deepinfra/turbo` are both bf16, `deepinfra/fp8` is fp8). Pin the full slug
  from the table rather than the provider name, so the entry names exactly one
  endpoint.

## Adding a provider

Add an entry to `providers.yaml`. The driver starts one `porter server` per
entry with that entry's env (`PORTER_BASE_URL`, `PORTER_MODEL`,
`PORTER_PROVIDER`, `PORTER_API_KEY`), runs every case a few times, and shuts it
down. Add `external: true` and set `env.PORTER_SERVER_URL` if a server is
already running and you do not want the driver to manage one.

`PORTER_PROVIDER` pins an endpoint. Turn fallbacks off, or the measurement
means nothing:

```yaml
  - name: deepseek-v4-flash-deepinfra-fp8
    env:
      PORTER_BASE_URL: "https://litellm.href.cat/v1"
      PORTER_MODEL: "openrouter/deepseek/deepseek-v4-flash"
      PORTER_PROVIDER: "deepinfra/fp8"
    addr: "127.0.0.1:8801"
    trials: 5
```

A plain slug pins that endpoint and turns fallbacks off, so a run measures it or
fails. For finer control, write the routing object itself:

```yaml
      PORTER_PROVIDER: '{"only":["deepinfra/fp8"],"allow_fallbacks":false,"quantizations":["fp8"]}'
```

Quote it. Unquoted, YAML reads that JSON as a map instead of a string, and the
driver stops with a clear message rather than starting a server with a broken
environment. A misspelled field also stops the server at startup instead of
being ignored, which would send the run somewhere you did not intend.

## Adding a case

Make a folder under `cases/` with a `case.yaml`:

```yaml
prompt: "What time is it? Give the current date and time."
check: time
```

`check` names a checker in `run_eval.py` (see `CHECKERS`). To score open-ended
answers such as weather, summaries, and plain-language rewrites, add a checker
that calls a fixed judge model with a short rubric.

## Files

| File | What it does |
|---|---|
| `run_eval.py` | Runs providers x cases x trials, parses JSONL, scores, writes `results.jsonl` |
| `report.py` | Turns `results.jsonl` into a table |
| `list_providers.py` | Lists the endpoints that serve a model, and writes eval entries for them |
| `providers.yaml` | Your provider list and keys (gitignored) |
| `providers.generated.yaml` | Entries written by `list_providers.py` (gitignored) |
| `test_checkers.py` | Tests for the checkers, run with `python3 test_checkers.py` |
| `cases/*/case.yaml` | A prompt and which checker scores it |
| `.eval-work/` | Per-provider server DBs, sandboxes, and logs (gitignored) |

Each run keeps its porter `session id` in the results row, and its sandbox
lives under `.eval-work/<provider>`. So a failure opens straight back into
porter's web UI for a closer look.

## Notes

- Run each case several times: network and model noise dominate a single run.
  `trials: 5` is a good start, and the spread between runs is often the real
  finding.
- Keep prompts in English. DeepSeek-flash answers in Chinese when tempted, and
  language faithfulness is one of your goals.
- Weather and gas prices change each run, so their checker should grade "did it
  get a sensible answer" instead of comparing to a saved number.
- A wrong model name comes back as a failed turn, with `error` set in the row,
  not as a pass.
- A pinned endpoint that is down fails the run too. That result is honest:
  uptime is part of what you measure. Watch the `error` column for it, so a
  provider outage does not read as a bad answer.
- Keep the key exported where you run the eval. The shell's `PORTER_API_KEY` and
  the one in `~/.zshrc` were different at one point, and the stale one was
  rejected with a 401 while the good one worked. When a whole grid fails with
  one error, check the key before the model.
- Run `python3 test_checkers.py` after you change a checker. A checker that
  misreads a correct answer makes every result wrong in the same way, which is
  hard to spot in a table of numbers.
