#!/usr/bin/env python3
"""List the endpoints that serve a model, and write eval entries for them.

OpenRouter reports one record per endpoint, with the provider slug, the
precision (quantization), price, context length, uptime, and which request
parameters the endpoint supports. One model often has many endpoints across
many providers, and one provider can serve several precisions. So what you test
is the triple of model, provider slug, and precision.

The list needs no API key:

    GET https://openrouter.ai/api/v1/models/{model-id}/endpoints

Examples:

    python3 list_providers.py deepseek/deepseek-v4-flash
    python3 list_providers.py deepseek/deepseek-v4-flash --quantization fp8
    python3 list_providers.py openai/gpt-oss-120b --write providers.generated.yaml
"""

import argparse
import json
import os
import sys
import urllib.error
import urllib.request

# The endpoint list always comes from OpenRouter: it is the only place that
# knows which provider serves which precision, and the call needs no key. The
# base URL written into generated entries is a different thing: it is the
# gateway porter talks to, such as LiteLLM forwarding to OpenRouter.
ENDPOINTS_URL = "https://openrouter.ai/api/v1/models/%s/endpoints"
DEFAULT_BASE_URL = "https://litellm.href.cat/v1"
# MODEL_PREFIX goes in front of a model name to make the gateway route it to
# OpenRouter. Without it the same-looking name can point elsewhere: on this
# gateway "deepseek/deepseek-v4-flash" goes straight to DeepSeek, where a
# provider pin does nothing.
MODEL_PREFIX = "openrouter/"


def fetch(model):
    """Return the endpoint list for one model."""
    url = ENDPOINTS_URL % model
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            data = json.load(r)["data"]
    except Exception as e:
        sys.exit("could not read %s: %s" % (url, e))
    return data.get("endpoints") or []


def price_per_million(endpoint, which):
    try:
        return float(endpoint.get("pricing", {}).get(which) or 0) * 1_000_000
    except (TypeError, ValueError):
        return None


def has_tools(endpoint):
    return "tools" in (endpoint.get("supported_parameters") or [])


def row_of(endpoint, model):
    return {
        "model": model,
        "slug": endpoint.get("tag") or "",
        "provider": endpoint.get("provider_name") or "",
        "quantization": endpoint.get("quantization") or "unknown",
        "context": endpoint.get("context_length"),
        "tools": has_tools(endpoint),
        # These two are often null: OpenRouter always reports uptime, but
        # throughput and latency only sometimes. Measure speed yourself instead
        # of ranking on these.
        "uptime_30m": endpoint.get("uptime_last_30m"),
        "throughput_30m": endpoint.get("throughput_last_30m"),
        "latency_30m": endpoint.get("latency_last_30m"),
        "status": endpoint.get("status"),
        "in_per_m": price_per_million(endpoint, "prompt"),
        "out_per_m": price_per_million(endpoint, "completion"),
    }


def dedupe(rows):
    """Drop repeated records. OpenRouter sometimes lists the same endpoint twice,
    showing one row per region, for example. Two entries with one name would
    clash in the generated file, so keep the record with the higher uptime.
    """
    best = {}
    for r in rows:
        key = (r["model"], r["slug"])
        seen = best.get(key)
        if seen is None or (r["uptime_30m"] or 0) > (seen["uptime_30m"] or 0):
            best[key] = r
    return list(best.values())


def probe(row, base_url, api_key, model_prefix, timeout=60):
    """Send a tiny pinned request and report whether this endpoint answers.

    OpenRouter's list includes endpoints your account cannot use: guardrails, a
    data policy, or a provider blocklist can remove them. A request to one
    comes back 404, which in a results file reads as the model failing rather
    than as a setup problem. One 3-token request tells the two apart before the
    real run.
    """
    body = json.dumps({
        "model": model_prefix + row["model"],
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 3,
        "provider": {"only": [row["slug"]], "allow_fallbacks": False},
    }).encode()
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions", method="POST", data=body,
        headers={"Authorization": "Bearer " + api_key,
                 "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            payload = json.load(r)
        served = payload.get("provider") or row["provider"]
        return True, served
    except urllib.error.HTTPError as e:
        raw = e.read().decode()
        try:
            msg = json.loads(raw).get("error", {}).get("message", raw)
        except ValueError:
            msg = raw
        msg = " ".join(str(msg).split())
        # Name the usual reasons in plain words, because the raw text is long.
        for needle, reason in (
            ("ZDR violation", "excluded by your zero-data-retention setting"),
            ("guardrail", "excluded by a guardrail on your account"),
            ("data policy", "excluded by your data policy"),
        ):
            if needle.lower() in msg.lower():
                return False, reason
        if "No allowed providers" in msg:
            return False, "not currently served"
        return False, msg[:160]
    except Exception as e:
        return False, "probe failed: %s" % e


def keep(row, args):
    if args.supports_tools and not row["tools"]:
        return False
    if args.quantization:
        wanted = [q.strip().lower() for q in args.quantization.split(",") if q.strip()]
        if row["quantization"].lower() not in wanted:
            return False
    if args.max_price_in is not None:
        if row["in_per_m"] is None or row["in_per_m"] > args.max_price_in:
            return False
    if args.min_uptime is not None:
        if row["uptime_30m"] is None or row["uptime_30m"] < args.min_uptime:
            return False
    return True


def sort_rows(rows, key):
    if key == "uptime":
        return sorted(rows, key=lambda r: -(r["uptime_30m"] or 0))
    if key == "slug":
        return sorted(rows, key=lambda r: r["slug"])
    return sorted(rows, key=lambda r: (r["in_per_m"] if r["in_per_m"] is not None else 1e9))


def print_table(rows):
    header = "%-26s %-16s %-9s %8s %6s %8s %9s %9s" % (
        "slug", "provider", "precision", "ctx", "tools", "up30m", "in $/M", "out $/M")
    print(header)
    print("-" * len(header))
    for r in rows:
        print("%-26s %-16s %-9s %8s %6s %8s %9s %9s" % (
            r["slug"][:26], (r["provider"] or "")[:16], r["quantization"],
            r["context"] or "-", "yes" if r["tools"] else "NO",
            ("%.1f%%" % r["uptime_30m"]) if r["uptime_30m"] is not None else "-",
            ("%.3f" % r["in_per_m"]) if r["in_per_m"] is not None else "-",
            ("%.3f" % r["out_per_m"]) if r["out_per_m"] is not None else "-"))


def gateway_model(model):
    """The model name as the gateway expects it.

    An OpenRouter model id needs the gateway's routing prefix, or the request
    can land on a different backend that ignores the pin.
    """
    return model if model.startswith(MODEL_PREFIX) else MODEL_PREFIX + model


def entry_name(row):
    """A short, filesystem-safe name for the model and endpoint."""
    model = row["model"].split("/")[-1]
    slug = row["slug"].replace("/", "-")
    return "%s-%s" % (model, slug)


def write_providers(rows, path, base_url, addr_start, trials, model_prefix=MODEL_PREFIX):
    """Write one eval entry per endpoint.

    Each entry pins its endpoint by slug and turns fallbacks off, so a run
    measures that endpoint or fails. The API key is left out on purpose: the
    driver passes its own environment through, so the key stays in your shell
    and out of this file.
    """
    lines = [
        "# Generated by list_providers.py. One entry per serving endpoint: a",
        "# model, a provider slug, and a precision. Each pins its endpoint with",
        "# no fallbacks, so a run measures it or fails.",
        "#",
    ]
    if model_prefix:
        lines += [
            "# Model names carry the %r prefix. That prefix is what makes the" % model_prefix,
            "# gateway route to OpenRouter; without it the same-looking name can",
            "# point at a different backend, where the pin does nothing.",
            "#",
        ]
    lines += [
        "# The API key is not written here: export PORTER_API_KEY and the driver",
        "# passes it through.",
        "providers:",
    ]
    for i, r in enumerate(rows):
        lines += [
            "  - name: %s" % entry_name(r),
            "    env:",
            "      PORTER_BASE_URL: %r" % base_url,
            "      PORTER_MODEL: %r" % (r["model"] if r["model"].startswith(model_prefix)
                                      else model_prefix + r["model"]),
            # Quote the JSON so YAML reads it as a string. Unquoted, it loads
            # as a map, and the server would get an env value that is not a
            # string.
            "      PORTER_PROVIDER: %s" % json.dumps(json.dumps(pin_for(r))),
            "    addr: '127.0.0.1:%d'" % (addr_start + i),
            "    trials: %d" % trials,
        ]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def pin_for(row):
    """The PORTER_PROVIDER JSON that pins this one endpoint.

    The slug comes from the API, so it names one endpoint, and it already
    carries the precision when a provider serves several (deepinfra/fp8 versus
    deepinfra/turbo). A bare provider name matches all of them.
    """
    return {"only": [row["slug"]], "allow_fallbacks": False}


def prefix_of(args):
    """The model prefix to use, given the flags."""
    return "" if args.direct else MODEL_PREFIX


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("models", nargs="+",
                    help="model ids, for example deepseek/deepseek-v4-flash")
    ap.add_argument("--quantization", default="",
                    help="keep only these precisions, comma-separated "
                         "(for example fp8,bf16)")
    ap.add_argument("--no-supports-tools", dest="supports_tools",
                    action="store_false", default=True,
                    help="do not hide endpoints that cannot call tools")
    ap.add_argument("--max-price-in", type=float, default=None,
                    help="drop endpoints priced above this, in dollars per million input tokens")
    ap.add_argument("--min-uptime", type=float, default=None,
                    help="drop endpoints below this 30-minute uptime percentage")
    ap.add_argument("--sort", default="price", choices=["price", "uptime", "slug"],
                    help="table order (default: price)")
    ap.add_argument("--limit", type=int, default=0, help="show at most this many")
    ap.add_argument("--write", default="", help="write eval provider entries to this file")
    ap.add_argument("--force", action="store_true",
                    help="overwrite --write's target if it already exists")
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL,
                    help="gateway base URL for the generated entries")
    ap.add_argument("--probe", action="store_true",
                    help="send a tiny pinned request to each endpoint and drop "
                         "the ones your account cannot use")
    ap.add_argument("--api-key", default="",
                    help="key for --probe; defaults to PORTER_API_KEY")
    ap.add_argument("--direct", action="store_true",
                    help="write OpenRouter's own URL with unprefixed model names, "
                         "to call OpenRouter directly instead of through a gateway")
    ap.add_argument("--addr-start", type=int, default=8801,
                    help="first local port for the generated entries")
    ap.add_argument("--trials", type=int, default=5,
                    help="trials per case in the generated entries")
    args = ap.parse_args()

    rows = []
    for model in args.models:
        endpoints = fetch(model)
        if not endpoints:
            print("no endpoints listed for %s" % model, file=sys.stderr)
        rows += [row_of(e, model) for e in endpoints]
    listed = len(rows)
    rows = dedupe(rows)
    total = len(rows)
    no_tools = sum(1 for r in rows if not r["tools"])

    kept = [r for r in rows if keep(r, args)]
    kept = sort_rows(kept, args.sort)
    limited = 0
    if args.limit and len(kept) > args.limit:
        limited = len(kept) - args.limit
        kept = kept[:args.limit]
    if not kept:
        sys.exit("no endpoints left after filtering")

    print_table(kept)
    print()

    # Say what was hidden, so a short table is not mistaken for a small
    # catalog. The tools filter is on by default, and hiding endpoints that
    # cannot call tools is worth naming, since every case here needs tools.
    def plural(n, one, many):
        return "%d %s" % (n, one if n == 1 else many)

    hidden = []
    if listed != total:
        hidden.append(plural(listed - total, "repeated record", "repeated records"))
    if args.supports_tools and no_tools:
        hidden.append("%d without tool support" % no_tools)
    if total != len(rows):
        hidden.append("%d filtered out by your options" % (total - len(rows)))
    if limited:
        hidden.append("%d past --limit" % limited)
    print("%d endpoints shown%s" % (
        len(kept), " (" + ", ".join(hidden) + " hidden)" if hidden else ""))

    if args.probe:
        key = os.environ.get("PORTER_API_KEY") or args.api_key
        if not key:
            sys.exit("--probe needs a key: export PORTER_API_KEY or pass --api-key")
        print("probing %d endpoints against %s ..." % (len(kept), args.base_url))
        available = []
        for r in kept:
            ok, info = probe(r, args.base_url, key, prefix_of(args))
            r["available"] = ok
            r["available_note"] = info
            mark = "ok    " if ok else "BLOCKED"
            print("  %-8s %-26s %s" % (mark, r["slug"], info))
            if ok:
                available.append(r)
        print()
        print("%d of %d endpoints answer this account" % (len(available), len(kept)))
        kept = available
        if not kept:
            sys.exit("no endpoints are usable; nothing to write")

    if args.write:
        if os.path.exists(args.write) and not args.force:
            sys.exit("%s exists; pass --force to overwrite" % args.write)
        base_url, prefix = args.base_url, prefix_of(args)
        if args.direct:
            base_url, prefix = "https://openrouter.ai/api/v1", ""
        write_providers(kept, args.write, base_url, args.addr_start, args.trials,
                        model_prefix=prefix)
        print("wrote %d provider entries to %s" % (len(kept), args.write))
        print("run:  python3 run_eval.py --providers %s" % args.write)


if __name__ == "__main__":
    main()
