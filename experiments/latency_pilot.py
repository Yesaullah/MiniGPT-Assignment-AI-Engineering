"""Generation latency of a running MiniGPT API (local container or the deployed service), D18c.

python -m experiments.latency_pilot --url http://localhost:8000 --label local_0.1cpu
python -m experiments.latency_pilot --url https://<service>.onrender.com --label render

Requests run one at a time after a few warm-up requests. Two prompt sets are timed separately:
short prompts (speaker cues, mid-line) and a full-window prompt longer than the context, because
with a short prompt many generation steps process fewer than T positions and look cheaper.
"generation" is the server's own timing (model already loaded); "request" adds HTTP and queueing.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import time
import urllib.request

SHORT_PROMPTS = ["ROMEO:\n", "First Citizen:\n", "KING RICHARD III:\n", "O, what a", "My lord, I"]
FULL_WINDOW_PROMPT = (  # the corpus opening, longer than the 128-character context
    "First Citizen:\nBefore we proceed any further, hear me speak.\n\nAll:\nSpeak, speak.\n\n"
    "First Citizen:\nYou are all resolved rather to die than to famish?\n\nAll:\nResolved. resolved.\n\n"
    "First Citizen:\nFirst, you know Caius Marcius is chief enemy to the people.\n")


def post(url: str, body: dict, timeout: float = 600) -> tuple[dict, float]:
    request = urllib.request.Request(url + "/generate", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    start = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = json.load(response)
    return data, time.perf_counter() - start


def p95(values: list[float]) -> float:
    """Nearest-rank 95th percentile (D24h)."""
    ordered = sorted(values)
    return ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://localhost:8000")
    p.add_argument("--label", default="local")
    p.add_argument("--tokens", default="100,300", help="max_new_tokens values to time")
    p.add_argument("--seeds", default="0,1", help="sampling seeds per prompt at the first length")
    p.add_argument("--warmup", type=int, default=2)
    args = p.parse_args()
    url = args.url.rstrip("/")
    with urllib.request.urlopen(url + "/model-info", timeout=600) as response:  # also wakes a sleeping service
        info = json.load(response)
    print(f"{url}: {info['parameters']:,} parameters, context {info['config']['context_length']}")
    for i in range(args.warmup):
        post(url, {"prompt": "ROMEO:\n", "max_new_tokens": 20, "seed": i})

    lengths = [int(n) for n in args.tokens.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]
    results = []
    for k, n in enumerate(lengths):
        use_seeds = seeds if k == 0 else seeds[:1]  # longer outputs: one seed per prompt keeps it short
        for prompt_set, prompts in (("short", SHORT_PROMPTS), ("full_window", [FULL_WINDOW_PROMPT])):
            for prompt in prompts:
                for seed in use_seeds:
                    data, wall = post(url, {"prompt": prompt, "max_new_tokens": n, "temperature": 0.8,
                                            "seed": seed})
                    results.append({"max_new_tokens": n, "prompt_set": prompt_set, "seed": seed,
                                    "generation_s": data["seconds"], "request_s": wall,
                                    "chars": len(data["generated"])})
                    print(f"{n:4d} tokens | {prompt_set:<11} | {prompt[:18]!r:<22} seed {seed}: "
                          f"generation {data['seconds']:.2f} s, request {wall:.2f} s")

    summary = []
    for n in lengths:
        for prompt_set in ("short", "full_window"):
            rows = [r for r in results if r["max_new_tokens"] == n and r["prompt_set"] == prompt_set]
            gen = [r["generation_s"] for r in rows]
            summary.append({"max_new_tokens": n, "prompt_set": prompt_set, "requests": len(rows),
                            "generation_median_s": statistics.median(gen), "generation_p95_s": p95(gen),
                            "request_p95_s": p95([r["request_s"] for r in rows]),
                            "chars_per_s": sum(r["chars"] for r in rows) / sum(gen)})
    lines = [f"Latency pilot: {args.label} ({url}); warm, one request at a time, temperature 0.8", "",
             "| new tokens | prompts | requests | generation median (s) | generation p95 (s) "
             "| request p95 (s) | chars/s |", "|---:|---|---:|---:|---:|---:|---:|"]
    lines += [f"| {s['max_new_tokens']} | {s['prompt_set']} | {s['requests']} | "
              f"{s['generation_median_s']:.2f} | {s['generation_p95_s']:.2f} | {s['request_p95_s']:.2f} | "
              f"{s['chars_per_s']:.1f} |" for s in summary]
    os.makedirs("outputs/evidence", exist_ok=True)
    with open(f"outputs/evidence/latency_{args.label}.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    with open(f"outputs/evidence/latency_{args.label}.json", "w", encoding="utf-8") as f:
        json.dump({"label": args.label, "url": url, "model": info, "summary": summary,
                   "requests": results}, f, indent=2)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
