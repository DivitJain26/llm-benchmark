#!/usr/bin/env python3
"""
Concurrent vLLM load benchmark — companion to bench_vllm.py.

Fires many requests at once (thread pool) and reports the numbers that only exist under load:
aggregate throughput, requests/s, TTFT + latency percentiles, and GPU utilisation / memory
sampled across the WHOLE batch instead of per request.

Same flags as bench_vllm.py, plus:
  --concurrency N     requests in flight at once                 (default 4)
  --requests M        total requests per prompt                  (default = concurrency)
  --mix               one batch cycling through ALL selected prompts (realistic mixed load)
  --all-responses     write every response to the log (default: only the first)

Logs go to logs/<prompt>_concurrent.txt (or logs/mixed_concurrent.txt with --mix); --log overrides.

Examples:
  python bench_vllm_concurrent.py --concurrency 8
  python bench_vllm_concurrent.py --prompt-file summary --concurrency 16 --requests 64 --thinking off
  python bench_vllm_concurrent.py --mix --concurrency 8 --requests 32 --runs 3
"""

import math
import os
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_vllm as base                                        # noqa: E402
from bench_vllm import (GpuSampler, build_parser, choose_model, fmt, list_models,   # noqa: E402
                        load_jobs, make_gpu_sampler, resolve_dirs, run_once, short_text)


# ----------------------------------------------------------------------------- helpers

def pct(vals, p):
    """Linear-interpolated percentile of a list (p in 0..100)."""
    if not vals:
        return None
    s = sorted(vals)
    k = (len(s) - 1) * p / 100.0
    lo, hi = math.floor(k), math.ceil(k)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def stats(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return {"mean": None, "p50": None, "p95": None, "min": None, "max": None}
    return {"mean": statistics.fmean(vals), "p50": pct(vals, 50), "p95": pct(vals, 95),
            "min": min(vals), "max": max(vals)}


def stat_line(label, st, nd=3, unit=""):
    return (f"  {label:<21}: {fmt(st['mean'], nd)} / {fmt(st['p50'], nd)} / "
            f"{fmt(st['p95'], nd)} / {fmt(st['max'], nd)}{unit}")


# ----------------------------------------------------------------------------- batch

def run_batch(args, model, requests_list, gpu):
    """
    requests_list: [(name, prompt, ppath), ...] — one entry per request to send.
    Returns (summary_dict, results_list). GPU is sampled across the whole batch.
    """
    results = []
    if gpu:
        gpu.start()
    t0 = time.perf_counter()

    def one(name, prompt):
        start = time.perf_counter() - t0
        m, resp, reas = run_once(args.url, args.api_key, model, prompt, args.system,
                                 args.max_tokens, args.temperature, None, args.thinking)
        m["start_s"] = start
        m["end_s"] = time.perf_counter() - t0
        return m, resp, reas

    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = {ex.submit(one, name, prompt): (i, name) for i, (name, prompt, _) in enumerate(requests_list)}
        done = 0
        for fut in as_completed(futs):
            i, name = futs[fut]
            done += 1
            try:
                m, resp, reas = fut.result()
                results.append({"idx": i, "name": name, "ok": True, "m": m,
                                "response": resp, "reasoning": reas})
                print(f"  [{done}/{len(requests_list)}] {name}: ttft {fmt(m['ttft_s'])}s  "
                      f"total {fmt(m['total_time_s'])}s  {fmt(m['completion_tokens'])} tok  "
                      f"{fmt(m['generation_tokens_per_s'], 1)} tok/s")
            except Exception as e:                                # noqa: BLE001
                results.append({"idx": i, "name": name, "ok": False, "error": repr(e)})
                print(f"  [{done}/{len(requests_list)}] {name}: FAILED {e!r}")
    wall = time.perf_counter() - t0
    if gpu:
        gpu.stop()

    results.sort(key=lambda r: r["idx"])
    ok = [r["m"] for r in results if r["ok"]]
    prompt_tok = sum(m["prompt_tokens"] or 0 for m in ok)
    comp_tok = sum(m["completion_tokens"] or 0 for m in ok)

    summary = {
        "model": model,
        "concurrency": args.concurrency,
        "requests": len(requests_list),
        "ok": len(ok),
        "failed": len(results) - len(ok),
        "wall_s": wall,
        "requests_per_s": len(ok) / wall if wall > 0 else None,
        "prompt_tokens_total": prompt_tok,
        "completion_tokens_total": comp_tok,
        "total_tokens": prompt_tok + comp_tok,
        "prompt_tokens_avg": prompt_tok / len(ok) if ok else None,
        "completion_tokens_avg": comp_tok / len(ok) if ok else None,
        "completion_tok_per_s": comp_tok / wall if wall > 0 else None,      # aggregate generation throughput
        "total_tok_per_s": (prompt_tok + comp_tok) / wall if wall > 0 else None,
        "ttft": stats([m["ttft_s"] for m in ok]),
        "latency": stats([m["total_time_s"] for m in ok]),
        "gen_tps": stats([m["generation_tokens_per_s"] for m in ok]),       # per-request decode speed
        "gpu": gpu.summary() if gpu else None,
    }
    return summary, results


def average_summaries(sums):
    """Average the scalar fields of several batch summaries (for --runs > 1)."""
    avg = dict(sums[0])
    scalar = ["wall_s", "requests_per_s", "prompt_tokens_total", "completion_tokens_total", "total_tokens",
              "prompt_tokens_avg", "completion_tokens_avg", "completion_tok_per_s", "total_tok_per_s",
              "ok", "failed"]
    for k in scalar:
        vals = [s[k] for s in sums if s[k] is not None]
        avg[k] = sum(vals) / len(vals) if vals else None
    for k in ("ttft", "latency", "gen_tps"):
        avg[k] = {sk: (statistics.fmean([s[k][sk] for s in sums if s[k][sk] is not None])
                       if any(s[k][sk] is not None for s in sums) else None)
                  for sk in ("mean", "p50", "p95", "min", "max")}
    avg["gpu"] = None      # per-batch GPU numbers are listed individually in the log
    return avg


# ----------------------------------------------------------------------------- output

def summary_lines(s, gpu_interval):
    L = []
    L.append("[load]")
    L.append(f"  concurrency          : {s['concurrency']}")
    L.append(f"  requests             : {s['requests']}  (ok {fmt(s['ok'], 0)}, failed {fmt(s['failed'], 0)})")
    L.append(f"  wall time            : {fmt(s['wall_s'])} s")
    L.append(f"  requests/s           : {fmt(s['requests_per_s'], 2)}")
    L.append("")
    L.append("[throughput]  aggregate over the whole batch (this is the number that scales with load)")
    L.append(f"  completion tok/s     : {fmt(s['completion_tok_per_s'], 1)}")
    L.append(f"  total tok/s          : {fmt(s['total_tok_per_s'], 1)}   (prompt + completion)")
    L.append(f"  prompt tokens        : {fmt(s['prompt_tokens_total'], 0)} total   ({fmt(s['prompt_tokens_avg'], 0)} avg / request)")
    L.append(f"  completion tokens    : {fmt(s['completion_tokens_total'], 0)} total   ({fmt(s['completion_tokens_avg'], 0)} avg / request)")
    L.append("")
    L.append("[latency]  per request         mean / p50 / p95 / max")
    L.append(stat_line("time to first token", s["ttft"], 3, " s"))
    L.append(stat_line("total time", s["latency"], 3, " s"))
    L.append(stat_line("generation tok/s", s["gen_tps"], 1, "   (per request, drops as concurrency rises)"))
    if s.get("gpu") is not None:
        L.append("")
        L.append(f"[gpu]  sampled every {gpu_interval}s across the whole batch with nvidia-smi")
        L.extend(GpuSampler.lines(s["gpu"]))
    return L


def per_request_lines(results):
    L = ["[per request]   #   start_s   ttft_s   total_s   prompt_tok   comp_tok   gen_tok_s   prompt"]
    for r in results:
        if r["ok"]:
            m = r["m"]
            L.append(f"  {r['idx'] + 1:>3}   {fmt(m['start_s']):>7}   {fmt(m['ttft_s']):>6}   {fmt(m['total_time_s']):>7}   "
                     f"{fmt(m['prompt_tokens']):>10}   {fmt(m['completion_tokens']):>8}   "
                     f"{fmt(m['generation_tokens_per_s'], 1):>9}   {r['name']}")
        else:
            L.append(f"  {r['idx'] + 1:>3}   FAILED  {r['error']}   {r['name']}")
    return L


def write_log(path, label, jobs, args, avg, batches, gpu_interval):
    """batches: [(summary, results), ...] one per --runs."""
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    L = []
    L.append("=" * 72)
    head = (f"{ts}  |  model: {avg['model']}  |  prompt: {label}  |  concurrency: {avg['concurrency']}  |  "
            f"requests: {avg['requests']}  |  runs: {len(batches)}")
    if args.thinking:
        head += f"  |  thinking: {args.thinking}"
    L.append(head)
    L.append(f"max_tokens: {args.max_tokens}  |  temperature: {args.temperature}")
    L.append("=" * 72)
    L.append("")
    if len(batches) > 1:
        L.append(f"[average of {len(batches)} batches]")
        L.append("")
    L.extend(summary_lines(avg, gpu_interval))
    for i, (s, results) in enumerate(batches, 1):
        L.append("")
        if len(batches) > 1:
            L.append(f"--- batch {i} ---")
            L.extend(summary_lines(s, gpu_interval))
            L.append("")
        L.extend(per_request_lines(results))
    if args.system:
        L.append("")
        L.append("[system]")
        L.append(short_text(args.system))
    L.append("")
    for name, prompt, ppath in jobs:
        words = len(prompt.split())
        note = f", first {base.LOG_WORDS} shown" if words > base.LOG_WORDS else ""
        L.append(f"[prompt]  {('file: ' + ppath) if ppath else 'inline'}  ({words} words{note})")
        L.append(short_text(prompt))
        L.append("")
    # responses: first successful one of the last batch (all of them with --all-responses)
    last = batches[-1][1]
    ok = [r for r in last if r["ok"]]
    to_show = ok if args.all_responses else ok[:1]
    for r in to_show:
        tag = f"request {r['idx'] + 1} of {len(last)}" + ("" if args.all_responses else " (others omitted; use --all-responses)")
        if r["reasoning"]:
            L.append(f"[reasoning]  {tag}")
            L.append(r["reasoning"].rstrip())
            L.append("")
        L.append(f"[response]  {tag}")
        L.append(r["response"].rstrip())
        L.append("")
    L.append("")
    with open(path, "a", encoding="utf-8") as f:
        f.write("\n".join(L))


# ----------------------------------------------------------------------------- main

def main():
    p = build_parser("Concurrent vLLM load benchmark (same flags as bench_vllm.py)")
    p.add_argument("--concurrency", type=int, default=4, help="requests in flight at once")
    p.add_argument("--requests", type=int, default=None,
                   help="total requests per prompt (default: same as --concurrency)")
    p.add_argument("--mix", action="store_true",
                   help="one batch cycling through all selected prompts instead of one batch per prompt")
    p.add_argument("--all-responses", action="store_true", help="log every response (default: only the first)")
    args = p.parse_args()
    base.LOG_WORDS = args.log_words
    if args.requests is None:
        args.requests = args.concurrency
    if args.concurrency < 1 or args.requests < 1:
        sys.exit("--concurrency and --requests must be >= 1")

    if args.list_models:
        for m in list_models(args.url, args.api_key):
            print(m)
        return

    prompt_dir, log_dir = resolve_dirs(args)
    jobs = load_jobs(args, prompt_dir)
    model = choose_model(args)
    gpu = make_gpu_sampler(args)

    print(f"using model: {model}")
    print(f"concurrency: {args.concurrency}   requests per batch: {args.requests}   runs: {args.runs}"
          + (f"   thinking: {args.thinking}" if args.thinking else "")
          + (f"   gpu sampling: {args.gpu_interval}s" if gpu else "   gpu sampling: off"))

    # group jobs -> one batch per prompt, or a single mixed batch
    if args.mix:
        groups = [("mixed", jobs, [jobs[i % len(jobs)] for i in range(args.requests)])]
    else:
        groups = [(name, [(name, prompt, ppath)], [(name, prompt, ppath)] * args.requests)
                  for name, prompt, ppath in jobs]

    for label, group_jobs, requests_list in groups:
        print(f"\n################ {label}  ({args.requests} requests x {args.runs} run(s)) ################")
        batches = []
        for i in range(args.runs):
            if args.runs > 1:
                print(f"\n-- batch {i + 1}/{args.runs} --")
            s, results = run_batch(args, model, requests_list, gpu)
            batches.append((s, results))
            print()
            print("\n".join(summary_lines(s, args.gpu_interval)))

        avg = average_summaries([b[0] for b in batches]) if args.runs > 1 else batches[0][0]
        if args.runs > 1:
            print(f"\n=== average of {args.runs} batches ===")
            print("\n".join(summary_lines(avg, args.gpu_interval)))

        if not args.quiet:
            first = next((r for r in batches[-1][1] if r["ok"]), None)
            if first:
                if first["reasoning"] and args.show_reasoning:
                    print(f"\n--- reasoning (request {first['idx'] + 1}) ---")
                    print(first["reasoning"])
                print(f"\n--- response (request {first['idx'] + 1}) ---")
                print(first["response"])

        log_path = args.log or os.path.join(log_dir, f"{label}_concurrent.txt")
        write_log(log_path, label, group_jobs, args, avg, batches, args.gpu_interval)
        print(f"\nlogged to {os.path.relpath(log_path)}")


if __name__ == "__main__":
    main()