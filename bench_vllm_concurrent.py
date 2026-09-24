#!/usr/bin/env python3
"""
Concurrent vLLM load benchmark — companion to bench_vllm.py.

Fires many requests at once and reports the numbers that only exist under load:
aggregate throughput, requests/s, TTFT / latency / inter-token-latency percentiles, a throughput
timeline, and GPU utilisation / memory sampled across the WHOLE batch instead of per request.

Same flags as bench_vllm.py, plus:
  --concurrency N     requests in flight at once                 (default 4)
  --requests M        total requests per prompt                  (default = concurrency)
  --duration S        keep N requests in flight for S seconds instead of a fixed count
                      (new requests stop being sent at the deadline; in-flight ones finish)
  --rate R            OPEN LOOP: send R new requests every second for --duration seconds, no matter
                      how many are still in flight (--concurrency is ignored). Shows how the server
                      behaves at a fixed arrival rate: backlog growth, TTFT climbing, errors.
  --max-in-flight M   safety cap for --rate: skip new requests while M are in flight (default 2000)
  --drain S           after the deadline wait at most S seconds for in-flight requests (default: wait)
  --bucket S          timeline bucket size in seconds            (default: auto, ~10-12 buckets)
  --mix               one batch cycling round-robin through EVERY prompt in prompts/
  --all-responses     write every response to the log (default: only the first)

Logs go to logs/<prompt>_concurrent.txt (or logs/mixed_concurrent.txt with --mix); --log overrides.

Examples:
  python bench_vllm_concurrent.py --concurrency 8
  python bench_vllm_concurrent.py --prompt-file summary --concurrency 16 --requests 64 --thinking off
  python bench_vllm_concurrent.py --mix --concurrency 8 --requests 32 --runs 3
  python bench_vllm_concurrent.py --mix --concurrency 16 --duration 120 --thinking off   # 2 min soak
  python bench_vllm_concurrent.py --mix --rate 5 --duration 300 --drain 120             # 5 req/s open loop
"""

import math
import os
import statistics
import sys
import threading
import time
from collections import Counter, OrderedDict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_vllm as base                                        # noqa: E402
from bench_vllm import (GpuSampler, build_parser, choose_model, empty_response_note, fmt,   # noqa: E402
                        list_models, load_jobs, load_prompt_dir, make_gpu_sampler, resolve_dirs,
                        run_once, short_text)

STAT_KEYS = ("mean", "p50", "p95", "p99", "min", "max")
BUCKET_STEPS = (1, 2, 5, 10, 15, 30, 60, 120, 300, 600)


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
        return {k: None for k in STAT_KEYS}
    return {"mean": statistics.fmean(vals), "p50": pct(vals, 50), "p95": pct(vals, 95),
            "p99": pct(vals, 99), "min": min(vals), "max": max(vals)}


def stat_line(label, st, nd=3, unit="", scale=1.0):
    def f(v):
        return fmt(v * scale if v is not None else None, nd)
    return (f"  {label:<21}: {f(st['mean'])} / {f(st['p50'])} / {f(st['p95'])} / "
            f"{f(st['p99'])} / {f(st['max'])}{unit}")


def auto_bucket(span):
    """Bucket size so a timeline of `span` seconds has roughly 10-12 rows."""
    for b in BUCKET_STEPS:
        if span / b <= 12:
            return b
    return BUCKET_STEPS[-1]


def safe_div(a, b):
    return a / b if b else None


# ----------------------------------------------------------------------------- batch

def run_batch(args, model, jobs, gpu):
    """
    jobs: [(name, prompt, ppath), ...] cycled round-robin: request i uses jobs[i % len(jobs)].

    closed loop (default): `args.concurrency` worker threads each keep pulling the next request until
        `args.requests` have been handed out (count mode) or `args.duration` seconds have passed.
    open loop (--rate):    one new request every 1/rate seconds on its own thread, for `args.duration`
        seconds (or `args.requests` in total), regardless of how many are in flight.
    In both modes requests already in flight run to completion after the deadline, up to --drain
    seconds; anything still running after that is reported as unfinished.
    Returns (summary_dict, results_list). GPU is sampled across the whole batch.
    """
    results = []
    lock = threading.Lock()
    counter = {"next": 0, "done": 0, "failed": 0, "skipped": 0, "peak_in_flight": 0, "send_lag_max": 0.0}
    inflight = {}                     # idx -> (name, start_s) for requests currently running
    state = {"closed": False}
    if gpu:
        gpu.start()
    started_at = time.strftime("%Y-%m-%d %H:%M:%S")
    t0 = time.perf_counter()
    deadline = t0 + args.duration if args.duration else None

    def progress(i, name, m=None, err=None):
        with lock:
            counter["done"] += 1
            if err is not None:
                counter["failed"] += 1
            done, nfail, n_in = counter["done"], counter["failed"], len(inflight)
        failed_tag = f", {nfail} failed" if nfail else ""
        in_tag = f", {n_in} in flight" if args.rate else ""
        if deadline is None:
            tag = f"[{done}/{args.requests}{failed_tag}{in_tag}]"
        else:
            tag = f"[{done} done{failed_tag}{in_tag}, {fmt(time.perf_counter() - t0, 0)}/{fmt(args.duration, 0)}s]"
        if err is None:
            print(f"  {tag} #{i + 1} {name}: ttft {fmt(m['ttft_s'])}s  total {fmt(m['total_time_s'])}s  "
                  f"{fmt(m['completion_tokens'])} tok  {fmt(m['generation_tokens_per_s'], 1)} tok/s"
                  + ("  [hit max_tokens]" if m.get("finish_reason") == "length" else ""))
        else:
            print(f"  {tag} #{i + 1} {name}: FAILED {err!r}")

    def run_request(i, name, prompt, sched_s=None):
        start = time.perf_counter() - t0
        with lock:
            inflight[i] = (name, start)
            counter["peak_in_flight"] = max(counter["peak_in_flight"], len(inflight))
            if sched_s is not None:
                counter["send_lag_max"] = max(counter["send_lag_max"], start - sched_s)
        try:
            m, resp, reas = run_once(args.url, args.api_key, model, prompt, args.system,
                                     args.max_tokens, args.temperature, None, args.thinking,
                                     args.reasoning_effort)
            m["start_s"] = start
            m["end_s"] = time.perf_counter() - t0
            m["sched_s"] = sched_s
            ct = m["completion_tokens"]
            m["itl_s"] = safe_div(m["generation_time_s"], ct - 1) if ct and ct > 1 else None
            rec = {"idx": i, "name": name, "ok": True, "m": m, "response": resp, "reasoning": reas}
            err = None
        except Exception as e:                                    # noqa: BLE001
            rec = {"idx": i, "name": name, "ok": False, "error": repr(e),
                   "start_s": start, "end_s": time.perf_counter() - t0}
            err = e
        with lock:
            inflight.pop(i, None)
            if state["closed"]:            # batch already summarised (drain timeout): drop silently
                return
            results.append(rec)
        progress(i, name, m=rec.get("m"), err=err)

    if args.rate:
        # ---- open loop: fixed arrival rate, one thread per request
        interval = 1.0 / args.rate
        total = None if args.duration else args.requests
        threads = []
        i = 0
        while True:
            if total is not None and i >= total:
                break
            target = t0 + i * interval
            wait = target - time.perf_counter()
            if wait > 0:
                if deadline is not None and target >= deadline:
                    break
                time.sleep(wait)
            if deadline is not None and time.perf_counter() >= deadline:
                break
            name, prompt, _ = jobs[i % len(jobs)]
            with lock:
                n_in = len(inflight)
            if n_in >= args.max_in_flight:
                with lock:
                    counter["skipped"] += 1
                if counter["skipped"] in (1, 10, 100, 1000):
                    print(f"  [skipped #{i + 1}: {n_in} requests in flight >= --max-in-flight {args.max_in_flight}]")
            else:
                t = threading.Thread(target=run_request, args=(i, name, prompt, i * interval), daemon=True)
                t.start()
                threads.append(t)
            i += 1
        scheduled = i
    else:
        # ---- closed loop: fixed number of workers
        def next_request():
            with lock:
                i = counter["next"]
                if deadline is None:
                    if i >= args.requests:
                        return None
                elif time.perf_counter() >= deadline:
                    return None
                counter["next"] = i + 1
            name, prompt, _ = jobs[i % len(jobs)]
            return i, name, prompt

        def worker():
            while True:
                nr = next_request()
                if nr is None:
                    return
                run_request(*nr)

        threads = [threading.Thread(target=worker, daemon=True) for _ in range(args.concurrency)]
        for t in threads:
            t.start()
        scheduled = None

    # ---- drain: wait for in-flight requests, up to --drain seconds after the deadline
    drain_end = None
    if args.drain is not None:
        drain_end = (deadline if deadline is not None else time.perf_counter()) + args.drain
    for t in threads:
        if drain_end is None:
            t.join()
        else:
            left = drain_end - time.perf_counter()
            if left > 0:
                t.join(left)
    with lock:
        state["closed"] = True
        wall = time.perf_counter() - t0
        for i, (name, start) in sorted(inflight.items()):
            results.append({"idx": i, "name": name, "ok": False, "unfinished": True,
                            "error": f"still running after --drain {fmt(args.drain, 1)} s",
                            "start_s": start, "end_s": None})
        drained = not inflight
    ended_at = time.strftime("%Y-%m-%d %H:%M:%S")
    if gpu:
        gpu.stop()
    if not drained:
        print(f"  [drain timeout: {sum(1 for r in results if r.get('unfinished'))} requests still running; "
              f"they are reported as unfinished]")

    results.sort(key=lambda r: r["idx"])
    extra = {"scheduled": scheduled if scheduled is not None else len(results),
             "skipped": counter["skipped"], "peak_in_flight": counter["peak_in_flight"],
             "send_lag_max": counter["send_lag_max"] if args.rate else None}
    return summarize(args, model, jobs, results, wall, started_at, ended_at,
                     gpu.summary() if gpu else None, extra), results


def summarize(args, model, jobs, results, wall, started_at, ended_at, gpu_summary, extra):
    ok = [r["m"] for r in results if r["ok"]]
    unfinished = [r for r in results if r.get("unfinished")]
    failed = [r for r in results if not r["ok"] and not r.get("unfinished")]
    prompt_tok = sum(m["prompt_tokens"] or 0 for m in ok)
    comp_tok = sum(m["completion_tokens"] or 0 for m in ok)
    busy = sum((r["m"]["total_time_s"] if r["ok"] else (r_end(r) if r["end_s"] is not None else wall) - r["start_s"])
               for r in results)

    s = {
        "model": model,
        "started_at": started_at,
        "ended_at": ended_at,
        "concurrency": args.concurrency,
        "rate": args.rate,
        "duration_s": args.duration,
        "requests": len(results),                                # sent (incl. unfinished)
        "scheduled": extra["scheduled"],
        "skipped": extra["skipped"],
        "sent_per_s": safe_div(len(results), args.duration or wall),
        "send_lag_max": extra["send_lag_max"],
        "peak_in_flight": extra["peak_in_flight"],
        "unfinished": len(unfinished),
        "ok": len(ok),
        "failed": len(failed),
        "error_rate": safe_div(len(failed), len(results)),
        "wall_s": wall,
        "requests_per_s": safe_div(len(ok), wall),
        "avg_in_flight": safe_div(busy, wall),                    # achieved concurrency
        "prompt_tokens_total": prompt_tok,
        "completion_tokens_total": comp_tok,
        "total_tokens": prompt_tok + comp_tok,
        "prompt_tokens_avg": safe_div(prompt_tok, len(ok)),
        "completion_tokens_avg": safe_div(comp_tok, len(ok)),
        "completion_tok_per_s": safe_div(comp_tok, wall),         # aggregate generation throughput
        "total_tok_per_s": safe_div(prompt_tok + comp_tok, wall),
        "truncated": sum(1 for m in ok if m.get("finish_reason") == "length"),
        "empty": sum(1 for r in results if r["ok"] and not r["response"].strip()),
        "ttft": stats([m["ttft_s"] for m in ok]),
        "latency": stats([m["total_time_s"] for m in ok]),
        "itl": stats([m["itl_s"] for m in ok]),                   # inter-token latency
        "gen_tps": stats([m["generation_tokens_per_s"] for m in ok]),   # per-request decode speed
        "comp_tok": stats([m["completion_tokens"] for m in ok]),
        "errors": dict(Counter(r["error"][:160] for r in failed)) if failed else None,
        "gpu": gpu_summary,
    }

    # duration mode: numbers over the target window only, i.e. without the tail after the deadline
    if args.duration:
        d = args.duration
        in_win = [m for m in ok if m["end_s"] <= d]
        w_comp = sum(m["completion_tokens"] or 0 for m in in_win)
        w_prompt = sum(m["prompt_tokens"] or 0 for m in in_win)
        s["window"] = {
            "tail_s": max(wall - d, 0.0),
            "in_flight_at_deadline": sum(1 for r in results
                                         if r_start(r) < d < r_end(r)),
            "completed": len(in_win),
            "requests_per_s": safe_div(len(in_win), d),
            "completion_tok_per_s": safe_div(w_comp, d),
            "total_tok_per_s": safe_div(w_comp + w_prompt, d),
            "completion_tokens": w_comp,
        }
    else:
        s["window"] = None

    # per prompt breakdown (only meaningful when several prompts are mixed)
    names = [n for n, _, _ in jobs]
    if len(set(names)) > 1:
        pp = OrderedDict()
        for name in names:
            rs = [r for r in results if r["name"] == name]
            ms = [r["m"] for r in rs if r["ok"]]
            pp[name] = {
                "n": len(rs), "ok": len(ms), "failed": len(rs) - len(ms),
                "ttft": stats([m["ttft_s"] for m in ms]),
                "latency": stats([m["total_time_s"] for m in ms]),
                "itl": stats([m["itl_s"] for m in ms]),
                "prompt_tok": safe_div(sum(m["prompt_tokens"] or 0 for m in ms), len(ms)),
                "comp_tok": safe_div(sum(m["completion_tokens"] or 0 for m in ms), len(ms)),
                "gen_tps": stats([m["generation_tokens_per_s"] for m in ms]),
                "truncated": sum(1 for m in ms if m.get("finish_reason") == "length"),
            }
        s["per_prompt"] = pp
    else:
        s["per_prompt"] = None

    # timeline: what happened in each time bucket (shows drift / warm-up / saturation over the run)
    span = args.duration or wall
    bucket = args.bucket or auto_bucket(span)
    s["bucket_s"] = bucket
    if args.duration or wall >= 2 * bucket:
        rows = []
        n_b = max(1, math.ceil(wall / bucket))
        for k in range(n_b):
            lo, hi = k * bucket, min((k + 1) * bucket, wall)
            done = [m for m in ok if lo <= m["end_s"] < hi or (k == n_b - 1 and m["end_s"] >= lo)]
            fail = sum(1 for r in failed if lo <= r_end(r) < hi)
            started = sum(1 for r in results if lo <= r_start(r) < hi)
            width = hi - lo
            ctok = sum(m["completion_tokens"] or 0 for m in done)
            rows.append({
                "t0": lo, "t1": hi,
                "started": started, "completed": len(done), "failed": fail,
                "in_flight_end": sum(1 for r in results if r_start(r) <= hi < r_end(r)),
                "completion_tokens": ctok,
                "completion_tok_per_s": safe_div(ctok, width),
                "ttft_mean": statistics.fmean([m["ttft_s"] for m in done if m["ttft_s"] is not None])
                if any(m["ttft_s"] is not None for m in done) else None,
                "latency_mean": statistics.fmean([m["total_time_s"] for m in done]) if done else None,
                "ttft_p95": pct([m["ttft_s"] for m in done if m["ttft_s"] is not None], 95),
            })
        s["timeline"] = rows
    else:
        s["timeline"] = None
    return s


def r_start(r):
    return r["m"]["start_s"] if r["ok"] else r["start_s"]


def r_end(r):
    """End time relative to the batch; +inf for requests still running at the drain timeout."""
    if r["ok"]:
        return r["m"]["end_s"]
    return r["end_s"] if r["end_s"] is not None else float("inf")


def average_summaries(sums):
    """Average the scalar fields of several batch summaries (for --runs > 1)."""
    avg = dict(sums[0])
    scalar = ["wall_s", "requests_per_s", "avg_in_flight", "prompt_tokens_total", "completion_tokens_total",
              "total_tokens", "prompt_tokens_avg", "completion_tokens_avg", "completion_tok_per_s",
              "total_tok_per_s", "requests", "ok", "failed", "error_rate", "truncated", "empty",
              "scheduled", "skipped", "sent_per_s", "send_lag_max", "peak_in_flight", "unfinished"]
    for k in scalar:
        vals = [s[k] for s in sums if s[k] is not None]
        avg[k] = sum(vals) / len(vals) if vals else None
    for k in ("ttft", "latency", "itl", "gen_tps", "comp_tok"):
        avg[k] = {sk: (statistics.fmean([s[k][sk] for s in sums if s[k][sk] is not None])
                       if any(s[k][sk] is not None for s in sums) else None)
                  for sk in STAT_KEYS}
    if sums[0]["window"]:
        avg["window"] = {}
        for k in sums[0]["window"]:
            vals = [s["window"][k] for s in sums if s["window"][k] is not None]
            avg["window"][k] = sum(vals) / len(vals) if vals else None
    avg["ended_at"] = sums[-1]["ended_at"]
    errs = Counter()
    for s in sums:
        errs.update(s["errors"] or {})
    avg["errors"] = dict(errs) or None
    # listed individually per batch in the log:
    avg["gpu"] = None
    avg["per_prompt"] = None
    avg["timeline"] = None
    return avg


# ----------------------------------------------------------------------------- output

def summary_lines(s, gpu_interval):
    L = []
    if s.get("rate"):
        L.append("[load]  open loop: a new request every 1/rate s regardless of how many are in flight")
        L.append(f"  target rate          : {fmt(s['rate'], 2)} req/s   (sent {fmt(s['requests'], 0)} = {fmt(s['sent_per_s'], 2)} req/s, "
                 f"max send lag {fmt(s['send_lag_max'])} s)")
        if s.get("skipped"):
            L.append(f"  skipped              : {fmt(s['skipped'], 0)}  (not sent: --max-in-flight reached)")
        L.append(f"  in flight            : peak {fmt(s['peak_in_flight'], 0)}, avg {fmt(s['avg_in_flight'], 1)}"
                 + (f", at deadline {fmt(s['window']['in_flight_at_deadline'], 0)}" if s.get("window") else "")
                 + "   (rising = server can't keep up with the rate)")
    else:
        L.append("[load]  closed loop: a fixed number of requests in flight")
        L.append(f"  concurrency          : {s['concurrency']}   (achieved avg in flight {fmt(s['avg_in_flight'], 2)})")
    if s.get("duration_s"):
        L.append(f"  duration             : {fmt(s['duration_s'], 1)} s   (no new requests after this; in-flight ones finish)")
    L.append(f"  requests             : {fmt(s['requests'], 0)}  (ok {fmt(s['ok'], 0)})")
    L.append(f"  failed               : {fmt(s['failed'], 0)}  ({fmt((s['error_rate'] or 0) * 100, 1)} % of requests)")
    if s.get("unfinished"):
        L.append(f"  unfinished           : {fmt(s['unfinished'], 0)}  (still running at the --drain timeout; excluded from latency stats)")
    L.append(f"  wall time            : {fmt(s['wall_s'])} s")
    L.append(f"  completed/s          : {fmt(s['requests_per_s'], 2)}")
    w = s.get("window")
    if w:
        L.append("")
        L.append(f"[window]  first {fmt(s['duration_s'], 1)} s only — steady-state numbers without the tail after the deadline")
        L.append(f"  completed in window  : {fmt(w['completed'], 0)}   ({fmt(w['requests_per_s'], 2)} req/s)")
        L.append(f"  in flight at deadline: {fmt(w['in_flight_at_deadline'], 0)}   (tail after deadline {fmt(w['tail_s'], 1)} s)")
        L.append(f"  completion tok/s     : {fmt(w['completion_tok_per_s'], 1)}   ({fmt(w['completion_tokens'], 0)} tokens)")
        L.append(f"  total tok/s          : {fmt(w['total_tok_per_s'], 1)}   (prompt + completion)")
    L.append("")
    L.append("[throughput]  aggregate over the whole batch incl. tail (this is the number that scales with load)")
    L.append(f"  completion tok/s     : {fmt(s['completion_tok_per_s'], 1)}")
    L.append(f"  total tok/s          : {fmt(s['total_tok_per_s'], 1)}   (prompt + completion)")
    L.append(f"  prompt tokens        : {fmt(s['prompt_tokens_total'], 0)} total   ({fmt(s['prompt_tokens_avg'], 0)} avg / request)")
    L.append(f"  completion tokens    : {fmt(s['completion_tokens_total'], 0)} total   ({fmt(s['completion_tokens_avg'], 0)} avg / request, "
             f"min {fmt(s['comp_tok']['min'], 0)} / p50 {fmt(s['comp_tok']['p50'], 0)} / max {fmt(s['comp_tok']['max'], 0)})")
    L.append(f"  hit max_tokens       : {fmt(s['truncated'], 0)} requests   (finish_reason = length)")
    L.append(f"  empty responses      : {fmt(s['empty'], 0)} requests   (no answer text; e.g. all tokens spent on reasoning)")
    L.append("")
    L.append("[latency]  per request         mean / p50 / p95 / p99 / max")
    L.append(stat_line("time to first token", s["ttft"], 3, " s"))
    L.append(stat_line("total time", s["latency"], 3, " s"))
    L.append(stat_line("inter-token latency", s["itl"], 1, " ms   (time per output token after the first)", scale=1000))
    L.append(stat_line("generation tok/s", s["gen_tps"], 1, "   (per request, drops as concurrency rises)"))
    if s.get("errors"):
        L.append("")
        L.append("[errors]")
        for err, n in sorted(s["errors"].items(), key=lambda kv: -kv[1]):
            L.append(f"  {n:>4} x {err}")
    if s.get("per_prompt"):
        L.append("")
        L.append("[per prompt]   n    ok  fail   prompt_tok   comp_tok   ttft_mean   ttft_p95   total_mean   total_p95   itl_ms   gen_tok_s   hit_max   prompt")
        for name, p in s["per_prompt"].items():
            L.append(f"  {p['n']:>12} {p['ok']:>5} {p['failed']:>5}   {fmt(p['prompt_tok'], 0):>10}   {fmt(p['comp_tok'], 0):>8}   "
                     f"{fmt(p['ttft']['mean']):>9}   {fmt(p['ttft']['p95']):>8}   {fmt(p['latency']['mean']):>10}   "
                     f"{fmt(p['latency']['p95']):>9}   {fmt((p['itl']['mean'] or 0) * 1000 if p['itl']['mean'] is not None else None, 1):>6}   "
                     f"{fmt(p['gen_tps']['mean'], 1):>9}   {p['truncated']:>7}   {name}")
    if s.get("timeline"):
        L.append("")
        L.append(f"[timeline]  per {fmt(s['bucket_s'], 0)} s bucket — watch for tok/s dropping or TTFT climbing over the run")
        L.append("       t0       t1   started   completed   failed   in_flight   comp_tok   comp_tok/s   ttft_mean   ttft_p95   total_mean")
        for b in s["timeline"]:
            L.append(f"  {fmt(b['t0'], 0):>7}  {fmt(b['t1'], 0):>7}   {b['started']:>7}   {b['completed']:>9}   {b['failed']:>6}   "
                     f"{b['in_flight_end']:>9}   {fmt(b['completion_tokens'], 0):>8}   {fmt(b['completion_tok_per_s'], 1):>10}   "
                     f"{fmt(b['ttft_mean']):>9}   {fmt(b['ttft_p95']):>8}   {fmt(b['latency_mean']):>10}")
    if s.get("gpu") is not None:
        L.append("")
        L.append(f"[gpu]  sampled every {gpu_interval}s across the whole batch with nvidia-smi")
        L.extend(GpuSampler.lines(s["gpu"]))
    return L


def failed_lines(results):
    """One row per failed request: when it started / failed, which prompt, and the error."""
    failed = [r for r in results if not r["ok"]]
    if not failed:
        return []
    L = [f"[failed requests]  {len(failed)} of {len(results)}     #   start_s   failed_at_s   prompt   error"]
    for r in failed:
        L.append(f"  {r['idx'] + 1:>3}   {fmt(r['start_s']):>7}   {fmt(r['end_s']):>11}   {r['name']}   {r['error']}")
    return L


def per_request_lines(results, duration=None):
    L = ["[per request]   #   start_s    end_s   ttft_s   total_s   itl_ms   prompt_tok   comp_tok   gen_tok_s   finish   prompt"
         + ("   (* = finished after the deadline)" if duration else "")]
    for r in results:
        if r["ok"]:
            m = r["m"]
            star = "*" if duration and m["end_s"] > duration else " "
            L.append(f"  {r['idx'] + 1:>3}   {fmt(m['start_s']):>7}  {fmt(m['end_s']):>7}{star}  {fmt(m['ttft_s']):>6}   {fmt(m['total_time_s']):>7}   "
                     f"{fmt(m['itl_s'] * 1000 if m['itl_s'] is not None else None, 1):>6}   "
                     f"{fmt(m['prompt_tokens']):>10}   {fmt(m['completion_tokens']):>8}   "
                     f"{fmt(m['generation_tokens_per_s'], 1):>9}   {(m.get('finish_reason') or '-'):>6}   {r['name']}")
        elif r.get("unfinished"):
            L.append(f"  {r['idx'] + 1:>3}   {fmt(r['start_s']):>7}        -   UNFINISHED  {r['error']}   {r['name']}")
        else:
            L.append(f"  {r['idx'] + 1:>3}   {fmt(r['start_s']):>7}  {fmt(r['end_s']):>7}   FAILED  {r['error']}   {r['name']}")
    return L


def write_log(path, label, jobs, args, avg, batches, gpu_interval):
    """batches: [(summary, results), ...] one per --runs."""
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    L = []
    L.append("=" * 72)
    head = f"{ts}  |  model: {avg['model']}  |  prompt: {label}  |  "
    head += f"rate: {fmt(args.rate, 2)} req/s  |  " if args.rate else f"concurrency: {avg['concurrency']}  |  "
    if args.duration:
        head += f"duration: {fmt(args.duration, 1)}s  |  requests: {fmt(avg['requests'], 0)}  |  runs: {len(batches)}"
    else:
        head += f"requests: {fmt(avg['requests'], 0)}  |  runs: {len(batches)}"
    if args.thinking:
        head += f"  |  thinking: {args.thinking}"
    L.append(head)
    L.append(f"max_tokens: {args.max_tokens}  |  temperature: {args.temperature}"
             + (f"  |  reasoning_effort: {args.reasoning_effort}" if args.reasoning_effort else "")
             + (f"  |  drain: {fmt(args.drain, 1)}s" if args.drain is not None else "")
             + f"  |  url: {args.url}  |  started: {batches[0][0]['started_at']}  |  ended: {batches[-1][0]['ended_at']}")
    L.append("=" * 72)
    L.append("")
    if len(batches) > 1:
        L.append(f"[average of {len(batches)} batches]")
        L.append("")
    L.extend(summary_lines(avg, gpu_interval))
    for i, (s, results) in enumerate(batches, 1):
        L.append("")
        if len(batches) > 1:
            L.append(f"--- batch {i}  ({s['started_at']} -> {s['ended_at']}) ---")
            L.extend(summary_lines(s, gpu_interval))
            L.append("")
        L.extend(per_request_lines(results, args.duration))
        fl = failed_lines(results)
        if fl:
            L.append("")
            L.extend(fl)
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
        if r["reasoning"] and not args.no_reasoning_log:
            L.append(f"[reasoning]  {tag}")
            L.append(r["reasoning"].rstrip())
            L.append("")
        note = empty_response_note(r["m"], r["response"], r["reasoning"])
        L.append(f"[response]  {tag}" + (f"  {note}" if note else ""))
        L.append(r["response"].rstrip())
        L.append("")
    L.append("")
    with open(path, "a", encoding="utf-8") as f:
        f.write("\n".join(L))


# ----------------------------------------------------------------------------- main

def main():
    p = build_parser("Concurrent vLLM load benchmark (same flags as bench_vllm.py)")
    p.add_argument("--concurrency", type=int, default=None,
                   help="closed loop: requests in flight at once (default 4; ignored with --rate)")
    p.add_argument("--rate", type=float, default=None, metavar="REQ_PER_S",
                   help="open loop: send this many new requests per second for --duration seconds (or "
                        "--requests in total), regardless of how many are in flight")
    p.add_argument("--max-in-flight", type=int, default=2000,
                   help="--rate safety cap: skip new requests while this many are in flight (default 2000)")
    p.add_argument("--drain", type=float, default=None, metavar="SECONDS",
                   help="after the deadline wait at most this long for in-flight requests; the rest are "
                        "reported as unfinished (default: wait for all)")
    p.add_argument("--requests", type=int, default=None,
                   help="total requests per prompt (default: same as --concurrency; ignored with --duration)")
    p.add_argument("--duration", type=float, default=None, metavar="SECONDS",
                   help="keep --concurrency requests in flight for this many seconds instead of a fixed "
                        "--requests count (no new requests after the deadline; in-flight ones finish)")
    p.add_argument("--bucket", type=float, default=None, metavar="SECONDS",
                   help="timeline bucket size in seconds (default: auto, about 10-12 buckets)")
    p.add_argument("--mix", action="store_true",
                   help="one batch cycling round-robin through every prompt in prompts/ "
                        "(--prompt / --prompt-file are ignored)")
    p.add_argument("--all-responses", action="store_true", help="log every response (default: only the first)")
    args = p.parse_args()
    base.LOG_WORDS = args.log_words
    if args.rate is not None:
        if args.rate <= 0:
            sys.exit("--rate must be > 0")
        if args.duration is None and args.requests is None:
            sys.exit("--rate needs --duration SECONDS (or --requests N for a fixed total)")
        if args.concurrency is not None:
            print("note: --concurrency is ignored with --rate (open loop has no in-flight limit; see --max-in-flight)")
        args.concurrency = None
        if args.max_in_flight < 1:
            sys.exit("--max-in-flight must be >= 1")
    elif args.concurrency is None:
        args.concurrency = 4
    if args.concurrency is not None and args.concurrency < 1:
        sys.exit("--concurrency must be >= 1")
    if args.bucket is not None and args.bucket <= 0:
        sys.exit("--bucket must be > 0 seconds")
    if args.drain is not None and args.drain < 0:
        sys.exit("--drain must be >= 0 seconds")
    if args.rate is not None and args.duration is None:      # fixed total at a fixed rate
        if args.requests < 1:
            sys.exit("--requests must be >= 1")
    elif args.duration is not None:
        if args.duration <= 0:
            sys.exit("--duration must be > 0 seconds")
        if args.requests is not None:
            print("note: --requests is ignored when --duration is given")
        args.requests = None
    else:
        if args.requests is None:
            args.requests = args.concurrency
        if args.requests < 1:
            sys.exit("--requests must be >= 1")

    if args.list_models:
        for m in list_models(args.url, args.api_key):
            print(m)
        return

    prompt_dir, log_dir = resolve_dirs(args)
    if args.mix:
        if args.prompt or args.prompt_file:
            print("note: --mix uses every prompt in the prompts folder; --prompt / --prompt-file are ignored")
        jobs = load_prompt_dir(prompt_dir)
    else:
        jobs = load_jobs(args, prompt_dir)
    model = choose_model(args)
    gpu = make_gpu_sampler(args)

    load_desc = (f"duration per batch: {fmt(args.duration, 1)}s" if args.duration
                 else f"requests per batch: {args.requests}")
    mode_desc = (f"open loop: {fmt(args.rate, 2)} req/s (max in flight {args.max_in_flight})" if args.rate
                 else f"concurrency: {args.concurrency}")
    print(f"using model: {model}")
    print(f"{mode_desc}   {load_desc}   runs: {args.runs}"
          + (f"   drain: {fmt(args.drain, 1)}s" if args.drain is not None else "")
          + (f"   thinking: {args.thinking}" if args.thinking else "")
          + (f"   gpu sampling: {args.gpu_interval}s" if gpu else "   gpu sampling: off"))

    # group jobs -> one batch per prompt, or a single mixed batch cycling through all of them
    if args.mix:
        print(f"mix: {len(jobs)} prompts round-robin from {os.path.relpath(prompt_dir)}/: "
              + ", ".join(name for name, _, _ in jobs))
        groups = [("mixed", jobs)]
    else:
        groups = [(name, [(name, prompt, ppath)]) for name, prompt, ppath in jobs]

    for label, group_jobs in groups:
        if args.rate:
            what = f"{fmt(args.rate, 2)} req/s " + (f"for {fmt(args.duration, 1)}s" if args.duration
                                                     else f"x {args.requests} requests")
        else:
            what = (f"{fmt(args.duration, 1)}s at concurrency {args.concurrency}" if args.duration
                    else f"{args.requests} requests")
        print(f"\n################ {label}  ({what} x {args.runs} run(s)) ################")
        batches = []
        for i in range(args.runs):
            if args.runs > 1:
                print(f"\n-- batch {i + 1}/{args.runs} --")
            s, results = run_batch(args, model, group_jobs, gpu)
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
                note = empty_response_note(first["m"], first["response"], first["reasoning"])
                print(f"\n--- response (request {first['idx'] + 1}) ---" + (f"  {note}" if note else ""))
                print(first["response"])

        log_path = args.log or os.path.join(log_dir, f"{label}_concurrent.txt")
        write_log(log_path, label, group_jobs, args, avg, batches, args.gpu_interval)
        tot_req = sum(b[0]["requests"] for b in batches)
        tot_fail = sum(b[0]["failed"] for b in batches)
        tot_unf = sum(b[0]["unfinished"] for b in batches)
        if not tot_fail and not tot_unf:
            verdict = f"all {tot_req} requests ok"
        else:
            verdict = f"{tot_req} requests: {tot_req - tot_fail - tot_unf} ok, {tot_fail} FAILED ({fmt(tot_fail / tot_req * 100, 1)} %)"
            if tot_unf:
                verdict += f", {tot_unf} unfinished at drain timeout"
        print(f"\n{label}: {verdict}   ->   logged to {os.path.relpath(log_path)}")


if __name__ == "__main__":
    main()
