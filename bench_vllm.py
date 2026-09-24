#!/usr/bin/env python3
"""
Simple vLLM request benchmark.

Prompts live in prompts/ (one .txt each); logs go to logs/<prompt-name>.txt (appended).
Per request it reports TTFT, total time, token counts (from vLLM's usage), tok/s and GPU
utilisation (nvidia-smi). Response + metrics are printed in the terminal AND written to the log.

Examples:
  python bench_vllm.py                                  # run every prompt in prompts/
  python bench_vllm.py --prompt-file summary            # prompts/summary.txt -> logs/summary.txt
  python bench_vllm.py --prompt "What is a LLM?"        # inline prompt      -> logs/inline.txt
  python bench_vllm.py --runs 3 --model Qwen/Qwen3-14B
  python bench_vllm.py --list-models | --pick-model | --no-gpu | --show-reasoning | --quiet
  python bench_vllm.py --thinking off                   # Qwen3 etc: turn thinking mode on/off
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time

import requests


class GpuSampler:
    """Samples nvidia-smi in a background thread while a request runs."""

    def __init__(self, interval=0.25):
        self.interval = interval
        self.samples = []          # (index, util%, mem_used_MiB, mem_total_MiB)
        self.available = shutil.which("nvidia-smi") is not None
        self._stop = threading.Event()
        self._thread = None

    def _query(self):
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5).stdout
        except Exception:
            return
        for line in out.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 4:
                self.samples.append((int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])))

    def _loop(self):
        while not self._stop.is_set():
            self._query()
            self._stop.wait(self.interval)

    def start(self):
        if not self.available:
            return
        self.samples = []
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        if self._thread:
            self._stop.set()
            self._thread.join()
            self._thread = None

    def summary(self):
        """dict per gpu: util_avg, util_max, mem_peak, mem_total, samples"""
        out = {}
        for idx, util, mem, tot in self.samples:
            g = out.setdefault(idx, {"util_sum": 0.0, "util_max": 0.0, "mem_peak": 0.0, "mem_total": tot, "samples": 0})
            g["util_sum"] += util
            g["util_max"] = max(g["util_max"], util)
            g["mem_peak"] = max(g["mem_peak"], mem)
            g["samples"] += 1
        for g in out.values():
            g["util_avg"] = g["util_sum"] / g["samples"] if g["samples"] else None
            del g["util_sum"]
        return dict(sorted(out.items()))

    @staticmethod
    def lines(summary):
        if not summary:
            return ["  (no samples)"]
        return [f"  gpu {i}: util avg {g['util_avg']:.0f}%  max {g['util_max']:.0f}%  |  "
                f"mem peak {g['mem_peak']:.0f}/{g['mem_total']:.0f} MiB  |  {g['samples']} samples"
                for i, g in summary.items()]


def headers(api_key):
    h = {"Content-Type": "application/json"}
    if api_key:
        h["Authorization"] = f"Bearer {api_key}"
    return h


def list_models(base_url: str, api_key=None):
    r = requests.get(f"{base_url}/models", headers=headers(api_key), timeout=10)
    r.raise_for_status()
    models = [m["id"] for m in r.json().get("data", [])]
    if not models:
        sys.exit("No models reported by server at /v1/models")
    return models


def pick_model(base_url: str, api_key=None) -> str:
    models = list_models(base_url, api_key)
    if len(models) == 1:
        return models[0]
    print("Available models:")
    for i, m in enumerate(models, 1):
        print(f"  [{i}] {m}")
    while True:
        choice = input(f"Pick a model [1-{len(models)}]: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(models):
            return models[int(choice) - 1]
        print("invalid choice")


REASONING_FIELDS = ("reasoning_content", "reasoning")   # vLLM (Qwen3, DeepSeek) / newer vLLM + gpt-oss


def run_once(base_url, api_key, model, prompt, system, max_tokens, temperature, gpu=None, thinking=None,
             reasoning_effort=None):
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True},  # vLLM sends usage in final chunk
    }
    if thinking is not None:   # only meaningful for models with a thinking mode (Qwen3, ...)
        payload["chat_template_kwargs"] = {"enable_thinking": thinking == "on"}
    if reasoning_effort:       # gpt-oss and other models that take an effort level
        payload["reasoning_effort"] = reasoning_effort

    text_parts = []
    reasoning_parts = []
    usage = {}
    ttft = None
    finish_reason = None
    delta_keys = set()      # which fields the stream actually carried (to explain an empty response)
    n_deltas = 0

    if gpu:
        gpu.start()
    t0 = time.perf_counter()
    with requests.post(f"{base_url}/chat/completions", headers=headers(api_key),
                       json=payload, stream=True, timeout=600) as r:
        if r.status_code >= 400:
            # vLLM puts the real reason in the body, e.g. context length exceeded / bad chat_template_kwargs
            try:
                detail = r.json().get("message") or r.json().get("error", {}).get("message") or r.text
            except Exception:
                detail = r.text
            raise requests.HTTPError(f"{r.status_code} from {r.url}: {detail[:1000]}", response=r)
        for line in r.iter_lines():
            if not line:
                continue
            line = line.decode("utf-8")
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            if chunk.get("usage"):
                usage = chunk["usage"]
            for choice in chunk.get("choices", []):
                if choice.get("finish_reason"):
                    finish_reason = choice["finish_reason"]     # "stop" or "length" (= hit max_tokens)
                delta = choice.get("delta", {})
                n_deltas += 1
                delta_keys.update(k for k, v in delta.items() if v not in (None, "", [], {}))
                reasoning = next((delta.get(k) for k in REASONING_FIELDS if delta.get(k)), None)
                content = delta.get("content")
                if reasoning:
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    reasoning_parts.append(reasoning)
                if content:
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    text_parts.append(content)
    total = time.perf_counter() - t0
    if gpu:
        gpu.stop()

    response_text = "".join(text_parts)
    reasoning_text = "".join(reasoning_parts)
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    total_tokens = usage.get("total_tokens")

    gen_time = total - (ttft or 0)
    gen_tps = completion_tokens / gen_time if completion_tokens and gen_time > 0 else None
    overall_tps = completion_tokens / total if completion_tokens and total > 0 else None

    return {
        "model": model,
        "ttft_s": ttft,
        "total_time_s": total,
        "generation_time_s": gen_time,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "generation_tokens_per_s": gen_tps,
        "overall_tokens_per_s": overall_tps,
        "response_chars": len(response_text),
        "reasoning_chars": len(reasoning_text),
        "finish_reason": finish_reason,
        "delta_keys": sorted(delta_keys),
        "n_deltas": n_deltas,
        "gpu": gpu.summary() if gpu else None,
    }, response_text, reasoning_text


def finish_note(m):
    """'length (hit max_tokens)' etc. for the finish_reason line."""
    fr = m.get("finish_reason")
    if fr is None:
        return "-"
    return fr + (" (hit max_tokens: output was cut off)" if fr == "length" else "")


def empty_response_note(m, response_text, reasoning_text):
    """Why the response is empty, or None if it is not."""
    if response_text.strip():
        return None
    ct = m.get("completion_tokens")
    keys = m.get("delta_keys") or []
    if reasoning_text.strip():
        return (f"(empty: all {fmt(ct, 0)} completion tokens went into reasoning, see [reasoning]; "
                f"raise --max-tokens or lower --reasoning-effort / turn thinking off)")
    if ct:
        return (f"(empty: {fmt(ct, 0)} completion tokens were generated but no content/reasoning was received; "
                f"stream delta fields seen: {keys or 'none'} in {m.get('n_deltas', 0)} deltas)")
    return "(empty: the model produced no tokens)"


def fmt(v, nd=3):
    return "-" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def print_metrics(m, label=""):
    print(f"--- {label or 'metrics'} ---")
    print(f"model                : {m['model']}")
    print(f"time to first token  : {fmt(m['ttft_s'])} s")
    print(f"total time           : {fmt(m['total_time_s'])} s")
    print(f"prompt tokens        : {fmt(m['prompt_tokens'])}")
    print(f"completion tokens    : {fmt(m['completion_tokens'])}")
    print(f"total tokens         : {fmt(m['total_tokens'])}")
    print(f"generation tok/s     : {fmt(m['generation_tokens_per_s'], 1)}")
    print(f"overall tok/s        : {fmt(m['overall_tokens_per_s'], 1)}")
    print(f"finish reason        : {finish_note(m)}")
    if m.get("gpu") is not None:
        print("gpu (this run)       :")
        print("\n".join(GpuSampler.lines(m["gpu"])))


LOG_WORDS = 30   # max prompt words shown in the log


def short_text(text, n=None):
    n = n or LOG_WORDS
    words = text.split()
    return " ".join(words[:n]) + (" ..." if len(words) > n else "")


def write_log(path, name, prompt, system, summary, runs, response_text, reasoning_text="",
              prompt_path=None, thinking=None, args=None):
    """Append one clean block per benchmark to a single .txt log."""
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    lines = []
    lines.append("=" * 72)
    head = f"{ts}  |  model: {summary['model']}  |  prompt: {name}  |  runs: {len(runs)}"
    if thinking:
        head += f"  |  thinking: {thinking}"
    lines.append(head)
    if args is not None:
        sub = f"max_tokens: {args.max_tokens}  |  temperature: {args.temperature}  |  url: {args.url}"
        if args.reasoning_effort:
            sub += f"  |  reasoning_effort: {args.reasoning_effort}"
        lines.append(sub)
    lines.append("=" * 72)
    lines.append("")
    lines.append("[metrics]")
    lines.append(f"  time to first token  : {fmt(summary['ttft_s'])} s")
    lines.append(f"  total time           : {fmt(summary['total_time_s'])} s")
    lines.append(f"  prompt tokens        : {fmt(summary['prompt_tokens'])}")
    lines.append(f"  completion tokens    : {fmt(summary['completion_tokens'])}")
    lines.append(f"  total tokens         : {fmt(summary['total_tokens'])}")
    lines.append(f"  generation tok/s     : {fmt(summary['generation_tokens_per_s'], 1)}")
    lines.append(f"  overall tok/s        : {fmt(summary['overall_tokens_per_s'], 1)}")
    lines.append(f"  finish reason        : {finish_note(runs[-1])}")
    if len(runs) > 1:
        lines.append("")
        lines.append("[per run]  ttft_s / total_s / completion_tokens / gen_tok_s")
        for i, r in enumerate(runs, 1):
            lines.append(f"  run {i}: {fmt(r['ttft_s'])} / {fmt(r['total_time_s'])} / "
                         f"{fmt(r['completion_tokens'])} / {fmt(r['generation_tokens_per_s'], 1)}")
    if runs and runs[0].get("gpu") is not None:
        lines.append("")
        lines.append("[gpu]  sampled with nvidia-smi during each request")
        for i, r in enumerate(runs, 1):
            lines.append(f"run {i}:")
            lines.extend(GpuSampler.lines(r["gpu"]))
    if system:
        lines.append("")
        lines.append("[system]")
        lines.append(short_text(system))
    words = len(prompt.split())
    note = f", first {LOG_WORDS} shown" if words > LOG_WORDS else ""
    lines.append("")
    if prompt_path:
        lines.append(f"[prompt]  file: {prompt_path}  ({words} words{note})")
    else:
        lines.append(f"[prompt]  inline  ({words} words{note})")
    lines.append(short_text(prompt))
    lines.append("")
    if reasoning_text and not (args is not None and args.no_reasoning_log):
        lines.append("[reasoning]")
        lines.append(reasoning_text.rstrip())
        lines.append("")
    note = empty_response_note(runs[-1], response_text, reasoning_text)
    lines.append("[response]" + (f"  {note}" if note else ""))
    lines.append(response_text.rstrip())
    lines.append("")
    lines.append("")
    with open(path, "a", encoding="utf-8") as f:
        f.write("\n".join(lines))


def build_parser(description="Benchmark a single vLLM request"):
    """Shared CLI — bench_vllm_concurrent.py adds its own flags on top of this."""
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--url", default=os.environ.get("VLLM_URL", "http://localhost:8000/v1"),
                   help="vLLM base URL (env: VLLM_URL)")
    p.add_argument("--api-key", default=os.environ.get("VLLM_API_KEY"),
                   help="API key, sent as Bearer token (env: VLLM_API_KEY)")
    p.add_argument("--model", default=os.environ.get("VLLM_MODEL"),
                   help="model id (env: VLLM_MODEL; default: first model from /v1/models)")
    p.add_argument("--pick-model", action="store_true", help="show a numbered list of models and choose one")
    p.add_argument("--list-models", action="store_true", help="print available models and exit")
    p.add_argument("--prompt", default=None, help="inline prompt text (logged to logs/inline.txt)")
    p.add_argument("--prompt-file", action="append", default=[],
                   help="prompt file: a name (summary), a filename (summary.txt) or a path (prompts/summary.txt), "
                        "looked up in prompts/ (repeatable). Default: all of prompts/")
    p.add_argument("--prompt-dir", default="prompts", help="folder with .txt prompts")
    p.add_argument("--log-dir", default="logs", help="folder for logs")
    p.add_argument("--system", default=None, help="optional system prompt")
    p.add_argument("--log", default=None, help="single log file for everything (default: logs/<prompt-name>.txt)")
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--runs", type=int, default=1, help="repeat N times and average")
    p.add_argument("--thinking", choices=["on", "off"], default=None,
                   help="turn thinking mode on/off (only for models that have one, e.g. Qwen3)")
    p.add_argument("--reasoning-effort", choices=["low", "medium", "high"], default=None,
                   help="reasoning_effort for models that take one (gpt-oss): low keeps more of --max-tokens for the answer")
    p.add_argument("--log-words", type=int, default=30, help="max prompt words shown in the log")
    p.add_argument("--no-gpu", action="store_true", help="don't sample GPU utilisation with nvidia-smi")
    p.add_argument("--gpu-interval", type=float, default=0.25, help="seconds between nvidia-smi samples")
    p.add_argument("--show-reasoning", action="store_true", help="also print the model's reasoning/thinking in the terminal")
    p.add_argument("--no-reasoning-log", action="store_true",
                   help="don't write the model's reasoning/thinking to the log (metrics + response only)")
    p.add_argument("--quiet", action="store_true", help="don't print the response in the terminal (it is still written to the log)")
    return p


def resolve_dirs(args):
    """Create prompts/ and logs/ next to this script (or at the absolute paths given)."""
    here = os.path.dirname(os.path.abspath(__file__))
    prompt_dir = args.prompt_dir if os.path.isabs(args.prompt_dir) else os.path.join(here, args.prompt_dir)
    log_dir = args.log_dir if os.path.isabs(args.log_dir) else os.path.join(here, args.log_dir)
    os.makedirs(prompt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    if not os.listdir(prompt_dir):
        with open(os.path.join(prompt_dir, "example.txt"), "w", encoding="utf-8") as f:
            f.write("What is a LLM? Answer in two short paragraphs.\n")
        print(f"created {prompt_dir}/example.txt — add more .txt prompts there")
    return prompt_dir, log_dir


def load_prompt_dir(prompt_dir):
    """Every .txt in prompt_dir, sorted by name -> [(name, prompt_text, prompt_path), ...]."""
    jobs = []
    for fn in sorted(os.listdir(prompt_dir)):
        if fn.endswith(".txt"):
            path = os.path.join(prompt_dir, fn)
            with open(path, encoding="utf-8") as fh:
                jobs.append((os.path.splitext(fn)[0], fh.read(), os.path.relpath(path)))
    if not jobs:
        sys.exit(f"no prompts found in {prompt_dir}/")
    return jobs


def find_prompt_file(f, prompt_dir):
    """
    Resolve a --prompt-file value. Accepts, in this order:
      an existing path (relative to cwd or absolute), the same path relative to this script's folder
      (so `prompts/summary.txt` works from any cwd), or a bare name / filename looked up in prompt_dir.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    base = os.path.basename(f)
    cands = [f, os.path.join(here, f),
             os.path.join(prompt_dir, f), os.path.join(prompt_dir, f + ".txt"),
             os.path.join(prompt_dir, base), os.path.join(prompt_dir, base + ".txt")]
    for cand in cands:
        if os.path.isfile(cand):
            return cand
    sys.exit(f"prompt file not found: {f} (looked in ./, {here}/ and {prompt_dir}/)")


def load_jobs(args, prompt_dir):
    """Return [(name, prompt_text, prompt_path_or_None), ...] from --prompt / --prompt-file / prompts/."""
    jobs = []
    if args.prompt:
        jobs.append(("inline", args.prompt, None))
    for f in args.prompt_file:
        cand = find_prompt_file(f, prompt_dir)
        name = os.path.splitext(os.path.basename(cand))[0]
        with open(cand, encoding="utf-8") as fh:
            jobs.append((name, fh.read(), os.path.relpath(cand)))
    if not jobs:
        jobs = load_prompt_dir(prompt_dir)
    return jobs


def choose_model(args):
    if args.model:
        return args.model
    if args.pick_model:
        return pick_model(args.url, args.api_key)
    return list_models(args.url, args.api_key)[0]


def make_gpu_sampler(args):
    if args.no_gpu:
        return None
    gpu = GpuSampler(args.gpu_interval)
    return gpu if gpu.available else None


def main():
    args = build_parser().parse_args()
    global LOG_WORDS
    LOG_WORDS = args.log_words

    if args.list_models:
        for m in list_models(args.url, args.api_key):
            print(m)
        return

    prompt_dir, log_dir = resolve_dirs(args)
    jobs = load_jobs(args, prompt_dir)
    model = choose_model(args)
    print(f"using model: {model}")
    print(f"prompts: {len(jobs)}   runs each: {args.runs}" + (f"   thinking: {args.thinking}" if args.thinking else "")
          + (f"   reasoning_effort: {args.reasoning_effort}" if args.reasoning_effort else ""))
    gpu = make_gpu_sampler(args)

    for name, prompt, ppath in jobs:
        print(f"\n################ {name}  ({ppath or 'inline'}) ################")
        runs = []
        response_text = reasoning_text = ""
        for i in range(args.runs):
            m, response_text, reasoning_text = run_once(args.url, args.api_key, model, prompt,
                                                        args.system, args.max_tokens, args.temperature,
                                                        gpu, args.thinking, args.reasoning_effort)
            runs.append(m)
            print_metrics(m, f"run {i + 1}/{args.runs}")

        if args.runs > 1:
            keys = ["ttft_s", "total_time_s", "generation_time_s", "prompt_tokens",
                    "completion_tokens", "total_tokens", "generation_tokens_per_s", "overall_tokens_per_s"]
            avg = {"model": model, "gpu": None}
            for k in keys:
                vals = [r[k] for r in runs if r[k] is not None]
                avg[k] = sum(vals) / len(vals) if vals else None
                if k.endswith("_tokens") and avg[k] is not None:
                    avg[k] = round(avg[k])
            print_metrics(avg, f"average of {args.runs} runs")
        else:
            avg = runs[0]

        if not args.quiet:
            if reasoning_text and args.show_reasoning:
                print("\n--- reasoning (last run) ---")
                print(reasoning_text)
            note = empty_response_note(runs[-1], response_text, reasoning_text)
            print("\n--- response (last run) ---" + (f"  {note}" if note else ""))
            print(response_text)

        log_path = args.log or os.path.join(log_dir, f"{name}.txt")
        write_log(log_path, name, prompt, args.system, avg, runs, response_text, reasoning_text,
                  prompt_path=ppath, thinking=args.thinking, args=args)
        print(f"\nlogged to {os.path.relpath(log_path)}")

if __name__ == "__main__":
    main()