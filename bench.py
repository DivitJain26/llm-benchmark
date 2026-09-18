#!/usr/bin/env python3
"""
Simple vLLM request benchmark.

Hits a vLLM server (OpenAI-compatible API), streams the response, and reports:
  - time to first token (TTFT)
  - total request time
  - prompt / completion / total token counts (from vLLM's usage field)
  - generation tokens/sec
The response and metrics are printed in the terminal AND appended (with the prompt)
to one .txt log file.

Examples:
  python bench_vllm.py --prompt "Explain quicksort in 3 lines"
  python bench_vllm.py --prompt-file prompt.txt
  python bench_vllm.py --prompt-file prompt.txt --runs 5 --log my_bench.txt
  python bench_vllm.py --pick-model --prompt-file prompt.txt
  python bench_vllm.py --list-models
  VLLM_API_KEY=... python bench_vllm.py --model Qwen/Qwen3-14B --prompt "What is a LLM?"
"""

import argparse
import json
import os
import sys
import time

import requests


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


def run_once(base_url, api_key, model, prompt, system, max_tokens, temperature):
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

    text_parts = []
    reasoning_parts = []
    usage = {}
    ttft = None

    t0 = time.perf_counter()
    with requests.post(f"{base_url}/chat/completions", headers=headers(api_key),
                       json=payload, stream=True, timeout=600) as r:
        r.raise_for_status()
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
                delta = choice.get("delta", {})
                reasoning = delta.get("reasoning_content")   # Qwen3 "thinking"
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
    }, response_text, reasoning_text


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


def write_log(path, prompt, system, summary, runs, response_text, reasoning_text=""):
    """Append one clean block per benchmark to a single .txt log."""
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    lines = []
    lines.append("=" * 72)
    lines.append(f"{ts}  |  model: {summary['model']}  |  runs: {len(runs)}")
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
    if len(runs) > 1:
        lines.append("")
        lines.append("[per run]  ttft_s / total_s / completion_tokens / gen_tok_s")
        for i, r in enumerate(runs, 1):
            lines.append(f"  run {i}: {fmt(r['ttft_s'])} / {fmt(r['total_time_s'])} / "
                         f"{fmt(r['completion_tokens'])} / {fmt(r['generation_tokens_per_s'], 1)}")
    if system:
        lines.append("")
        lines.append("[system]")
        lines.append(system.rstrip())
    lines.append("")
    lines.append("[prompt]")
    lines.append(prompt.rstrip())
    lines.append("")
    if reasoning_text:
        lines.append("[reasoning]")
        lines.append(reasoning_text.rstrip())
        lines.append("")
    lines.append("[response]")
    lines.append(response_text.rstrip())
    lines.append("")
    lines.append("")
    with open(path, "a", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    p = argparse.ArgumentParser(description="Benchmark a single vLLM request")
    p.add_argument("--url", default=os.environ.get("VLLM_URL", "http://localhost:8000/v1"),
                   help="vLLM base URL (env: VLLM_URL)")
    p.add_argument("--api-key", default=os.environ.get("VLLM_API_KEY"),
                   help="API key, sent as Bearer token (env: VLLM_API_KEY)")
    p.add_argument("--model", default=os.environ.get("VLLM_MODEL"),
                   help="model id (env: VLLM_MODEL; default: first model from /v1/models)")
    p.add_argument("--pick-model", action="store_true", help="show a numbered list of models and choose one")
    p.add_argument("--list-models", action="store_true", help="print available models and exit")
    p.add_argument("--prompt", default=None, help="prompt text")
    p.add_argument("--prompt-file", default=None, help="read prompt from this file")
    p.add_argument("--system", default=None, help="optional system prompt")
    p.add_argument("--log", default="bench_log.txt", help="append prompt, response and metrics here")
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--runs", type=int, default=1, help="repeat N times and average")
    p.add_argument("--show-reasoning", action="store_true", help="also print the model's reasoning/thinking in the terminal")
    p.add_argument("--quiet", action="store_true", help="don't print the response in the terminal (it is still written to the log)")
    args = p.parse_args()

    if args.list_models:
        for m in list_models(args.url, args.api_key):
            print(m)
        return

    if args.prompt_file:
        with open(args.prompt_file, encoding="utf-8") as f:
            prompt = f.read()
    elif args.prompt:
        prompt = args.prompt
    else:
        sys.exit("Give --prompt or --prompt-file")

    if args.model:
        model = args.model
    elif args.pick_model:
        model = pick_model(args.url, args.api_key)
    else:
        model = list_models(args.url, args.api_key)[0]
    print(f"using model: {model}\n")

    runs = []
    response_text = reasoning_text = ""
    for i in range(args.runs):
        m, response_text, reasoning_text = run_once(args.url, args.api_key, model, prompt,
                                                    args.system, args.max_tokens, args.temperature)
        runs.append(m)
        print_metrics(m, f"run {i + 1}/{args.runs}")

    if args.runs > 1:
        keys = ["ttft_s", "total_time_s", "generation_time_s", "prompt_tokens",
                "completion_tokens", "total_tokens", "generation_tokens_per_s", "overall_tokens_per_s"]
        avg = {"model": model}
        for k in keys:
            vals = [r[k] for r in runs if r[k] is not None]
            avg[k] = sum(vals) / len(vals) if vals else None
        print_metrics(avg, f"average of {args.runs} runs")
    else:
        avg = runs[0]

    if not args.quiet:
        if reasoning_text and args.show_reasoning:
            print("\n--- reasoning (last run) ---")
            print(reasoning_text)
        print("\n--- response (last run) ---")
        print(response_text)

    write_log(args.log, prompt, args.system, avg, runs, response_text, reasoning_text)
    print(f"\nlogged to {args.log}")

if __name__ == "__main__":
    main()