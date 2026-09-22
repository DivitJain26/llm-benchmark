# vLLM request benchmark

A small script that sends one prompt to a running vLLM server and reports:

- time to first token (TTFT)
- total request time
- prompt / completion / total token counts (exact, from vLLM's `usage` field)
- generation tokens/sec and overall tokens/sec
- GPU utilisation (avg / peak %) and peak memory per GPU during the request, sampled with `nvidia-smi`

**Where the output goes:** the response and metrics are printed in the terminal
**and** appended, together with the prompt, to a single `.txt` log file
(`bench_log.txt` by default). Every run adds a new block, so one file holds your
whole history.

## Folder layout

```
bench_vllm.sh / bench_vllm.py     single request at a time
bench_vllm_concurrent.py          many requests at once (load test) — same flags + --concurrency
prompts/          one .txt per prompt  (created automatically with prompts/example.txt)
logs/             one log per prompt   (prompts/summary.txt -> logs/summary.txt, appended every run)
                  concurrent runs go to logs/summary_concurrent.txt (or logs/mixed_concurrent.txt with --mix)
```

Drop as many `.txt` files as you like in `prompts/`. Running with no prompt arguments benchmarks
**all of them**, each into its own log. An inline `--prompt "..."` is logged to `logs/inline.txt`.

## Two versions, same flags, same output

| File | Use when |
|---|---|
| `bench_vllm.sh` | **no Python in the container** (e.g. RunPod vLLM image). Needs only `curl` + grep/sed/awk. |
| `bench_vllm.py` | Python is available. Needs `requests` (`bash setup.sh` installs it). |
| `bench_vllm_concurrent.py` | **Load testing**: N requests in flight at once. Same flags as `bench_vllm.py` plus `--concurrency`, `--requests`, `--mix`, `--all-responses`. Imports `bench_vllm.py`, so keep the two files together. |

## Run

```bash
python3 bench_vllm.py \
  --prompt-file prompts/krisala-lead-suggestion.txt \
  --log logs/qwen3-14b-thinking-off_krisala-lead-suggestion.txt \
  --max-tokens 5000 \
  --temperature 0.3 \
  --thinking off
```

Replace `./bench_vllm.sh` with `python3 bench_vllm.py` for the Python version — identical flags.
`--prompt-file` accepts a name (`summary`), a filename (`summary.txt`) or any path.

## Qwen3 thinking

Qwen3 streams its reasoning as `reasoning_content` before the answer. The script counts the first
reasoning token as time-to-first-token, saves the reasoning in the log under `[reasoning]`, and
prints it in the terminal only if you pass `--show-reasoning`. Completion tokens from vLLM include
the reasoning tokens.

## Choosing the model

```bash
python bench_vllm.py --list-models                       # show what the server has
python bench_vllm.py --pick-model --prompt-file prompt.txt   # numbered menu, pick one
python bench_vllm.py --model Qwen/Qwen2.5-7B-Instruct --prompt-file prompt.txt   # by id
```

With none of these, the first model the server reports is used.

## Concurrent load test

`bench_vllm_concurrent.py` fires many requests at the same time and reports the numbers that only
exist under load. It accepts every flag `bench_vllm.py` does, plus:

| Option | Default | Meaning |
|---|---|---|
| `--concurrency` | `4` | requests in flight at once (thread pool size) |
| `--requests` | `= --concurrency` | total requests per batch; if larger than `--concurrency`, the rest queue up |
| `--mix` | off | one batch cycling through **all** selected prompts (realistic mixed traffic) instead of one batch per prompt |
| `--all-responses` | off | write every response to the log (default: only the first, to keep the log readable) |
| `--runs` | `1` | repeat the whole batch N times and average the aggregate numbers |

```bash
python3 bench_vllm_concurrent.py --concurrency 8                                 # 8 x every prompt in prompts/
python3 bench_vllm_concurrent.py --prompt-file krisala-lead-suggestion \
  --concurrency 16 --requests 64 --max-tokens 5000 --thinking off                # 64 requests, 16 at a time
python3 bench_vllm_concurrent.py --mix --concurrency 8 --requests 32 --runs 3    # mixed prompts, averaged
```

Logs go to `logs/<prompt>_concurrent.txt` (or `logs/mixed_concurrent.txt` with `--mix`) so they never
get mixed into the single-request logs. `--log` still forces one file.

### What it measures

- **[load]** concurrency, requests sent / ok / failed, wall time for the whole batch, requests/s
- **[throughput]** *aggregate* completion tok/s and total tok/s over the batch — this is the number that
  should climb as you raise `--concurrency` until the GPU saturates; the per-request tok/s will fall
- **[latency]** mean / p50 / p95 / max for time-to-first-token, total request time and per-request
  generation tok/s — p95 TTFT is what your users feel under load
- **[per request]** a row per request: when it started (relative to the batch), TTFT, total time,
  token counts, tok/s, which prompt it was
- **[gpu]** `nvidia-smi` sampled across the **whole batch**, not per request (avg / peak utilisation,
  peak memory per GPU). Use `--gpu-interval 0.1` for a finer picture on short batches.

Typical use: run the same prompt at `--concurrency 1 4 8 16 32` and watch where aggregate tok/s stops
growing and p95 TTFT starts climbing — that's the useful capacity of the model on that GPU.

### Log format

```
========================================================================
2026-09-22 08:44:07  |  model: Qwen/Qwen3-14B  |  prompt: summary  |  concurrency: 8  |  requests: 32  |  runs: 1  |  thinking: off
max_tokens: 512  |  temperature: 0.0
========================================================================

[load]
  concurrency          : 8
  requests             : 32  (ok 32, failed 0)
  wall time            : 41.2 s
  requests/s           : 0.78

[throughput]  aggregate over the whole batch (this is the number that scales with load)
  completion tok/s     : 812.4
  total tok/s          : 1105.0   (prompt + completion)
  prompt tokens        : 12064 total   (377 avg / request)
  completion tokens    : 33470 total   (1046 avg / request)

[latency]  per request         mean / p50 / p95 / max
  time to first token  : 0.412 / 0.390 / 0.780 / 0.910 s
  total time           : 10.1 / 9.8 / 12.4 / 13.0 s
  generation tok/s     : 104.2 / 103.0 / 118.7 / 121.0   (per request, drops as concurrency rises)

[gpu]  sampled every 0.25s across the whole batch with nvidia-smi
  gpu 0: util avg 97%  max 100%  |  mem peak 31276/81920 MiB  |  165 samples

[per request]   #   start_s   ttft_s   total_s   prompt_tok   comp_tok   gen_tok_s   prompt
    1     0.000    0.390     9.812          377       1040       107.9   summary
  ...

[prompt]  file: prompts/summary.txt  (412 words, first 30 shown)
...

[response]  request 1 of 32 (others omitted; use --all-responses)
...
```

With `--runs` > 1 the block starts with the average, then one `--- batch N ---` section (with its own
GPU numbers and per-request table) per run.

## Options

| Option | Default | Meaning |
|---|---|---|
| `--url` | `$VLLM_URL` or `http://localhost:8000/v1` | vLLM base URL |
| `--api-key` | `$VLLM_API_KEY` | Bearer token |
| `--model` | `$VLLM_MODEL` or first from server | model id |
| `--pick-model` | off | interactive numbered model menu |
| `--list-models` | off | print models and exit |
| `--prompt` | – | inline prompt (logged to `logs/inline.txt`) |
| `--prompt-file` | all of `prompts/` | prompt name/file, repeatable |
| `--prompt-dir` | `prompts` | where prompts live |
| `--log-dir` | `logs` | where logs go |
| `--log` | `logs/<prompt>.txt` | force a single log file for everything |
| `--system` | – | optional system prompt |
| `--max-tokens` | `512` | max output tokens |
| `--temperature` | `0.0` | sampling temperature |
| `--runs` | `1` | repeat N times and print the average |
| `--thinking` | model default | `on` / `off` — only for models with a thinking mode (Qwen3 etc.); sends vLLM's `enable_thinking` |
| `--log-words` | `30` | max prompt words written to the log (file name and word count are always logged) |
| `--no-gpu` | off | skip GPU sampling |
| `--gpu-interval` | `0.25` | seconds between `nvidia-smi` samples |
| `--show-reasoning` | off | print Qwen3 reasoning in the terminal too |
| `--quiet` | off | don't print the response in the terminal (still goes to the log) |

## Thinking mode (Qwen3 and similar)

Some models can run with or without a "thinking" phase. `--thinking off` disables it (much lower TTFT
and completion tokens), `--thinking on` forces it; without the flag the model's default is used. The
setting is recorded in the log header. Models without a thinking mode ignore the flag.

## Log format

```
========================================================================
2026-09-18 06:03:44  |  model: Qwen/Qwen3-14B  |  prompt: summary  |  runs: 3  |  thinking: off
========================================================================

[metrics]
  time to first token  : 0.120 s
  total time           : 2.500 s
  prompt tokens        : 20
  completion tokens    : 150
  total tokens         : 170
  generation tok/s     : 63.0
  overall tok/s        : 60.0

[per run]  ttft_s / total_s / completion_tokens / gen_tok_s
  run 1: 0.118 / 2.480 / 150 / 63.5
  ...

[gpu]  sampled every 0.25s with nvidia-smi
run 1:
  gpu 0: util avg 92%  max 98%  |  mem peak 31276/81920 MiB  |  10 samples
  ...

[prompt]  file: prompts/summary.txt  (412 words, first 30 shown)
Summarize the following meeting notes into five bullet points ... 

[reasoning]
...

[response]
...
```

`[gpu]` appears only when `nvidia-smi` is available and `--no-gpu` isn't set; `[per run]` only when `--runs` > 1; `[system]` only with `--system`; `[reasoning]` only when the model emits it.

## GPU numbers — what they mean

`nvidia-smi` is polled every 0.25 s from the moment the request is sent until the last token
arrives. Utilisation is the average and peak of those samples; memory is the peak `memory.used`.
For very short requests you may only get a few samples — use a longer prompt / `--max-tokens` or
`--gpu-interval 0.1` for a finer picture. Memory will look almost constant because vLLM pre-allocates
its KV cache at startup; the interesting number is utilisation.