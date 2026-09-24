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
| `bench_vllm_concurrent.py` | **Load testing**: N requests in flight at once. Same flags as `bench_vllm.py` plus `--concurrency`, `--requests`, `--duration`, `--bucket`, `--mix`, `--all-responses`. Imports `bench_vllm.py`, so keep the two files together. |

## Run

```bash
python3 bench_vllm.py \
  --prompt-file prompts/krisala-lead-suggestion.txt \
  --log logs/qwen3-14b-thinking-off_krisala.txt \
  --max-tokens 5000 \
  --temperature 0.3 \
  --thinking off
```

Replace `./bench_vllm.sh` with `python3 bench_vllm.py` for the Python version — identical flags.
`--prompt-file` accepts a name (`summary`), a filename (`summary.txt`) or a path such as
`prompts/summary.txt` (resolved against the script's folder too, so it works from any working directory).

## Reasoning models (Qwen3, gpt-oss, DeepSeek ...)

These models stream their reasoning before the answer, as `reasoning_content` (Qwen3, DeepSeek) or
`reasoning` (gpt-oss on newer vLLM). Both are read. The first reasoning token counts as
time-to-first-token, the reasoning goes to the log under `[reasoning]`, and it is printed in the
terminal only with `--show-reasoning`. Completion tokens from vLLM include the reasoning tokens.

If `[response]` is empty, the log says why. The usual cause is `finish reason: length`: the model
spent the whole `--max-tokens` budget on reasoning and was cut off before the answer. Raise
`--max-tokens`, or for gpt-oss pass `--reasoning-effort low` (Qwen3: `--thinking off`). The log
header records the effort level, and the concurrent bench counts `hit max_tokens` and
`empty responses` per batch so you can see when a whole run produced no answers.

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
| `--requests` | `= --concurrency` | total requests per batch; if larger than `--concurrency`, the rest queue up (ignored with `--duration`) |
| `--duration` | off | seconds to keep `--concurrency` requests in flight, instead of a fixed `--requests` count. No new requests are sent after the deadline; the ones already in flight finish, so wall time overshoots by up to one request's latency |
| `--bucket` | auto | timeline bucket size in seconds (auto picks about 10-12 buckets over the run) |
| `--mix` | off | one batch cycling **round-robin through every `.txt` in `prompts/`** (realistic mixed traffic) instead of one batch per prompt. `--prompt` / `--prompt-file` are ignored |
| `--all-responses` | off | write every response to the log (default: only the first, to keep the log readable) |
| `--runs` | `1` | repeat the whole batch N times and average the aggregate numbers |

```bash
python3 bench_vllm_concurrent.py --concurrency 8                                 # 8 x every prompt in prompts/
python3 bench_vllm_concurrent.py --prompt-file krisala-lead-suggestion \
  --concurrency 16 --requests 64 --max-tokens 5000 --thinking off                # 64 requests, 16 at a time
python3 bench_vllm_concurrent.py --mix --concurrency 8 --requests 32 --runs 3    # mixed prompts, averaged
python3 bench_vllm_concurrent.py --mix --concurrency 16 --duration 300 --thinking off   # 5 min soak, all prompts
python3 bench_vllm_concurrent.py --prompt-file prompts/krisala-lead-suggestion-1.txt \
  --concurrency 8 --duration 60                                                   # one prompt for 60 s

python3 bench_vllm_concurrent.py \
  --prompt-file krisala-lead-suggestion \
  --concurrency 50 \
  --requests 50 \
  --temperature 0.3 \
  --max-tokens 5000 \
  --thinking off \
  --log logs/Qwen3-32B-AWQ_krisala_50-req.txt
```

Logs go to `logs/<prompt>_concurrent.txt` (or `logs/mixed_concurrent.txt` with `--mix`) so they never
get mixed into the single-request logs. `--log` still forces one file.

### What it measures

- **[load]** concurrency and the *achieved* average in-flight count, requests sent / ok / failed and
  error rate, wall time for the whole batch, requests/s. With `--duration` also the target duration;
  `requests` is then how many were sent in that window.
- **[window]** (`--duration` only) the steady-state numbers over the first N seconds, ignoring the tail:
  requests completed inside the window and req/s, how many were still in flight at the deadline and
  how long the tail took, completion tok/s and total tok/s over exactly N seconds. Compare runs on
  these numbers, since the tail length varies with the last requests' latency.
- **[throughput]** *aggregate* completion tok/s and total tok/s over the batch (including the tail) —
  this is the number that should climb as you raise `--concurrency` until the GPU saturates; the
  per-request tok/s will fall. Also completion tokens min / p50 / max and how many requests hit
  `--max-tokens` (`finish_reason = length`), which means the answers were cut off.
- **[latency]** mean / p50 / p95 / p99 / max for time-to-first-token, total request time, inter-token
  latency (ms per output token after the first, what streaming feels like) and per-request generation
  tok/s — p95/p99 TTFT is what your users feel under load
- **[errors]** count per distinct error message (HTTP status + vLLM's reason), when anything failed
- **[per prompt]** with `--mix`: one row per prompt in the folder — count, ok / failed, avg prompt and
  completion tokens, TTFT mean / p95, total time mean / p95, inter-token latency, tok/s, truncations
- **[timeline]** one row per `--bucket` seconds: requests started / completed / failed, in flight at
  the end of the bucket, completion tokens and tok/s, TTFT mean / p95 and mean total time. On a long
  `--duration` run this shows warm-up, drift and the point where TTFT starts climbing. Printed for
  every `--duration` run and for count runs that last at least two buckets.
- **[per request]** a row per request: start and end (relative to the batch; `*` = finished after the
  deadline), TTFT, total time, inter-token latency, token counts, tok/s, finish reason, which prompt
- **[gpu]** `nvidia-smi` sampled across the **whole batch**, not per request (avg / peak utilisation,
  peak memory per GPU). Use `--gpu-interval 0.1` for a finer picture on short batches.

The log header also records the URL and the wall-clock start / end of the run, and with `--runs > 1`
each batch gets its own full section plus an averaged one at the top.

Typical use: run the same prompt at `--concurrency 1 4 8 16 32` and watch where aggregate tok/s stops
growing and p95 TTFT starts climbing — that's the useful capacity of the model on that GPU. For a
soak test use `--mix --duration 600` and read the `[window]` and `[timeline]` sections.

### Log format

`--mix --concurrency 8 --duration 120 --thinking off` looks like this (`[window]`, `[per prompt]` and
the `*` marker only appear with `--duration` / `--mix`; the rest is the same for count runs):

```
========================================================================
2026-09-23 10:12:07  |  model: Qwen/Qwen3-14B  |  prompt: mixed  |  concurrency: 8  |  duration: 120.0s  |  requests: 96  |  runs: 1  |  thinking: off
max_tokens: 5000  |  temperature: 0.3  |  url: http://localhost:8000/v1  |  started: 2026-09-23 10:10:01  |  ended: 2026-09-23 10:12:07
========================================================================

[load]
  concurrency          : 8   (achieved avg in flight 7.91)
  duration             : 120.0 s   (no new requests after this; in-flight ones finish)
  requests             : 96  (ok 96, failed 0, error rate 0.0 %)
  wall time            : 126.4 s
  requests/s           : 0.76

[window]  first 120.0 s only — steady-state numbers without the tail after the deadline
  completed in window  : 88   (0.73 req/s)
  in flight at deadline: 8   (tail after deadline 6.4 s)
  completion tok/s     : 788.1   (94572 tokens)
  total tok/s          : 1069.4   (prompt + completion)

[throughput]  aggregate over the whole batch incl. tail (this is the number that scales with load)
  completion tok/s     : 794.6
  total tok/s          : 1080.2   (prompt + completion)
  prompt tokens        : 36096 total   (376 avg / request)
  completion tokens    : 100440 total   (1046 avg / request, min 812 / p50 1031 / max 1420)
  hit max_tokens       : 0 requests   (finish_reason = length)

[latency]  per request         mean / p50 / p95 / p99 / max
  time to first token  : 0.412 / 0.390 / 0.780 / 0.880 / 0.910 s
  total time           : 10.1 / 9.8 / 12.4 / 12.9 / 13.0 s
  inter-token latency  : 9.3 / 9.2 / 10.1 / 10.6 / 10.8 ms   (time per output token after the first)
  generation tok/s     : 104.2 / 103.0 / 118.7 / 120.4 / 121.0   (per request, drops as concurrency rises)

[per prompt]   n    ok  fail   prompt_tok   comp_tok   ttft_mean   ttft_p95   total_mean   total_p95   itl_ms   gen_tok_s   hit_max   prompt
            24    24     0          402       1102       0.418      0.790       10.6        12.6       9.4      104.0         0   krisala-lead-suggestion-1
            24    24     0          351        990       0.405      0.771        9.6        12.1       9.3      104.5         0   krisala-lead-suggestion-2
   ...

[timeline]  per 10 s bucket — watch for tok/s dropping or TTFT climbing over the run
       t0       t1   started   completed   failed   in_flight   comp_tok   comp_tok/s   ttft_mean   ttft_p95   total_mean
        0       10         8           0        0           8          0          0.0          -          -            -
       10       20         8           8        0           8       8412        841.2       0.421      0.780       10.2
   ...
      120      126         0           8        0           0       8360       1306.2       0.402      0.744       10.0

[gpu]  sampled every 0.25s across the whole batch with nvidia-smi
  gpu 0: util avg 97%  max 100%  |  mem peak 31276/81920 MiB  |  505 samples

[per request]   #   start_s    end_s   ttft_s   total_s   itl_ms   prompt_tok   comp_tok   gen_tok_s   finish   prompt   (* = finished after the deadline)
    1     0.000    9.812    0.390     9.812      9.1          377       1040       107.9     stop   krisala-lead-suggestion-1
  ...
   96   119.850  126.401*   0.402     6.551      9.0          352        720       109.9     stop   krisala-lead-suggestion-4

[prompt]  file: prompts/krisala-lead-suggestion-1.txt  (412 words, first 30 shown)
...

[response]  request 1 of 96 (others omitted; use --all-responses)
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
| `--reasoning-effort` | model default | `low` / `medium` / `high` — for models that take one (gpt-oss); sends `reasoning_effort`. `low` leaves more of `--max-tokens` for the answer |
| `--log-words` | `30` | max prompt words written to the log (file name and word count are always logged) |
| `--no-gpu` | off | skip GPU sampling |
| `--gpu-interval` | `0.25` | seconds between `nvidia-smi` samples |
| `--show-reasoning` | off | print the model's reasoning in the terminal too |
| `--quiet` | off | don't print the response in the terminal (still goes to the log) |

## Thinking mode (Qwen3 and similar)

Some models can run with or without a "thinking" phase. `--thinking off` disables it (much lower TTFT
and completion tokens), `--thinking on` forces it; without the flag the model's default is used. The
setting is recorded in the log header. Models without a thinking mode ignore the flag. gpt-oss
has no on/off switch; use `--reasoning-effort low|medium|high` instead.

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