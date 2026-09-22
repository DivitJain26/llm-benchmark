# vLLM request benchmark

A small script that sends one prompt to a running vLLM server and reports:

- time to first token (TTFT)
- total request time
- prompt / completion / total token counts (exact, from vLLM's `usage` field)
- generation tokens/sec and overall tokens/sec
- GPU utilisation (avg / peak %) and peak memory per GPU during the request, sampled with `nvidia-smi`
- **concurrent mode** (`--concurrency N`): fire N requests at once and get aggregate tok/s, requests/s,
  TTFT / latency percentiles, a per-request table and every response in the log

**Where the output goes:** the response and metrics are printed in the terminal
**and** appended, together with the prompt, to a single `.txt` log file
(`bench_log.txt` by default). Every run adds a new block, so one file holds your
whole history.

## Folder layout

```
bench_vllm.sh / bench_vllm.py
prompts/          one .txt per prompt  (created automatically with prompts/example.txt)
logs/             one log per prompt   (prompts/summary.txt -> logs/summary.txt, appended every run)
```

Drop as many `.txt` files as you like in `prompts/`. Running with no prompt arguments benchmarks
**all of them**, each into its own log. An inline `--prompt "..."` is logged to `logs/inline.txt`.

## Two versions, same flags, same output

| File | Use when |
|---|---|
| `bench_vllm.sh` | **no Python in the container** (e.g. RunPod vLLM image). Needs only `curl` + grep/sed/awk. |
| `bench_vllm.py` | Python is available. Needs `requests` (`bash setup.sh` installs it). |

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
`--prompt-file` accepts a name (`summary`), a filename (`summary.txt`) or any path.

## Concurrent load test

`--concurrency N` sends N requests **in parallel** (all with the same prompt) instead of one after another.
`--requests M` sets the total number of requests per prompt (default: same as `--concurrency`); with
`M > N` the extra requests queue up so at most N are in flight, like real traffic.

```bash
python3 bench_vllm.py --prompt-file krisala-lead-suggestion --concurrency 8              # 8 at once
python3 bench_vllm.py --prompt-file krisala-lead-suggestion --concurrency 4 --requests 16 # 16 total, 4 in flight
python3 bench_vllm.py --prompt-file summary --concurrency 8 --runs 3                      # repeat the batch 3x
python3 bench_vllm.py --prompt-file summary --concurrency 8 --log-responses first         # log only response 1
```

The terminal shows each request as it finishes, then the batch summary. The log gets:

- `[concurrent metrics]` — wall time, requests/s, **aggregate tok/s** (sum of completion tokens / wall time),
  and avg / min / p50 / p95 / max of per-request TTFT, total time, generation tok/s and completion tokens
- `[per request]` — one row per request: start offset, TTFT, total, tokens, tok/s, ok / FAILED (with the error)
- `[gpu]` — sampled over the **whole batch**, not per request
- `[response req N ...]` — every request's response (and `[reasoning req N ...]` if the model emits it),
  each tagged with its own TTFT / total / tokens. `--log-responses first` keeps only the first, `none` skips them.

A failed request (HTTP error, timeout, ...) is recorded in the table and the batch continues; the
aggregate numbers only count successful requests. `--runs 3` repeats the whole batch three times and
logs each batch's metrics; responses are written from the last batch. With several prompt files, each
prompt gets its own batch and its own log, as in sequential mode.

Note: in a batch, each request's `generation tok/s` is what *that* client saw; `aggregate tok/s` is
what the server actually produced. The gap between them is the point of the test.

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
| `--system-file` | – | read the system prompt from a file |
| `--max-tokens` | `512` | max output tokens |
| `--temperature` | `0.0` | sampling temperature |
| `--runs` | `1` | repeat N times and print the average (concurrent mode: repeat the batch) |
| `--concurrency` | `1` | send this many requests in parallel; `1` = classic sequential mode |
| `--requests` | = `--concurrency` | total requests per prompt in concurrent mode (extra ones queue) |
| `--log-responses` | `all` | concurrent mode: `all` / `first` / `none` responses written to the log |
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

Concurrent mode writes a different block:

```
========================================================================
2026-09-22 09:31:47  |  model: Qwen/Qwen3-14B  |  prompt: summary  |  concurrent: 8 requests, 4 in flight  |  runs: 1
========================================================================

[concurrent metrics]
  requests             : 8  (ok 8, failed 0)  concurrency 4
  wall time            : 1.222 s
  requests/s           : 6.55
  prompt tokens (sum)  : 96
  completion tok (sum) : 80
  aggregate tok/s      : 65.5   (sum completion tokens / wall time)
  ttft per request     : avg 0.567 s  min 0.556 s  p50 0.567 s  p95 0.578 s  max 0.578 s
  total per request    : avg 0.609 s  min 0.608 s  p50 0.608 s  p95 0.611 s  max 0.611 s
  gen tok/s per request: avg 250.9  min 192.9  p50 243.4  p95 322.2  max 330.1
  completion tokens    : avg 10  min 10  p50 10  p95 10  max 10

[per request]
  req   start_s  ttft_s   total_s  prompt_tok  compl_tok  gen_tok_s  status
  1       0.000    0.578     0.608          12         10      330.1  ok
  2       0.001    0.578     0.610          12         10      307.4  ok
  ...

[gpu]  sampled with nvidia-smi during the whole batch
  gpu 0: util avg 97%  max 100%  |  mem peak 43871/46068 MiB  |  5 samples

[prompt]  file: prompts/summary.txt  (412 words, first 30 shown)
...

[response req 1  |  ttft 0.578 s  |  total 0.608 s  |  10 tok  |  330.1 tok/s]
...
[response req 2  |  ttft 0.578 s  |  total 0.610 s  |  10 tok  |  307.4 tok/s]
...
```

## GPU numbers — what they mean

`nvidia-smi` is polled every 0.25 s from the moment the request is sent until the last token
arrives. Utilisation is the average and peak of those samples; memory is the peak `memory.used`.
For very short requests you may only get a few samples — use a longer prompt / `--max-tokens` or
`--gpu-interval 0.1` for a finer picture. Memory will look almost constant because vLLM pre-allocates
its KV cache at startup; the interesting number is utilisation.