# vLLM benchmark

Two small Python scripts that measure a running vLLM server from the outside, the way a client sees it.

| Script | What it does |
|---|---|
| `bench_vllm.py` | Sends **one request at a time** and reports TTFT, total time, token counts, tok/s and GPU usage per request. Use it to check a prompt, a model, or a setting. |
| `bench_vllm_concurrent.py` | Sends **many requests at once** (load test) and reports throughput, latency percentiles, a timeline and failures. Use it to find out what the server can take. |

Both accept the same flags. The concurrent script imports the single one, so keep the two files together.

---

## 1. Quick start

```bash
bash setup.sh                                   # installs `requests` (makes a venv unless vllm is already installed)
export VLLM_URL=http://localhost:8000/v1        # default, change if the server is elsewhere
export VLLM_API_KEY=your-key                    # only if vllm serve was started with --api-key
export VLLM_MODEL=openai/gpt-oss-120b           # optional; otherwise the first model the server reports

python3 bench_vllm.py --list-models             # check the connection
python3 bench_vllm.py                           # run every prompt in prompts/ once
python3 bench_vllm_concurrent.py --concurrency 8 --duration 60   # 8 requests in flight for 60 s
```

Everything is printed in the terminal **and** appended to a `.txt` file in `logs/`. Every run adds a
new block to its log, so one file can hold the history of a whole experiment.

## 2. Files and folders

```
bench_vllm.py                 single-request benchmark
bench_vllm_concurrent.py      load test (needs bench_vllm.py next to it)
setup.sh                      one-time install
prompts/                      one .txt file per prompt; all of them are used unless you pick one
logs/                         bench_vllm.py         -> logs/<prompt>.txt
                              bench_vllm_concurrent -> logs/<prompt>_concurrent.txt, or logs/mixed_concurrent.txt with --mix
                              --log path.txt        -> forces one file for everything
```

Picking a prompt: `--prompt-file` takes a name (`summary`), a filename (`summary.txt`) or a path
(`prompts/summary.txt`). Paths are also resolved against the script's folder, so they work from any
working directory. Repeat the flag for several prompts. `--prompt "text"` sends an inline prompt and
logs it to `logs/inline.txt`. With no prompt flag at all, every `.txt` in `prompts/` is used.

---

## 3. Single request: `bench_vllm.py`

```bash
python3 bench_vllm.py \
  --prompt-file prompts/krisala-lead-suggestion-1.txt \
  --max-tokens 4000 \
  --temperature 0.3 \
  --reasoning-effort low \
  --runs 3 \
  --log logs/gpt-oss-120b_krisala.txt
```

Per request it reports:

| Metric | Meaning |
|---|---|
| time to first token (TTFT) | seconds until the first token (reasoning or answer) arrives. Mostly prompt processing. |
| total time | seconds until the stream ends |
| prompt / completion / total tokens | exact counts from vLLM's `usage` field. Completion tokens **include** reasoning tokens. |
| generation tok/s | completion tokens divided by the time after the first token: the decode speed |
| overall tok/s | completion tokens divided by total time |
| finish reason | `stop` = model finished. `length` = it hit `--max-tokens` and the answer was cut off. |
| gpu | `nvidia-smi` sampled every 0.25 s during the request: avg / peak utilisation, peak memory |

With `--runs N` the request is repeated and the averages are printed; each run is still listed in the log.
The response is printed in the terminal (`--quiet` to skip) and always written to the log. The model's
reasoning goes to the log too, and to the terminal with `--show-reasoning`.

Log block:

```
========================================================================
2026-09-23 13:57:18  |  model: openai/gpt-oss-120b  |  prompt: krisala-lead-suggestion-1  |  runs: 1
max_tokens: 4000  |  temperature: 0.3  |  url: http://localhost:8000/v1  |  reasoning_effort: low
========================================================================

[metrics]
  time to first token  : 1.820 s
  total time           : 16.246 s
  prompt tokens        : 17484
  completion tokens    : 1540
  total tokens         : 19024
  generation tok/s     : 106.8
  overall tok/s        : 94.8
  finish reason        : stop

[gpu]  sampled with nvidia-smi during each request
run 1:
  gpu 0: util avg 96%  max 100%  |  mem peak 76819/81920 MiB  |  54 samples

[prompt]  file: prompts/krisala-lead-suggestion-1.txt  (7025 words, first 30 shown)
PERSONA: You are a production Metroleads sales copilot ...

[reasoning]
...

[response]
...
```

`[per run]` appears with `--runs > 1`, `[system]` with `--system`, `[reasoning]` only when the model
emits it, `[gpu]` only when `nvidia-smi` is available and `--no-gpu` is not set.

---

## 4. Load test: `bench_vllm_concurrent.py`

### 4.1 Two ways to generate load

**Closed loop, `--concurrency N`** (default). N requests are always in flight. As soon as one
finishes, the next one is sent. The *rate* is whatever the server manages: roughly N divided by the
mean request latency. Use this to find the capacity of a model on a GPU.

**Open loop, `--rate R`.** R new requests are sent every second, evenly spaced, no matter how many are
still running. The rate is fixed, the in-flight count floats. Use this to check whether the server
survives a known traffic level. If `in flight` keeps rising, TTFT climbs, or requests fail, the server
cannot keep up with that rate.

> `--concurrency 100` is **not** 100 requests per second. It is 100 requests in flight. With a request
> that takes 20 s, that is about 5 requests per second. For requests per second use `--rate`.

How long to run: `--requests M` sends a fixed number and stops. `--duration S` keeps going for S seconds.
Nothing new is sent after the deadline, but requests already running are allowed to finish (bounded
by `--drain`).

### 4.2 Worked example

```bash
python3 bench_vllm_concurrent.py \
  --mix \
  --concurrency 100 \
  --duration 300 \
  --max-tokens 4000 \
  --temperature 0.3 \
  --reasoning-effort medium \
  --all-responses \
  --log logs/gpt-oss-120b_effort-medium_c100_300s_krisala.txt
```

| Flag | What it does here |
|---|---|
| `--mix` | every `.txt` in `prompts/` is used, round-robin: request 1 gets prompt 1, request 2 gets prompt 2, ... and around again. One batch, one log, plus a `[per prompt]` table. Without `--mix`, each prompt gets its own separate batch and log. |
| `--concurrency 100` | closed loop: 100 requests in flight at all times for the whole run |
| `--duration 300` | keep that up for 5 minutes. The number of requests is whatever fits: at ~25 s per request that is about 1200. |
| `--max-tokens 4000` | output budget per request. Reasoning counts against it: if the model spends it all on reasoning, the answer is empty and `finish reason` is `length`. |
| `--temperature 0.3` | sampling temperature (default 0.0) |
| `--reasoning-effort medium` | gpt-oss effort level; recorded in the log header so runs at different levels can be told apart |
| `--all-responses` | write **every** response to the log instead of only the first. With ~1200 responses of up to 4000 tokens the log will be tens of MB; leave it off unless you need to inspect the outputs. |
| `--log ...` | one log file for this run. Name it after what the run *is* (model, effort, concurrency, duration) so it is still meaningful later. |

What happens: the script prints one line per finished request as it goes, with a running done and
failed count. After 300 s it stops sending, waits for the last 100 to finish, and prints the summary.
The same summary plus the per-request table goes to the log.

What to look at first, in this order:

1. `[load]` → `failed`. Any failures mean the server rejected or dropped requests (context length,
   timeouts, out of memory). Details are under `[errors]` and `[failed requests]`.
2. `[throughput]` → `hit max_tokens` and `empty responses`. If most requests hit the limit, the numbers
   below are for cut-off answers. Raise `--max-tokens` or lower `--reasoning-effort`.
3. `[window]` → `completion tok/s` and `completed in window`. This is the sustained throughput at this
   concurrency, measured over exactly 300 s.
4. `[latency]` → `time to first token` p95 and `total time` p95. This is what a user would wait.
5. `[timeline]` → is `comp_tok/s` flat and `ttft_p95` flat over the run, or degrading?

Then run the same command at `--concurrency 25` and `50`. Where `completion tok/s` stops growing and
p95 TTFT starts climbing, that is the useful capacity of the model on that GPU.

### 4.3 More recipes

```bash
# capacity sweep: same prompt, rising concurrency, 60 s each
for c in 1 4 8 16 32 64; do
  python3 bench_vllm_concurrent.py --prompt-file krisala-lead-suggestion-1 \
    --concurrency $c --duration 60 --max-tokens 2000 --log logs/sweep_c$c.txt
done

# fixed number of requests, queued: 64 requests, 16 at a time, repeated 3 times and averaged
python3 bench_vllm_concurrent.py --prompt-file summary --concurrency 16 --requests 64 --runs 3

# open loop: can the server take 5 requests per second for 5 minutes?
python3 bench_vllm_concurrent.py --mix --rate 5 --duration 300 --drain 120 --max-tokens 2000

# 10 minute soak with all prompts, 1 s timeline buckets
python3 bench_vllm_concurrent.py --mix --concurrency 16 --duration 600 --bucket 30
```

### 4.4 Load-test flags

On top of every flag in section 6:

| Flag | Default | Meaning |
|---|---|---|
| `--concurrency N` | `4` | closed loop: requests in flight at once. Ignored with `--rate`. |
| `--requests M` | `= concurrency` | total requests per batch (the rest queue behind the concurrency). Ignored with `--duration`. |
| `--duration S` | off | run for S seconds instead of a fixed count |
| `--rate R` | off | open loop: send R new requests per second for `--duration` seconds (or `--requests` in total) |
| `--max-in-flight M` | `2000` | safety cap for `--rate`: while M requests are in flight, new ones are skipped and counted instead of sent |
| `--drain S` | wait for all | after the deadline, wait at most S seconds for running requests. The rest are reported as `unfinished` and excluded from the latency stats. Set it on open-loop runs, otherwise an overloaded server keeps the bench waiting for its whole backlog. |
| `--mix` | off | one batch over all prompts in `prompts/`, round-robin. `--prompt` / `--prompt-file` are ignored. |
| `--runs N` | `1` | repeat the whole batch N times; the log gets each batch plus an average |
| `--bucket S` | auto | timeline bucket size in seconds. Auto picks about 10 to 12 buckets. |
| `--all-responses` | off | log every response, not just the first |

### 4.5 Reading the log

Sections appear in this order. Some only show up when relevant.

| Section | What is in it | When |
|---|---|---|
| header | timestamp, model, prompt or `mixed`, concurrency or rate, duration, requests, runs, thinking / reasoning effort, max_tokens, temperature, url, wall-clock start and end | always |
| `[load]` | closed loop: target concurrency and the achieved average in flight. Open loop: target rate vs sent rate, max send lag (how late the client was, should be ~0), in-flight peak / avg / at deadline, skipped requests. Both: requests sent and ok, **failed** with error rate, **unfinished** if `--drain` expired, wall time, completed per second. | always |
| `[window]` | the same throughput numbers over exactly the first `--duration` seconds, ignoring the tail after the deadline. Plus how many were in flight at the deadline and how long the tail took. Compare runs on these. | `--duration` |
| `[throughput]` | aggregate completion tok/s and total tok/s over the whole batch including the tail; prompt and completion token totals and averages; completion tokens min / p50 / max; **hit max_tokens** count; **empty responses** count | always |
| `[latency]` | mean / p50 / p95 / p99 / max for TTFT, total time, inter-token latency (ms per output token after the first, what streaming feels like) and per-request generation tok/s | always |
| `[errors]` | count per distinct error message, including vLLM's reason | any failures |
| `[per prompt]` | one row per prompt: count, ok, fail, avg prompt and completion tokens, TTFT mean / p95, total mean / p95, inter-token latency, tok/s, hit max_tokens | `--mix` |
| `[timeline]` | one row per bucket: started, completed, failed, in flight at the end of the bucket, completion tokens and tok/s, TTFT mean / p95, mean total time. Shows warm-up, drift and the point where things degrade. | `--duration`, or count runs longer than two buckets |
| `[gpu]` | `nvidia-smi` sampled across the whole batch: avg / peak utilisation, peak memory per GPU | GPU available |
| `[per request]` | one row per request: start and end relative to the batch (`*` = finished after the deadline), TTFT, total, inter-token ms, tokens, tok/s, finish reason, prompt name. Failed rows show the error, unfinished rows are marked. | always |
| `[failed requests]` | the failed rows on their own: number, start, time of failure, prompt, error | any failures |
| `[prompt]` | file name, word count and the first 30 words of each prompt used | always |
| `[reasoning]` / `[response]` | the first successful response of the last batch, or all of them with `--all-responses`. An empty response carries a note saying why. | always |

With `--runs > 1` the block starts with the averaged summary, followed by one `--- batch N ---`
section per run with its own tables.

Example, `--mix --concurrency 8 --duration 120`:

```
========================================================================
2026-09-23 10:12:07  |  model: openai/gpt-oss-120b  |  prompt: mixed  |  concurrency: 8  |  duration: 120.0s  |  requests: 96  |  runs: 1
max_tokens: 4000  |  temperature: 0.3  |  reasoning_effort: medium  |  url: http://localhost:8000/v1  |  started: 2026-09-23 10:10:01  |  ended: 2026-09-23 10:12:07
========================================================================

[load]  closed loop: a fixed number of requests in flight
  concurrency          : 8   (achieved avg in flight 7.91)
  duration             : 120.0 s   (no new requests after this; in-flight ones finish)
  requests             : 96  (ok 95)
  failed               : 1  (1.0 % of requests)
  wall time            : 126.4 s
  completed/s          : 0.75

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
  empty responses      : 0 requests   (no answer text; e.g. all tokens spent on reasoning)

[latency]  per request         mean / p50 / p95 / p99 / max
  time to first token  : 0.412 / 0.390 / 0.780 / 0.880 / 0.910 s
  total time           : 10.1 / 9.8 / 12.4 / 12.9 / 13.0 s
  inter-token latency  : 9.3 / 9.2 / 10.1 / 10.6 / 10.8 ms   (time per output token after the first)
  generation tok/s     : 104.2 / 103.0 / 118.7 / 120.4 / 121.0   (per request, drops as concurrency rises)

[errors]
     1 x HTTPError('400 from http://localhost:8000/v1/chat/completions: This model's maximum context length is ...')

[per prompt]   n    ok  fail   prompt_tok   comp_tok   ttft_mean   ttft_p95   total_mean   total_p95   itl_ms   gen_tok_s   hit_max   prompt
            24    24     0          402       1102       0.418      0.790        10.6        12.6      9.4      104.0         0   krisala-lead-suggestion-1
            24    23     1          351        990       0.405      0.771         9.6        12.1      9.3      104.5         0   krisala-lead-suggestion-2
   ...

[timeline]  per 10 s bucket — watch for tok/s dropping or TTFT climbing over the run
       t0       t1   started   completed   failed   in_flight   comp_tok   comp_tok/s   ttft_mean   ttft_p95   total_mean
        0       10         8           0        0           8          0          0.0           -          -            -
       10       20         8           8        0           8       8412        841.2       0.421      0.780       10.2
   ...
      120      126         0           8        0           0       8360       1306.2       0.402      0.744       10.0

[gpu]  sampled every 0.25s across the whole batch with nvidia-smi
  gpu 0: util avg 97%  max 100%  |  mem peak 76819/81920 MiB  |  505 samples

[per request]   #   start_s    end_s   ttft_s   total_s   itl_ms   prompt_tok   comp_tok   gen_tok_s   finish   prompt   (* = finished after the deadline)
    1     0.000    9.812    0.390     9.812      9.1          377       1040       107.9     stop   krisala-lead-suggestion-1
  ...
   96   119.850  126.401*   0.402     6.551      9.0          352        720       109.9     stop   krisala-lead-suggestion-4

[failed requests]  1 of 96     #   start_s   failed_at_s   prompt   error
   14    18.201        18.240   krisala-lead-suggestion-2   HTTPError('400 from ...')

[prompt]  file: prompts/krisala-lead-suggestion-1.txt  (7025 words, first 30 shown)
...

[response]  request 1 of 96 (others omitted; use --all-responses)
...
```

---

## 5. Reasoning models: Qwen3, gpt-oss, DeepSeek

These models stream their reasoning before the answer. vLLM sends it as `reasoning_content` (Qwen3,
DeepSeek) or `reasoning` (gpt-oss on newer vLLM); both are read. The first reasoning token counts as
TTFT. The reasoning is written to the log under `[reasoning]` and printed with `--show-reasoning`.
vLLM's completion token count includes the reasoning tokens.

Controls:

| Flag | Models | Effect |
|---|---|---|
| `--thinking off` or `on` | Qwen3 and others with a thinking switch | disables or forces the thinking phase (sends `enable_thinking`). Off means much lower TTFT and far fewer completion tokens. |
| `--reasoning-effort low`, `medium` or `high` | gpt-oss and others with an effort level | sends `reasoning_effort`. `low` leaves most of `--max-tokens` for the answer. |

Both are recorded in the log header. Models that do not have the setting ignore it.

**Empty response?** The log says why next to `[response]`. The usual cause is `finish reason: length`:
the model used the whole `--max-tokens` budget on reasoning and was cut off before the answer. Raise
`--max-tokens`, lower the effort, or turn thinking off. In the load test, `hit max_tokens` and
`empty responses` under `[throughput]` tell you how many requests were affected. If tokens were
generated but neither field carried them, the note lists the stream fields that were seen, which is
what you need to add support for a new model.

---

## 6. Shared flags

| Flag | Default | Meaning |
|---|---|---|
| `--url` | `$VLLM_URL` or `http://localhost:8000/v1` | vLLM base URL |
| `--api-key` | `$VLLM_API_KEY` | sent as a Bearer token |
| `--model` | `$VLLM_MODEL`, else the first model the server reports | model id |
| `--list-models` | | print the server's models and exit |
| `--pick-model` | | numbered menu to choose a model |
| `--prompt "text"` | | inline prompt (logged to `logs/inline.txt`) |
| `--prompt-file X` | all of `prompts/` | prompt name, filename or path; repeatable |
| `--prompt-dir` | `prompts` | where prompts live |
| `--system "text"` | | optional system prompt |
| `--max-tokens` | `512` | output budget per request, reasoning included |
| `--temperature` | `0.0` | sampling temperature |
| `--runs N` | `1` | repeat and average |
| `--thinking on` or `off` | model default | see section 5 |
| `--reasoning-effort low`, `medium` or `high` | model default | see section 5 |
| `--log path` | `logs/<prompt>.txt` | one log file for everything in this run |
| `--log-dir` | `logs` | where logs go |
| `--log-words N` | `30` | how many words of each prompt to copy into the log |
| `--show-reasoning` | | print the reasoning in the terminal too |
| `--quiet` | | do not print responses in the terminal (they still go to the log) |
| `--no-gpu` | | skip `nvidia-smi` sampling |
| `--gpu-interval S` | `0.25` | seconds between `nvidia-smi` samples |

---

## 7. GPU numbers

`nvidia-smi` is polled every `--gpu-interval` seconds while requests are running: per request in
`bench_vllm.py`, across the whole batch in the load test. Utilisation is the average and peak of those
samples, memory is the peak `memory.used`. Memory looks almost constant because vLLM pre-allocates its
KV cache at startup; the number that matters is utilisation. For very short requests you get only a
few samples, so use `--gpu-interval 0.1`.

## 8. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `prompt file not found` | the name was looked up in the current dir, the script's folder and `prompts/`. Check the spelling; `.txt` is added automatically. |
| `[response]` is empty, `finish reason: length` | all of `--max-tokens` went into reasoning. Raise it or lower the effort (section 5). |
| `failed` > 0 with `400 ... maximum context length` | prompt plus `--max-tokens` exceeds the model's context window. Shorten the prompt or lower `--max-tokens`. |
| `failed` > 0 with connection errors or timeouts | the server is overloaded or restarted. Look at `[timeline]` to see when it started, and at the vLLM log. Each request has a 600 s timeout. |
| `unfinished` > 0 | `--drain` expired while requests were still running. They are not counted in the latency stats. Raise `--drain` or lower the load. |
| TTFT is `-` | no token was ever received. See the note next to `[response]` for which stream fields arrived. |
| very few `[gpu]` samples | the request was shorter than the sampling interval. Use `--gpu-interval 0.1`. |
