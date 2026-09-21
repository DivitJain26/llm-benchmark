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