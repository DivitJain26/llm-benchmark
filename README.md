# vLLM request benchmark

A small script that sends one prompt to a running vLLM server and reports:

- time to first token (TTFT)
- total request time
- prompt / completion / total token counts (exact, from vLLM's `usage` field)
- generation tokens/sec and overall tokens/sec

**Where the output goes:** the response and metrics are printed in the terminal
**and** appended, together with the prompt, to a single `.txt` log file
(`bench_log.txt` by default). Every run adds a new block, so one file holds your
whole history.

## Files

| File | Purpose |
|---|---|
| `bench_vllm.py` | the benchmark script |
| `setup.sh` | one-time setup (venv + `requests` + sample `prompt.txt`) |
| `prompt.txt` | your prompt (created by `setup.sh`, edit freely) |
| `bench_log.txt` | the log, created on first run |

## Setup

Your vLLM server must already be running, e.g. `vllm serve Qwen/Qwen3-14B --api-key $VLLM_API_KEY`.

```bash
bash setup.sh
```

Inside the vllm container (`/vllm-workspace`) it uses the system Python; elsewhere it makes a `.venv`
(then `source .venv/bin/activate`).

Set your server details once per shell (all optional except the key if your server requires one):

```bash
export VLLM_API_KEY=your-key            # sent as "Authorization: Bearer ..."
export VLLM_MODEL=Qwen/Qwen3-14B        # else the first model from the server is used
export VLLM_URL=http://localhost:8000/v1
```

## Run

```bash
python3 bench_vllm.py --prompt-file prompt.txt
python3 bench_vllm.py --prompt "What is a LLM?" --max-tokens 500 --temperature 0.3
```

Equivalent of your curl: model, max_tokens and temperature can also be passed as flags
(`--model Qwen/Qwen3-14B --max-tokens 500 --temperature 0.3`), and `--api-key` overrides the env var.

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
| `--prompt` | – | prompt text inline |
| `--prompt-file` | – | read prompt from a file |
| `--system` | – | optional system prompt |
| `--log` | `bench_log.txt` | log file to append to |
| `--max-tokens` | `512` | max output tokens |
| `--temperature` | `0.0` | sampling temperature |
| `--runs` | `1` | repeat N times and print the average |
| `--show-reasoning` | off | print Qwen3 reasoning in the terminal too |
| `--quiet` | off | don't print the response in the terminal (still goes to the log) |

## Examples

```bash
# inline prompt
python bench_vllm.py --prompt "Explain quicksort in 3 lines"

# 5 runs, averaged
python bench_vllm.py --prompt-file prompt.txt --runs 5

# remote server, longer output, separate log
python bench_vllm.py --url http://10.0.0.5:8000/v1 --max-tokens 1024 \
  --prompt-file prompt.txt --log remote_bench.txt

# system prompt, response only in the log
python bench_vllm.py --prompt-file prompt.txt --system "Be concise." --quiet
```

## Log format

```
========================================================================
2026-09-18 06:03:44  |  model: Qwen/Qwen2.5-7B-Instruct  |  runs: 3
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

[prompt]
...

[reasoning]
...

[response]
...
```

`[per run]` appears only when `--runs` > 1; `[system]` only with `--system`; `[reasoning]` only when the model emits it.# llm-benchmark
