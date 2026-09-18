#!/usr/bin/env bash
# Setup for bench_vllm.py — works inside the vllm container (/vllm-workspace) or anywhere else.
set -e

# only make a venv if we're NOT already in a container that has vllm installed
if python3 -c "import vllm" 2>/dev/null; then
  echo "vllm environment detected, using system python"
else
  python3 -m venv .venv
  source .venv/bin/activate
  pip install --upgrade pip
fi
python3 -c "import requests" 2>/dev/null || pip install requests

# prompts/ and logs/ are created automatically on first run; add a sample prompt now
mkdir -p prompts logs
[[ -e prompts/example.txt ]] || echo "What is a LLM? Answer in two short paragraphs." > prompts/example.txt

echo
echo "Setup done. Set your server details once:"
echo "  export VLLM_API_KEY=your-key          # same key you pass to vllm serve --api-key"
echo "  export VLLM_MODEL=Qwen/Qwen3-14B      # optional, else picked from server"
echo "  export VLLM_URL=http://localhost:8000/v1   # optional, this is the default"
echo
echo "Then run:"
echo "  python3 bench_vllm.py                      # all prompts in prompts/ -> logs/"
echo "  python3 bench_vllm.py --prompt-file example --runs 3"