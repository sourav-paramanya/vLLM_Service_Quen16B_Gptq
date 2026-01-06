#!/bin/bash
set -e
MODEL_NAME="hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4"
PORT="8080"

# You can add --token <HF_TOKEN> if private model
exec python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_NAME" \
    --host 0.0.0.0 \
    --port $PORT \
    --tensor-parallel-size 1
