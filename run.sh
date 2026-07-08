#!/usr/bin/env bash
set -euo pipefail

IMAGE="thesis-evaluate"

echo "Building Docker image..."
docker build --platform linux/amd64 -t "$IMAGE" .

mkdir -p ckpts qafacteval_models hf_cache

echo "Running baseline evaluation..."
docker run --platform linux/amd64 --rm \
    -e PYTHONUNBUFFERED=1 \
    -e TOKENIZERS_PARALLELISM=false \
    -e REQUESTS_CA_BUNDLE="" \
    -e CURL_CA_BUNDLE="" \
    -e TRANSFORMERS_OFFLINE=1 \
    -e HF_HUB_DISABLE_XET=1 \
    -e BART_MODEL_PATH=/root/.cache/huggingface/models--facebook--bart-large/snapshots/cb48c1365bd826bd521f650dc2e0940aee54720c \
    -v "$(pwd)/qwen3:/app/qwen3" \
    -v "$(pwd)/ckpts:/app/ckpts" \
    -v "$(pwd)/qafacteval_models:/app/qafacteval_models" \
    -v "$(pwd)/hf_cache:/root/.cache/huggingface" \
    "$IMAGE" python -u qwen3/sanity_check_evaluators.py

echo "Done. Results saved"