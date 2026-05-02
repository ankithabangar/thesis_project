#!/usr/bin/env bash
set -euo pipefail

IMAGE="thesis-evaluate"
RESULTS_FILE="$(pwd)/baseline_evaluation_results.json"

# Ensure output file exists so Docker can bind-mount it as a file (not dir)
touch "$RESULTS_FILE"

echo "Building Docker image..."
docker build --platform linux/amd64 -t "$IMAGE" .

echo "Running evaluation..."
docker run --platform linux/amd64 --rm \
    -v "$(pwd)/data:/app/data:ro" \
    -v "$RESULTS_FILE:/app/baseline_evaluation_results.json" \
    -v "$(pwd)/ckpts:/app/ckpts" \
    -v "$(pwd)/qafacteval_models:/app/qafacteval_models" \
    -v "$(pwd)/hf_cache:/root/.cache/huggingface" \
    "$IMAGE"

echo "Done. Results saved to $RESULTS_FILE"
