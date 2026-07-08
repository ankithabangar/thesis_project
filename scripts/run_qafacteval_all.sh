#!/usr/bin/env bash
# Add QAFactEval scores into all 18 existing evaluation files.

set -uo pipefail

cd "$(dirname "$0")/.."

IMAGE="thesis-evaluate"

if ! docker info > /dev/null 2>&1; then
  echo "ERROR: docker daemon not running. Start Docker Desktop." >&2
  exit 1
fi
if ! docker image inspect "$IMAGE" > /dev/null 2>&1; then
  echo "ERROR: image '$IMAGE' not found. Build it with:" >&2
  echo "  docker build --platform linux/amd64 -t $IMAGE ." >&2
  exit 1
fi

DOCKER_ARGS=(
  --platform linux/amd64 --rm
  -e PYTHONUNBUFFERED=1
  -e TOKENIZERS_PARALLELISM=false
  -e REQUESTS_CA_BUNDLE=""
  -e CURL_CA_BUNDLE=""
  -e HF_HUB_OFFLINE=1
  -e TRANSFORMERS_OFFLINE=1
  -e BART_MODEL_PATH=/root/.cache/huggingface/models--facebook--bart-large/snapshots/cb48c1365bd826bd521f650dc2e0940aee54720c
  -v "$(pwd)/src:/app/src"
  -v "$(pwd)/data:/app/data"
  -v "$(pwd)/results:/app/results"
  -v "$(pwd)/ckpts:/app/ckpts"
  -v "$(pwd)/qafacteval_models:/app/qafacteval_models"
  -v "$(pwd)/hf_cache:/root/.cache/huggingface"
)

MODELS=("grok" "o4mini" "gemini")
RUNS=(1 2 3)
STAGES=("baseline" "reflection")

mkdir -p logs

for model in "${MODELS[@]}"; do
  for run in "${RUNS[@]}"; do
    for stage in "${STAGES[@]}"; do
      tag="${model}_run${run}_${stage}_qafe"
      echo "=== $tag ===" | tee -a logs/master_qafe.log

      docker run "${DOCKER_ARGS[@]}" "$IMAGE" \
        python -u src/evaluate_qafacteval.py \
          --model "$model" --run "$run" --stage "$stage" \
        2>&1 | tee -a "logs/${tag}.log"
    done
  done
done

echo "=== DONE ===" | tee -a logs/master_qafe.log
