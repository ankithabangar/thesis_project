#!/usr/bin/env bash
# Run QAFactEval + AlignScore + MiniCheck-FT5 across the full thesis grid.
#
# 3 models x 3 runs x 2 stages x 20 meetings = 360 evaluations.
#
# Eval lives inside Docker because QAFactEval / AlignScore have legacy
# torch / spacy / allennlp pins that don't run on Apple Silicon Python 3.11.
#
# Each (model, run, stage) call loads the 3 metric models once 
# and then scores 20 meetings serially. Resumable per-combo: the
# Python script skips meetings already in its evaluation output file.

set -uo pipefail
# NOT set -e: one bad combo should not kill the rest of the grid.

cd "$(dirname "$0")/.."

IMAGE="thesis-evaluate"

# Verify Docker is up and the image exists.
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
  # Note: do NOT throttle threads with OMP_NUM_THREADS / MKL_NUM_THREADS here.
  # Containers run sequentially in this script, so each one should use all
  # available cores. Limiting to 1 thread makes evaluation ~10x slower.
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

MODELS=("gemini" "grok" "o4mini")
RUNS=(1 2 3)
STAGES=("baseline" "reflection")

mkdir -p logs

for model in "${MODELS[@]}"; do
  for run in "${RUNS[@]}"; do
    for stage in "${STAGES[@]}"; do
      tag="${model}_run${run}_${stage}_eval"
      echo "=== $tag ===" | tee -a logs/master_eval.log

      docker run "${DOCKER_ARGS[@]}" "$IMAGE" \
        python -u src/evaluate.py --model "$model" --run "$run" --stage "$stage" \
        2>&1 | tee -a "logs/${tag}.log"
    done
  done
done

echo "=== DONE ===" | tee -a logs/master_eval.log
