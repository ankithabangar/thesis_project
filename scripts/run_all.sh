#!/usr/bin/env bash
# Drive the baseline + self-reflection grid for the thesis experiments.

set -uo pipefail

cd "$(dirname "$0")/.."

MODELS=("qwen" "gemini" "grok")
RUNS=(1 2 3)

mkdir -p logs

for model in "${MODELS[@]}"; do
  for run in "${RUNS[@]}"; do
    echo "=== ${model} run${run} ===" | tee -a logs/master.log

    echo "--- baseline ---" | tee -a logs/master.log
    python src/baseline.py --model "$model" --run "$run" \
      2>&1 | tee -a "logs/${model}_run${run}_baseline.log"

    echo "--- self_reflection ---" | tee -a logs/master.log
    python src/self_reflection.py --model "$model" --run "$run" \
      2>&1 | tee -a "logs/${model}_run${run}_reflection.log"
  done
done

echo "=== DONE ===" | tee -a logs/master.log
