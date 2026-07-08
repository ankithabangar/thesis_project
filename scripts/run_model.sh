#!/usr/bin/env bash
# Run all 3 (baseline + self_reflection) runs for ONE model.

set -uo pipefail

if [ $# -ne 1 ]; then
  echo "Usage: $0 <model>     (one of: qwen, gemini, grok, o4mini)" >&2
  exit 1
fi
model="$1"

cd "$(dirname "$0")/.."
mkdir -p logs

for run in 1 2 3; do
  echo "[$model] === run${run} ===" | tee -a logs/master.log

  echo "[$model] --- baseline ---" | tee -a logs/master.log
  python src/baseline.py --model "$model" --run "$run" \
    2>&1 | tee -a "logs/${model}_run${run}_baseline.log"

  echo "[$model] --- self_reflection ---" | tee -a logs/master.log
  python src/self_reflection.py --model "$model" --run "$run" \
    2>&1 | tee -a "logs/${model}_run${run}_reflection.log"
done

echo "[$model] === DONE ===" | tee -a logs/master.log
