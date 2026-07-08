"""Backfill QAFactEval scores into existing evaluation files.

For one (model, run, stage) combo:
  1. Reads  results/{model}/run{run}/{stage}_summaries.json  (source + summary)
  2. Reads  results/{model}/run{run}/{stage}_evaluation.json (existing scores)
  3. For each meeting that has an entry but no 'qafacteval_score' field,
     runs QAFactEval and merges the score into that entry.
  4. Writes after every meeting so a crash never loses prior work.

Reuses load_qafacteval / score_qafacteval from evaluate.py to keep the
truncation config in one place.
"""
import argparse
import json
import time
from pathlib import Path

from evaluate import (
    SUPPORTED_MODELS,
    SUPPORTED_STAGES,
    load_qafacteval,
    score_qafacteval,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=SUPPORTED_MODELS)
    p.add_argument("--run", required=True, type=int, choices=(1, 2, 3))
    p.add_argument("--stage", required=True, choices=SUPPORTED_STAGES)
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only score the first N missing meetings (for smoke tests).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    summaries_path = (
        PROJECT_ROOT
        / f"results/{args.model}/run{args.run}/{args.stage}_summaries.json"
    )
    eval_path = (
        PROJECT_ROOT
        / f"results/{args.model}/run{args.run}/{args.stage}_evaluation.json"
    )

    if not summaries_path.exists():
        raise SystemExit(f"summaries file not found: {summaries_path}")
    if not eval_path.exists():
        raise SystemExit(
            f"eval file not found: {eval_path}\n"
            f"Run evaluate.py first to create it, then backfill QAFactEval here."
        )

    summaries = json.loads(summaries_path.read_text(encoding="utf-8"))
    results = json.loads(eval_path.read_text(encoding="utf-8"))

    todo = [
        (mid, m)
        for mid, m in summaries.items()
        if mid in results and "qafacteval_score" not in results[mid]
    ]
    if args.limit is not None:
        todo = todo[: args.limit]
        print(f"  --limit {args.limit} applied: only first {len(todo)} will run.")

    print("=" * 72)
    print(f"  QAFactEval backfill  ({args.model} run{args.run} {args.stage})")
    print(f"  Total meetings in eval file: {len(results)}")
    print(f"  Missing qafacteval_score:    {len(todo)}")
    print("=" * 72)

    if not todo:
        print("Nothing to do -- every meeting already has qafacteval_score.")
        return

    qafacteval = load_qafacteval()

    for i, (mid, meeting) in enumerate(todo, start=1):
        summary = meeting["summary"]
        source = meeting["transcript"]
        n_words = len(summary.split())
        print(f"\n[{i}/{len(todo)}] {mid}  ({n_words} words)")

        t0 = time.time()
        qa = score_qafacteval(qafacteval, summary, source)
        trunc_flag = " [trunc]" if qa["qafacteval_source_truncated"] else ""
        print(
            f"  QAFactEval:  {qa['qafacteval_score']:.4f}{trunc_flag}  "
            f"({time.time() - t0:.1f}s)"
        )

        # Merge into the existing per-meeting entry (do not touch other fields).
        results[mid].update(qa)
        eval_path.write_text(
            json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    print(f"\nBackfilled {len(todo)} QAFactEval scores into {eval_path}")


if __name__ == "__main__":
    main()
