# Baseline summarisation for one (model, run) combo of the thesis grid.
# Reads  data/cleaned_meetings.json
# Writes results/{model}/run{run}/baseline_summaries.json

import argparse
import json
import time
from pathlib import Path

from models import SUPPORTED_MODELS, get_model, invoke
from prompts import SUMMARY_PROMPT


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=SUPPORTED_MODELS)
    p.add_argument("--run", required=True, type=int, choices=(1, 2, 3))
    return p.parse_args()


def summarise(chain, meeting_id: str, meeting: dict) -> dict:
    print(f"\n{'=' * 60}\nProcessing {meeting_id}\n{'=' * 60}")
    start = time.time()
    summary = invoke(chain, {"transcript": meeting["transcript"]})
    elapsed = time.time() - start
    print(f"  Done in {elapsed:.1f}s — {len(summary.split())} words")
    return {
        "meeting_id": meeting_id,
        "domain": meeting["domain"],
        "transcript": meeting["transcript"],
        "ground_truth": meeting["ground_truth"],
        "summary": summary,
    }


def main() -> None:
    args = parse_args()

    input_path = PROJECT_ROOT / "data/cleaned_meetings.json"
    output_path = (
        PROJECT_ROOT / f"results/{args.model}/run{args.run}/baseline_summaries.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    meetings = json.loads(input_path.read_text(encoding="utf-8"))
    results = (
        json.loads(output_path.read_text(encoding="utf-8"))
        if output_path.exists()
        else {}
    )

    print(f"Model: {args.model}  Run: {args.run}  Seed: {args.run}")
    print(f"Loaded {len(meetings)} meetings; {len(results)} already done.")

    model = get_model(args.model, seed=args.run)
    chain = SUMMARY_PROMPT | model

    for meeting_id, meeting in meetings.items():
        if meeting_id in results:
            print(f"  [skip] {meeting_id}")
            continue
        results[meeting_id] = summarise(chain, meeting_id, meeting)
        output_path.write_text(
            json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    print(f"\nSaved {len(results)} summaries to {output_path}")


if __name__ == "__main__":
    main()
