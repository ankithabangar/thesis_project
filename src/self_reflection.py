"""Self-reflection (critic -> refiner) on baseline summaries for one
(model, run) combo of the thesis grid.

Reads  results/{model}/run{run}/baseline_summaries.json
Writes results/{model}/run{run}/reflection_summaries.json

One iteration = one critic call + one refiner call (MAX_ITERATIONS=1).
"""
import argparse
import json
import time
from pathlib import Path

from models import SUPPORTED_MODELS, get_model, invoke
from prompts import CRITIC_PROMPT, REFINER_PROMPT


MAX_ITERATIONS = 1
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=SUPPORTED_MODELS)
    p.add_argument("--run", required=True, type=int, choices=(1, 2, 3))
    return p.parse_args()


def reflect(critic_chain, refiner_chain, summary: str, transcript: str) -> dict:
    iterations = []
    current = summary

    for i in range(1, MAX_ITERATIONS + 1):
        print(f"\n  -- Iteration {i}/{MAX_ITERATIONS} --")

        t0 = time.time()
        print("    [Critic] running...")
        feedback = invoke(
            critic_chain, {"transcript": transcript, "summary": current}
        )
        print(
            f"    [Critic] {time.time() - t0:.1f}s "
            f"-- feedback {len(feedback.split())} words"
        )

        t0 = time.time()
        print("    [Refiner] running...")
        refined = invoke(
            refiner_chain,
            {"transcript": transcript, "summary": current, "feedback": feedback},
        )
        print(
            f"    [Refiner] {time.time() - t0:.1f}s "
            f"-- new summary {len(refined.split())} words"
        )

        iterations.append(
            {
                "iteration": i,
                "summary_before": current,
                "feedback": feedback,
                "summary_after": refined,
            }
        )
        current = refined

    return {
        "final_summary": current,
        "iterations": iterations,
        "total_iterations": MAX_ITERATIONS,
    }


def main() -> None:
    args = parse_args()

    baseline_path = (
        PROJECT_ROOT / f"results/{args.model}/run{args.run}/baseline_summaries.json"
    )
    output_path = (
        PROJECT_ROOT / f"results/{args.model}/run{args.run}/reflection_summaries.json"
    )

    if not baseline_path.exists():
        raise SystemExit(
            f"baseline file not found: {baseline_path}\n"
            f"Run baseline.py first for this (model, run)."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    baseline_data = json.loads(baseline_path.read_text(encoding="utf-8"))
    results = (
        json.loads(output_path.read_text(encoding="utf-8"))
        if output_path.exists()
        else {}
    )

    print(f"Model: {args.model}  Run: {args.run}  Seed: {args.run}")
    print(
        f"Loaded {len(baseline_data)} baseline summaries; "
        f"{len(results)} already reflected."
    )

    model = get_model(args.model, seed=args.run)
    critic_chain = CRITIC_PROMPT | model
    refiner_chain = REFINER_PROMPT | model

    for meeting_id, meeting in baseline_data.items():
        if meeting_id in results:
            print(f"  [skip] {meeting_id}")
            continue

        print(f"\n{'=' * 60}\nProcessing {meeting_id}\n{'=' * 60}")
        start = time.time()
        out = reflect(
            critic_chain,
            refiner_chain,
            summary=meeting["summary"],
            transcript=meeting["transcript"],
        )
        elapsed = time.time() - start
        print(f"  Total {elapsed:.1f}s ({elapsed / 60:.1f} min)")

        results[meeting_id] = {
            "meeting_id": meeting_id,
            "domain": meeting["domain"],
            "transcript": meeting["transcript"],
            "ground_truth": meeting["ground_truth"],
            "baseline_summary": meeting["summary"],
            "summary": out["final_summary"],
            "self_refine": {
                "total_iterations": out["total_iterations"],
                "iterations": out["iterations"],
            },
        }
        output_path.write_text(
            json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    print(f"\nSaved {len(results)} reflected summaries to {output_path}")


if __name__ == "__main__":
    main()
