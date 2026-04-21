import json
import time
import re
from pathlib import Path

from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate


BASELINE_PATH = Path("baseline_summaries.json")
OUTPUT_PATH = Path("self_refined_summaries.json")

MAX_ITERATIONS = 1

model = OllamaLLM(model="qwen3:30b")

critic_prompt = ChatPromptTemplate.from_template("""
You are an expert meeting summary critic.
Your task is to evaluate the factual accuracy of a meeting summary by comparing it against the original transcript.

TRANSCRIPT:
{transcript}

SUMMARY TO EVALUATE:
{summary}

Instructions:
1. Read the transcript carefully.
2. Check EACH sentence of the summary against the transcript.

Be strict. Only flag genuinely unsupported or incorrect information. Minor paraphrasing differences are acceptable.
""")

refiner_prompt = ChatPromptTemplate.from_template("""
You are an expert meeting summariser.
Your task is to revise a meeting summary based on feedback and the original transcript.

TRANSCRIPT:
{transcript}

CURRENT SUMMARY:
{summary}

FEEDBACK FROM REVIEWER:
{feedback}

Instructions:
1. Read the feedback carefully.
2. Replace unsupported claims with information that IS present in the transcript.
3. Do not change correct information.
4. Maintain the same overall structure and length of the summary.
5. Do not add new information that is not in the transcript.
6. Do not add any preamble, explanation, or notes — output ONLY the revised summary.

Important: Only include information that is explicitly stated in the transcript.
Do not add, infer, or assume any details that are not directly mentioned.

Write only the revised summary as a single paragraph of no more than 150 words.
""")

critic_chain = critic_prompt | model
refiner_chain = refiner_prompt | model


def self_refine(
    initial_summary: str,
    transcript: str,
    max_iterations: int = MAX_ITERATIONS,
) -> dict:
    current_summary = initial_summary
    iterations = []

    print(f"  Starting Self-Refine loop (max {max_iterations} iterations)")

    for i in range(1, max_iterations + 1):
        print(f"\n  ── Iteration {i}/{max_iterations} ──")

        # Critic step
        critic_start = time.time()
        print("    [Critic] Evaluating summary against transcript...")
        feedback = critic_chain.invoke({"transcript": transcript, "summary": current_summary}).strip()

        # Refiner step
        refiner_start = time.time()
        print(f"    [Refiner] Revising summary based on feedback...")
        refined_summary = refiner_chain.invoke(
            {
                "transcript": transcript,
                "summary": current_summary,
                "feedback": feedback,
            }
        ).strip()
        print(
            f"    [Refiner] Done in {time.time() - refiner_start:.1f}s — new summary: {len(refined_summary.split())} words"
        )

        iterations.append(
            {
                "iteration": i,
                "summary_before": current_summary,
                "feedback": feedback,
                "summary_after": refined_summary,
                "action": "refined",
            }
        )
        current_summary = refined_summary

    print(f"    [Stop] Reached max iterations ({max_iterations}).")
    return {
        "final_summary": current_summary,
        "iterations": iterations,
        "total_iterations": max_iterations,
        "stopped_reason": "max_iterations",
    }


def main():
    with open(BASELINE_PATH, "r", encoding="utf-8") as f:
        baseline_data = json.load(f)

    print(f"Loaded {len(baseline_data)} baseline summaries from {BASELINE_PATH}")
    print(f"Self-Refine config: max_iterations={MAX_ITERATIONS}")

    results = {}

    for meeting_id, meeting in baseline_data.items():
        print(f"\n{'=' * 60}")
        print(f"Processing {meeting_id}")
        print(f"{'=' * 60}")

        initial_summary = meeting["summary"]
        transcript = meeting["transcript"]

        print(f"  Baseline summary: {len(initial_summary.split())} words")

        start_time = time.time()
        refine_result = self_refine(initial_summary=initial_summary, transcript=transcript)
        elapsed = time.time() - start_time

        print(f"\n  Total time: {elapsed:.1f}s ({elapsed / 60:.1f} min)")
        print(f"  Iterations used: {refine_result['total_iterations']}")
        print(f"  Stopped because: {refine_result['stopped_reason']}")
        print(f"  Final summary: {len(refine_result['final_summary'].split())} words")

        results[meeting_id] = {
            "meeting_id": meeting_id,
            "domain": meeting["domain"],
            "transcript": meeting["transcript"],
            "ground_truth": meeting["ground_truth"],
            "baseline_summary": initial_summary,
            "summary": refine_result["final_summary"],
            "self_refine": {
                "total_iterations": refine_result["total_iterations"],
                "stopped_reason": refine_result["stopped_reason"],
                "iterations": refine_result["iterations"],
            },
        }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(results)} self-refined summaries to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()