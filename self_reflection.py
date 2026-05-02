import json
import time
import re
from pathlib import Path

from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate


BASELINE_PATH = Path("./test_dir/baseline_summaries.json")
CLEANED_PATH = Path("./test_dir/cleaned_meetings.json")
OUTPUT_PATH = Path("./test_dir/loop_3/self_refined_summaries.json")

MAX_ITERATIONS = 3

model = OllamaLLM(model="llama3.1:8b")

critic_prompt = ChatPromptTemplate.from_template("""
You are a meeting summary fact-checker. Compare the summary against the transcript section and find ONLY factual errors.

TRANSCRIPT SECTION:
{chunk}

SUMMARY OF THIS SECTION:
{chunk_summary}

Flag a sentence ONLY if it:
- States something that NEVER happened in this transcript section (fabrication)
- Gets a fact wrong — wrong number, wrong person, wrong decision (error)
- Describes something that means the OPPOSITE of what was said (contradiction)

Do NOT flag a sentence if:
- The information IS in this transcript section but worded differently (paraphrase is fine)
- The sentence summarises or simplifies a longer discussion (that is expected)
- The sentence omits some details (omission is not an error)
- You are unsure — if in doubt, do NOT flag it

For each real error found:
ISSUE [N]: [Copy the problematic sentence]
REASON: [One sentence: what specific fact is wrong and what the transcript actually says]

If every sentence is factually supported by this transcript section, respond with exactly: NO_ISSUES
""")

refiner_prompt = ChatPromptTemplate.from_template("""
You are a meeting summary editor. Fix ONLY the errors identified in the feedback. Keep everything else exactly as it is.

TRANSCRIPT SECTION:
{chunk}

CURRENT SUMMARY OF THIS SECTION:
{chunk_summary}

ERRORS TO FIX:
{chunk_feedback}

Rules:
1. For each ISSUE in the feedback, either correct the fact using the transcript section or remove the sentence.
2. DO NOT change, weaken, or rephrase any sentence that was NOT flagged as an issue.
3. DO NOT add hedging words like "possibly", "a possible", "approximately", "it was mentioned that".
4. State facts directly and confidently when they are supported by the transcript.
5. DO NOT add new information beyond what is needed to fix the flagged issues.
6. Keep the summary concise — no more than 80 words.

Output ONLY the revised summary. No preamble, no explanation.
""")

merge_prompt = ChatPromptTemplate.from_template("""
You are an expert meeting summariser.
Below are summaries of consecutive sections of a single meeting transcript.
Combine them into one concise, accurate, and coherent summary in plain prose
(no bullet points, no headers).

Important: Only include information that is present in the section summaries.
Do not add, infer, or assume any details that are not directly mentioned.
Remove any redundancies across sections but do not drop important information.
Do not refer to speakers by name or role. Focus on the content of the discussion, not who said what.

Write only the summary as a single paragraph of no more than 150 words. Do not include any preamble or explanation.

SECTION SUMMARIES:
{combined_summaries}
""")

critic_chain = critic_prompt | model
refiner_chain = refiner_prompt | model
merge_chain = merge_prompt | model


def count_issues(critic_response: str) -> int:
    if re.fullmatch(r"NO_ISSUES[.!]?", critic_response.strip().upper()):
        return 0
    issues = re.findall(r"ISSUE\s*\[?\d+\]?\s*:", critic_response, re.IGNORECASE)
    return max(len(issues), 1) if issues else 0


def critique_chunk(chunk: str, chunk_summary: str) -> tuple[str, int]:
    chunk_feedback = critic_chain.invoke(
        {"chunk": chunk, "chunk_summary": chunk_summary}
    ).strip()
    num_issues = count_issues(chunk_feedback)
    return chunk_feedback, num_issues


def refine_chunk(chunk: str, chunk_summary: str, chunk_feedback: str) -> str:
    refined = refiner_chain.invoke(
        {
            "chunk": chunk,
            "chunk_summary": chunk_summary,
            "chunk_feedback": chunk_feedback,
        }
    ).strip()
    return refined


def merge_summaries(chunk_summaries: list[str]) -> str:
    combined = "\n\n".join(
        f"[Section {i}] {s}" for i, s in enumerate(chunk_summaries, start=1)
    )
    return merge_chain.invoke({"combined_summaries": combined}).strip()


def self_refine(
    chunk_summaries: list[str],
    chunks: list[str],
    max_iterations: int = MAX_ITERATIONS,
) -> dict:
    current_chunk_summaries = list(chunk_summaries)
    num_chunks = len(chunks)
    iterations = []

    print(
        f"  Starting Self-Refine loop (max {max_iterations} iterations, {num_chunks} chunks)"
    )

    for i in range(1, max_iterations + 1):
        print(f"\n  ── Iteration {i}/{max_iterations} ──")

        critic_start = time.time()
        all_chunk_feedback = []
        total_issues = 0

        for c in range(num_chunks):
            print(f"    [Critic] Chunk {c + 1}/{num_chunks}...")
            chunk_feedback, num_issues = critique_chunk(
                chunks[c], current_chunk_summaries[c]
            )
            all_chunk_feedback.append(chunk_feedback)
            total_issues += num_issues
            if num_issues > 0:
                print(f"      → {num_issues} issue(s)")
            else:
                print("      → no issues")

        print(
            f"    [Critic] Done in {time.time() - critic_start:.1f}s — {total_issues} total issue(s)"
        )

        if total_issues == 0:
            print("    [Stop] No issues found — stopping early.")
            iterations.append(
                {
                    "iteration": i,
                    "chunk_summaries_before": list(current_chunk_summaries),
                    "chunk_feedback": all_chunk_feedback,
                    "num_issues": 0,
                    "action": "stopped_no_issues",
                }
            )
            break

        refiner_start = time.time()
        refined_chunk_summaries = []

        for c in range(num_chunks):
            chunk_issues = count_issues(all_chunk_feedback[c])
            if chunk_issues > 0:
                print(
                    f"    [Refiner] Chunk {c + 1}/{num_chunks} — fixing {chunk_issues} issue(s)..."
                )
                refined = refine_chunk(
                    chunks[c], current_chunk_summaries[c], all_chunk_feedback[c]
                )
                refined_chunk_summaries.append(refined)
            else:
                print(
                    f"    [Refiner] Chunk {c + 1}/{num_chunks} — no issues, keeping as-is"
                )
                refined_chunk_summaries.append(current_chunk_summaries[c])

        print(f"    [Refiner] Done in {time.time() - refiner_start:.1f}s")

        iterations.append(
            {
                "iteration": i,
                "chunk_summaries_before": list(current_chunk_summaries),
                "chunk_feedback": all_chunk_feedback,
                "num_issues": total_issues,
                "chunk_summaries_after": list(refined_chunk_summaries),
                "action": "refined",
            }
        )
        current_chunk_summaries = refined_chunk_summaries

    print(f"\n  [Merge] Combining {num_chunks} refined chunk summaries...")
    merge_start = time.time()
    final_summary = merge_summaries(current_chunk_summaries)
    print(
        f"  [Merge] Done in {time.time() - merge_start:.1f}s — {len(final_summary.split())} words"
    )

    stopped_reason = "critic_no_issues" if total_issues == 0 else "max_iterations"

    return {
        "final_summary": final_summary,
        "final_chunk_summaries": current_chunk_summaries,
        "iterations": iterations,
        "total_iterations": len(iterations),
        "stopped_reason": stopped_reason,
    }


def main():
    with open(BASELINE_PATH, "r", encoding="utf-8") as f:
        baseline_data = json.load(f)
    with open(CLEANED_PATH, "r", encoding="utf-8") as f:
        cleaned_data = json.load(f)

    print(f"Loaded {len(baseline_data)} baseline summaries from {BASELINE_PATH}")
    print(f"Self-Refine config: max_iterations={MAX_ITERATIONS}")

    results = {}

    for meeting_id, meeting in baseline_data.items():
        print(f"\n{'=' * 60}")
        print(f"Processing {meeting_id}")
        print(f"{'=' * 60}")

        if meeting_id not in cleaned_data:
            print(f"  WARNING: {meeting_id} not found in cleaned data, skipping.")
            continue

        initial_summary = meeting["summary"]
        chunk_summaries = meeting["chunk_summaries"]
        chunks = cleaned_data[meeting_id]["chunks"]

        if len(chunk_summaries) != len(chunks):
            print(
                f"  WARNING: chunk count mismatch — "
                f"{len(chunk_summaries)} summaries vs {len(chunks)} chunks, skipping."
            )
            continue

        print(f"  Baseline summary: {len(initial_summary.split())} words")
        print(f"  Chunks: {len(chunks)}")
        for c, cs in enumerate(chunk_summaries):
            print(f"    Chunk {c + 1} summary: {len(cs.split())} words")

        start_time = time.time()
        refine_result = self_refine(
            chunk_summaries=chunk_summaries,
            chunks=chunks,
        )
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
            "baseline_chunk_summaries": chunk_summaries,
            "summary": refine_result["final_summary"],
            "refined_chunk_summaries": refine_result["final_chunk_summaries"],
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
