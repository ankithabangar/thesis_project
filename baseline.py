import json
import time
from pathlib import Path

from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate


INPUT_PATH = Path("./data/preprocessed/cleaned_meetings.json")
OUTPUT_PATH = Path("./data/results/baseline_summaries.json")


with open(INPUT_PATH, "r", encoding="utf-8") as f:
    cleaned_meetings = json.load(f)


model = OllamaLLM(model="llama3.1:8b")


chunk_template = """
You are an expert meeting summariser.
Read the following section of a meeting transcript carefully and write
a concise, accurate summary in plain prose (no bullet points, no headers).
Capture the key discussion points, decisions made, and any action items
mentioned in this section.
 
Important: Only include information that is explicitly stated in the transcript.
Do not add, infer, or assume any details that are not directly mentioned.
 
Write only the summary as a single paragraph. Do not include any preamble or explanation.
 
TRANSCRIPT SECTION ({chunk_index} of {total_chunks}):
{chunk}
"""

chunk_prompt = ChatPromptTemplate.from_template(chunk_template)
chunk_chain = chunk_prompt | model


# Merge chunk summaries
merge_template = """
You are an expert meeting summariser.
Below are summaries of consecutive sections of a single meeting transcript.
Combine them into one concise, accurate, and coherent summary in plain prose
(no bullet points, no headers). The final summary should cover the full meeting
from beginning to end, capturing the key discussion points, decisions made,
and any action items mentioned.
 
Important: Only include information that is present in the section summaries.
Do not add, infer, or assume any details that are not directly mentioned.
Remove any redundancies across sections but do not drop important information.
Do not refer to speakers by name or role. Focus on the content of the discussion, not who said what.
 
Write only the summary as a single paragraph of no more than 150 words. Do not include any preamble or explanation.
 
SECTION SUMMARIES:
{combined_summaries}
"""

merge_prompt = ChatPromptTemplate.from_template(merge_template)
merge_chain = merge_prompt | model

results = {}

test_ids = ["Bmr006"]
meetings = cleaned_meetings.items()

if test_ids:
    meetings = [(mid, m) for mid, m in meetings if mid in test_ids]

for meeting_id, meeting in meetings:
    chunks = meeting["chunks"]
    print(f"\n{'=' * 60}")
    print(f"Processing {meeting_id} — {len(chunks)} chunks")
    print(f"{'=' * 60}")

    start_time = time.time()
    # Summarise each chunk
    chunk_summaries = []
    for i, chunk in enumerate(chunks, start=1):
        print(f" Summarising chunk {i}/{len(chunks)} ({len(chunk.split())} words)...")
        safe_chunk = chunk.replace("{", "{{").replace("}", "}}")
        summary = chunk_chain.invoke(
            {
                "chunk": safe_chunk,
                "chunk_index": i,
                "total_chunks": len(chunks),
            }
        )
        chunk_summaries.append(summary.strip())

    combined = "\n\n".join(
        f"[Section {i}] {s}" for i, s in enumerate(chunk_summaries, start=1)
    )

    print(f"  Merging {len(chunk_summaries)} chunk summaries...")
    final_summary = merge_chain.invoke({"combined_summaries": combined}).strip()

    elapsed = time.time() - start_time
    print(f"  Done in {elapsed:.1f} seconds.")
    print(f"  Done. Summary length: {len(final_summary.split())} words")

    results[meeting_id] = {
        "meeting_id": meeting_id,
        "domain": meeting["domain"],
        "summary": final_summary,
        "chunk_summaries": chunk_summaries,
        "ground_truth": meeting["ground_truth"],
    }

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)


with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\nSaved {len(results)} summaries to {OUTPUT_PATH}")
