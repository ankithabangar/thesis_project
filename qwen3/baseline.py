import json
import time
from pathlib import Path

from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate


INPUT_PATH = Path("./data/preprocessed/cleaned_meetings.json")
OUTPUT_PATH = Path("baseline_summaries.json")

# TEST_IDS = ["ES2004a"]


model = OllamaLLM(model="qwen3:30b")


summary_template = """
You are an expert meeting summariser.
Read the following section meeting transcript carefully and write
a concise, accurate summary in plain prose (no bullet points, no headers).
Capture the key discussion points, decisions made, and any action items
mentioned in this section.
 
Important: Only include information that is explicitly stated in the transcript.
Do not add, infer, or assume any details that are not directly mentioned.
 
Write only the summary as a single paragraph of no more than 150 words. Do not include any preamble or explanation.
 
{transcript}
"""

summary_prompt = ChatPromptTemplate.from_template(summary_template)
summary_chain = summary_prompt | model


def summarise_meeting(meeting_id: str, meeting: dict) -> str:
    transcript = meeting["transcript"]
    print(f"\n{'=' * 60}")
    print(f"Processing {meeting_id}")
    print(f"{'=' * 60}")

    start_time = time.time()
    # Summarise transcript
    summary = summary_chain.invoke({"transcript": transcript}).strip()

    elapsed = time.time() - start_time
    print(
        f"  Done in {elapsed:.1f}s ({elapsed / 60:.1f} min). Summary: {len(summary.split())} words"
    )

    return {
        "meeting_id": meeting_id,
        "domain": meeting["domain"],
        "transcript": meeting["transcript"],
        "ground_truth": meeting["ground_truth"],
        "summary": summary,
    }


def main():
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        cleaned_meetings = json.load(f)

    meetings = cleaned_meetings.items()

    results = {mid: summarise_meeting(mid, m) for mid, m in meetings}

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(results)} summaries to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
