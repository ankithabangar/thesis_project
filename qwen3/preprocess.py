import json
import re
from pathlib import Path


RAW_DIRS = [
    Path("../data/QMSum/data/ALL/train"),
    Path("../data/QMSum/data/ALL/val"),
    Path("../data/QMSum/data/ALL/test"),
]
OUT_DIR = Path("data/preprocessed")
OUT_DIR.mkdir(exist_ok=True)

NOISE_PATTERNS = re.compile(
    r"\{disfmarker\}|\{gap\}|\{pause\}|\{vocalsound\}|\{nonvocalsound\}|\{comment\}|@"
)


DOMAIN_PREFIXES = {
    "Academic": ["bed", "bmr", "bro", "bdb", "buw"],
    "Product": ["es", "is", "ts"],
    "Committee": ["covid", "education"],
}


def get_domain(meeting_id: str) -> str:
    m_id = meeting_id.lower()
    for domain, prefixes in DOMAIN_PREFIXES.items():
        if any(m_id.startswith(p) for p in prefixes):
            return domain
    return "Unknown"


def clean_text(text: str) -> str:
    text = NOISE_PATTERNS.sub("", text)
    text = text.replace("{", "").replace("}", "")
    return re.sub(r"\s+", " ", text).strip()


def format_transcript(turns: list[dict]) -> list[str]:
    lines = []
    for turn in turns:
        speaker = turn.get("speaker", "Unknown").strip()
        content = clean_text(turn.get("content", ""))
        if content:
            lines.append(f"{speaker}: {content}")
    return lines


def extract_ground_truth(raw: dict) -> str:
    try:
        return raw["general_query_list"][0]["answer"].strip()
    except (KeyError, IndexError):
        return None


def process_meetings() -> dict:
    all_meetings: dict = {}

    for raw_dir in RAW_DIRS:
        json_files = sorted(raw_dir.glob("*.json"))
        if not json_files:
            print(f"ERROR: No JSON files found in {raw_dir}")
            continue

        for fpath in json_files:
            meeting_id = fpath.stem

            with open(fpath, "r", encoding="utf-8") as f:
                raw = json.load(f)

            ground_truth = extract_ground_truth(raw)

            turns = raw.get("meeting_transcripts", [])

            lines = format_transcript(turns)
            transcript = " ".join(lines)

            all_meetings[meeting_id] = {
                "domain": get_domain(meeting_id),
                "transcript": transcript,
                "ground_truth": ground_truth,
                "word_count": len(transcript.split()),
                "turn_count": len(turns),
            }

    return all_meetings


def main():
    all_meetings = process_meetings()

    if not all_meetings:
        print("ERROR: No meetings were processed. Check your data directories.")
        return

    out_path = OUT_DIR / "cleaned_meetings.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_meetings, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(all_meetings)} meetings to {out_path}")


if __name__ == "__main__":
    main()
