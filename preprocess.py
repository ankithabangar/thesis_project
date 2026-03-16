import json  # To read json files
import re  # Stands for - regular expressions, used for finding and removing patterns in text
import argparse
import logging as log  # For logging messages to the console
from pathlib import Path  # to write file path


RAW_DIRS = [
    Path("./data/raw/QMSum/data/ALL/train"),
    Path("./data/raw/QMSum/data/ALL/val"),
    Path("./data/raw/QMSum/data/ALL/test"),
]
OUT_DIR = Path("./data/preprocessed")
OUT_DIR.mkdir(exist_ok=True)  # Create the output directory if it doesn't exist


# Maximum number of words per chunk for the map-reduce summarisation strategy.
# Chunks are split at speaker turn boundaries to avoid cutting mid-utterance.
CHUNK_WORD_LIMIT = 2000

# Define a regex pattern to match all noise markers we want to remove
NOISE_PATTERNS = re.compile(
    r"\{disfmarker\}|\{gap\}|\{pause\}|\{vocalsound\}|\{nonvocalsound\}|\{comment\}|@"
)

# Domain lookup: prefix → domain name
DOMAIN_PREFIXES = {
    "Academic": ["bed", "bmr", "bro", "bdb", "buw"],
    "Product": ["es", "is", "ts"],
    "Committee": ["covid", "education"],
}


# This function takes a meeting ID and returns the domain it belongs to based on its prefix.
def get_domain(meeting_id: str) -> str:
    m_id = meeting_id.lower()
    for domain, prefixes in DOMAIN_PREFIXES.items():
        if any(m_id.startswith(p) for p in prefixes):
            return domain
    return "Unknown"


# This function takes a string of text and removes all the noise markers defined in NOISE_PATTERNS.
def clean_text(text: str) -> str:
    text = NOISE_PATTERNS.sub("", text)
    text = text.replace("{", "").replace("}", "")
    return re.sub(r"\s+", " ", text).strip()


# This function takes a list of turns (each turn is a dict with 'speaker' and 'content') and formats it into a single string.
def format_transcript(turns: list) -> str:
    lines = []
    for turn in turns:
        speaker = turn.get("speaker", "Unknown").strip()
        content = clean_text(turn.get("content", ""))
        if content:
            lines.append(f"{speaker}: {content}")
    return lines


def chunk_lines(lines: list[str], word_limit: int) -> list[str]:
    chunks = []
    current_chunk: list[str] = []
    current_words = 0

    for line in lines:
        line_words = len(line.split())

        if current_chunk and (current_words + line_words) > word_limit:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_words = 0

        current_chunk.append(line)
        current_words += line_words

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def extract_ground_truth(raw: dict) -> str | None:
    """Return the general-query ground truth summary, or None if missing."""
    try:
        return raw["general_query_list"][0]["answer"].strip()
    except (KeyError, IndexError):
        return None


def process_meetings(chunk_word_limit: int) -> dict:
    all_meetings: dict = {}

    # Get all .json files in the raw directory
    for raw_dir in RAW_DIRS:
        json_files = sorted(raw_dir.glob("*.json"))
        if not json_files:
            print(f"ERROR: No JSON files found in {raw_dir}")
            print("Make sure you run this script from the thesis_project/ directory.")
            continue

    for fpath in json_files:
        # Get the filename without extension to use as meeting ID
        meeting_id = fpath.stem

        with open(fpath, "r", encoding="utf-8") as f:
            raw = json.load(f)

        ground_truth = extract_ground_truth(raw)
        if ground_truth is None:
            print(f"No ground truth for {meeting_id} - skipping.")
            continue

        # Process the transcript
        turns = raw.get("meeting_transcripts", [])
        if not turns:
            print(f"  WARNING: Empty transcript for {meeting_id}, skipping.")
            continue

        lines = format_transcript(turns)
        transcript = " ".join(lines)
        chunks = chunk_lines(lines, word_limit=chunk_word_limit)

        all_meetings[meeting_id] = {
            "domain": get_domain(meeting_id),
            "transcript": transcript,
            "chunks": chunks,
            "ground_truth": ground_truth,
            "word_count": len(transcript.split()),
            "turn_count": len(turns),
            "chunk_count": len(chunks),
        }

    return all_meetings


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess QMSum transcripts for meeting summarisation."
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=CHUNK_WORD_LIMIT,
        help=f"Approximate max words per chunk (default: {CHUNK_WORD_LIMIT}).",
    )
    args = parser.parse_args()

    OUT_DIR.mkdir(exist_ok=True)

    all_meetings = process_meetings(chunk_word_limit=args.chunk_size)

    if not all_meetings:
        log.error("No meetings were processed. Check your data directories.")
        return

    out_path = OUT_DIR / "cleaned_meetings.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_meetings, f, indent=2, ensure_ascii=False)

    log.info(
        "Saved %d meetings to %s (chunk size: %d words).",
        len(all_meetings),
        out_path,
        args.chunk_size,
    )


if __name__ == "__main__":
    main()
