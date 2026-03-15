import json  # To read json files
import re  # Stands for - regular expressions, used for finding and removing patterns in text
import os  # Let's python interact with your operating system, e.g. to read files from a folder
from pathlib import Path  # to write file path


RAW_DIR = Path("./data/raw/QMSum/data/ALL/val")
OUT_DIR = Path("./data/preprocessed/val")
OUT_DIR.mkdir(exist_ok=True)  # Create the output directory if it doesn't exist


# This function takes a meeting ID and returns the domain it belongs to based on its prefix.
def get_domain(meeting_id: str) -> str:
    m_id = meeting_id.lower()
    if any(m_id.startswith(p) for p in ["bed", "bmr", "bro", "bdb", "buw"]):
        return "Academic"
    elif any(m_id.startswith(p) for p in ["es", "is", "ts"]):
        return "Product"
    elif any(m_id.startswith(p) for p in ["covid", "education"]):
        return "Committee"
    else:
        return "Unknown"


# Define a regex pattern to match all noise markers we want to remove
NOISE_PATTERNS = re.compile(
    r"\{disfmarker\}|\{gap\}|\{pause\}|\{vocalsound\}|\{nonvocalsound\}|\{comment\}|@"
)


# This function takes a string of text and removes all the noise markers defined in NOISE_PATTERNS.
def clean_text(text: str) -> str:
    text = NOISE_PATTERNS.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# This function takes a list of turns (each turn is a dict with 'speaker' and 'content') and formats it into a single string.
def format_transcript(turns: list) -> str:
    lines = []
    for turn in turns:
        speaker = turn.get("speaker", "Unknown").strip()
        content = clean_text(turn.get("content", ""))
        if content:
            lines.append(f"{speaker}: {content}")
    return "\n".join(lines)


json_files = sorted(RAW_DIR.glob("*.json"))  # Get all .json files in the raw directory

if not json_files:
    print(f"ERROR: No JSON files found in {RAW_DIR}")
    print("Make sure you run this script from the thesis_project/ directory.")
    exit(1)

domain_counts = {"Academic": 0, "Product": 0, "Committee": 0, "Unknown": 0}


for fpath in json_files:
    meeting_id = fpath.stem  # Get the filename without extension to use as meeting ID
    # print(f"Processing meeting: {meeting_id}")

    with open(fpath, "r", encoding="utf-8") as f:
        raw = json.load(f)
        # print(f"Raw json data loaded {raw}")

    try:  # Extract the ground truth summary
        ground_truth = raw["general_query_list"][0]["answer"].strip()
    except (KeyError, IndexError):
        print(f"  WARNING: No ground truth found for {meeting_id}, skipping.")
        exit(1)
    # print(f"Ground Truth:\n{ground_truth}\n")

    # Process the transcript
    turns = raw.get("meeting_transcripts", [])
    # print(f"turns: {turns}")
    if not turns:
        print(f"  WARNING: Empty transcript for {meeting_id}, skipping.")
        exit(1)

    transcript = format_transcript(turns)
    domain = get_domain(meeting_id)

    # print(f"Transcript:\n{transcript[:500]}...\n")

    record = {
        "meeting_id": meeting_id,
        "domain": domain,
        "transcript": transcript,
        "ground_truth": ground_truth,
        "word_count": len(transcript.split()),
        "turn_count": len(turns),
    }

    # Save individual clean file
    out_path = OUT_DIR / f"{meeting_id}_clean.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)
