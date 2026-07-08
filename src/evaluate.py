"""Evaluation for one (model, run, stage) combo of the thesis grid.

Reference-free, faithfulness-focused (headline thesis signal):
  1. AlignScore-large              -- NLI sentence-pair faithfulness
  2. MiniCheck-Flan-T5-Large       -- sentence-level support, TofuEval-MeetingBank SOTA

Reference-based (overlap with QMSum ground-truth, for comparability with prior work):
  4. ROUGE-1 / ROUGE-2 / ROUGE-L  -- n-gram overlap
  5. BERTScore-F1 (roberta-large)  -- semantic similarity

Reads  results/{model}/run{run}/{stage}_summaries.json
Writes results/{model}/run{run}/{stage}_evaluation.json

MUST be run inside the Docker container (`thesis-evaluate`): QAFactEval +
AlignScore have legacy pin tangles that don't install on Apple Silicon /
Python 3.11.
"""
import argparse
import json
import os
import time
from pathlib import Path

import torch

torch.set_num_threads(os.cpu_count() or 1)


SUPPORTED_MODELS = ("gemini", "grok", "o4mini", "qwen")
SUPPORTED_STAGES = ("baseline", "reflection")
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
        help="Only evaluate the first N to-do meetings (for smoke tests).",
    )
    return p.parse_args()


# ── QAFactEval ────────────────────────────────────────────────────────────────
QAFACTEVAL_MAX_SOURCE_WORDS = 3000


def load_qafacteval():
    from qafacteval import QAFactEval

    model_folder = str(PROJECT_ROOT / "qafacteval_models")
    print("  Loading QAFactEval...")
    t0 = time.time()
    metric = QAFactEval(
        lerc_quip_path=f"{model_folder}/quip-512-mocha",
        generation_model_path=f"{model_folder}/generation/model.tar.gz",
        answering_model_dir=f"{model_folder}/answering",
        lerc_model_path=f"{model_folder}/lerc/model.tar.gz",
        lerc_pretrained_model_path=f"{model_folder}/lerc/pretraining.tar.gz",
        cuda_device=-1,
        use_lerc_quip=True,
        verbose=False,
        generation_batch_size=8,
        answering_batch_size=8,
        lerc_batch_size=4,
    )
    print(f"  QAFactEval loaded in {time.time() - t0:.1f}s")
    return metric


def score_qafacteval(metric, summary: str, source: str) -> dict:
    words = source.split()
    truncated = len(words) > QAFACTEVAL_MAX_SOURCE_WORDS
    if truncated:
        source = " ".join(words[:QAFACTEVAL_MAX_SOURCE_WORDS])
    out = metric.score_batch_qafacteval([source], [[summary]], return_qa_pairs=True)
    return {
        "qafacteval_score": round(float(out[0][0]["qa-eval"]["lerc_quip"]), 4),
        "qafacteval_source_truncated": truncated,
    }


# ── AlignScore-large ──────────────────────────────────────────────────────────
def load_alignscore():
    from alignscore import AlignScore

    ckpt = str(PROJECT_ROOT / "ckpts/AlignScore-large.ckpt")
    print("  Loading AlignScore-large...")
    t0 = time.time()
    scorer = AlignScore(
        model="roberta-large",
        batch_size=16,
        device="cpu",
        ckpt_path=ckpt,
        evaluation_mode="nli_sp",
        verbose=False,
    )
    print(f"  AlignScore loaded in {time.time() - t0:.1f}s")
    return scorer


def score_alignscore(scorer, summary: str, source: str) -> dict:
    source_words = len(source.split())
    oom_truncated = False
    try:
        scores = scorer.score(contexts=[source], claims=[summary])
    except RuntimeError as e:
        if "memory" in str(e).lower() or "alloc" in str(e).lower():
            print(
                f"    !! AlignScore OOM — retrying with first 6000 words "
                f"(of {source_words})"
            )
            scores = scorer.score(
                contexts=[" ".join(source.split()[:6000])], claims=[summary]
            )
            oom_truncated = True
        else:
            raise
    return {
        "alignscore": round(float(scores[0]), 4),
        "alignscore_oom_truncated": oom_truncated,
    }


# ── MiniCheck-Flan-T5-Large ───────────────────────────────────────────────────
def _chunk_words(text: str, chunk_size: int = 250, overlap: int = 50) -> list[str]:
    words = text.split()
    if len(words) <= chunk_size:
        return [text]
    step = max(1, chunk_size - overlap)
    chunks = []
    for start in range(0, len(words), step):
        chunk = words[start : start + chunk_size]
        if not chunk:
            break
        chunks.append(" ".join(chunk))
        if start + chunk_size >= len(words):
            break
    return chunks


def load_minicheck():
    import spacy
    from minicheck.minicheck import MiniCheck

    print("  Loading MiniCheck-Flan-T5-Large + spaCy...")
    t0 = time.time()
    scorer = MiniCheck(
        model_name="flan-t5-large",
        cache_dir="/root/.cache/huggingface/hub",
        enable_prefix_caching=False,
    )
    nlp = spacy.load("en_core_web_sm")
    print(f"  MiniCheck loaded in {time.time() - t0:.1f}s")
    return scorer, nlp


def score_minicheck(scorer, nlp, summary: str, source: str, threshold: float = 0.5) -> dict:
    sentences = [s.text.strip() for s in nlp(summary).sents if s.text.strip()]
    if not sentences:
        return {
            "minicheck_ft5_mean": 1.0,
            "minicheck_ft5_supported_ratio": 1.0,
            "minicheck_ft5_n_sentences": 0,
        }
    # chunk_size=500 halves the number of windows per transcript
    # (~68 -> ~34 for a 13k-word meeting), cutting MiniCheck wall-clock from
    # ~11 min to ~5-6 min per meeting under x86 emulation. Each chunk still
    # fits comfortably in FT5-Large's ~512-token context window, and the
    # max-across-chunks semantics preserves the "supported if any chunk
    # supports it" intent of MiniCheck.
    chunks = _chunk_words(source, chunk_size=500, overlap=100)
    sentence_max = []
    for sent in sentences:
        _, probs, _, _ = scorer.score(docs=chunks, claims=[sent] * len(chunks))
        sentence_max.append(max(probs))
    mean_prob = round(sum(sentence_max) / len(sentence_max), 4)
    supported = sum(1 for p in sentence_max if p >= threshold)
    return {
        "minicheck_ft5_mean": mean_prob,
        "minicheck_ft5_supported_ratio": round(supported / len(sentence_max), 4),
        "minicheck_ft5_n_sentences": len(sentence_max),
    }


# ── ROUGE (reference-based) ───────────────────────────────────────────────────
def load_rouge():
    from rouge_score import rouge_scorer

    print("  Loading ROUGE scorer...")
    return rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)


def score_rouge(scorer, summary: str, ground_truth: str) -> dict:
    s = scorer.score(ground_truth, summary)
    return {
        "rouge1": round(s["rouge1"].fmeasure, 4),
        "rouge2": round(s["rouge2"].fmeasure, 4),
        "rougeL": round(s["rougeL"].fmeasure, 4),
    }


# ── BERTScore-F1 (reference-based) ────────────────────────────────────────────
def load_bertscore():
    from bert_score import BERTScorer

    print("  Loading BERTScore (roberta-large)...")
    t0 = time.time()
    scorer = BERTScorer(
        lang="en",
        model_type="roberta-large",
        device="cpu",
        rescale_with_baseline=False,
    )
    print(f"  BERTScore loaded in {time.time() - t0:.1f}s")
    return scorer


def score_bertscore(scorer, summary: str, ground_truth: str) -> dict:
    P, R, F = scorer.score([summary], [ground_truth])
    return {
        "bertscore_precision": round(float(P[0]), 4),
        "bertscore_recall": round(float(R[0]), 4),
        "bertscore_f1": round(float(F[0]), 4),
    }


# ── main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()

    input_path = (
        PROJECT_ROOT
        / f"results/{args.model}/run{args.run}/{args.stage}_summaries.json"
    )
    output_path = (
        PROJECT_ROOT
        / f"results/{args.model}/run{args.run}/{args.stage}_evaluation.json"
    )

    if not input_path.exists():
        raise SystemExit(f"summary file not found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summaries = json.loads(input_path.read_text(encoding="utf-8"))
    results = (
        json.loads(output_path.read_text(encoding="utf-8"))
        if output_path.exists()
        else {}
    )

    print("=" * 72)
    print(f"  Evaluating {args.stage} summaries  ({args.model} run{args.run})")
    print(f"  Meetings: {len(summaries)}; already scored: {len(results)}")
    print("=" * 72)

    todo = [(mid, m) for mid, m in summaries.items() if mid not in results]
    if args.limit is not None:
        todo = todo[: args.limit]
        print(f"  --limit {args.limit} applied: only first {len(todo)} will run.")
    if not todo:
        print("Nothing to do — all meetings already evaluated.")
        return

    # Load all scorers once at startup. 
    alignscore = load_alignscore()
    minicheck, nlp = load_minicheck()
    rouge = load_rouge()
    bertscore = load_bertscore()

    for i, (mid, meeting) in enumerate(todo, start=1):
        summary = meeting["summary"]
        source = meeting["transcript"]
        ground_truth = meeting["ground_truth"]
        n_words = len(summary.split())

        print(f"\n[{i}/{len(todo)}] {mid}  ({n_words} words)")

        t0 = time.time()
        align = score_alignscore(alignscore, summary, source)
        print(f"  AlignScore:  {align['alignscore']:.4f}  ({time.time() - t0:.1f}s)")

        t0 = time.time()
        mc = score_minicheck(minicheck, nlp, summary, source)
        print(
            f"  MiniCheck:   mean={mc['minicheck_ft5_mean']:.4f}  "
            f"supported={mc['minicheck_ft5_supported_ratio']:.4f}  "
            f"({mc['minicheck_ft5_n_sentences']} sents, {time.time() - t0:.1f}s)"
        )

        t0 = time.time()
        rg = score_rouge(rouge, summary, ground_truth)
        print(
            f"  ROUGE:       1={rg['rouge1']:.4f}  2={rg['rouge2']:.4f}  "
            f"L={rg['rougeL']:.4f}  ({time.time() - t0:.1f}s)"
        )

        t0 = time.time()
        bs = score_bertscore(bertscore, summary, ground_truth)
        print(
            f"  BERTScore:   F1={bs['bertscore_f1']:.4f}  "
            f"({time.time() - t0:.1f}s)"
        )

        results[mid] = {
            **align,
            **mc,
            **rg,
            **bs,
            "summary_words": n_words,
        }
        output_path.write_text(
            json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    print(f"\nSaved {len(results)} evaluations to {output_path}")


if __name__ == "__main__":
    main()
