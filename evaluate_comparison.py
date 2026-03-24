import json
import nltk
from pathlib import Path

from rouge_score import rouge_scorer
from bert_score import score as bert_score
from summac.model_summac import SummaCConv
from minicheck.minicheck import MiniCheck


BASELINE_RESULTS = Path(".results/llama3.1/evaluation_results.json")
SELF_REFINE_PATH = Path(
    "./results/llama3.1/self_refined/iteration_3/self_refined_summaries.json"
)
OUTPUT_PATH = Path(
    "./results/llama3.1/evaluation_results/iteration_3/comparison_results.json"
)

METRICS = [
    "rouge1",
    "rouge2",
    "rougeL",
    "bertscore_f1",
    "summac",
    "minicheck_mean_prob",
    "minicheck_supported_ratio",
]

nltk.download("punkt_tab", quiet=True)


def compute_rouge(prediction: str, reference: str) -> dict:
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    s = scorer.score(reference, prediction)
    return {
        "rouge1": round(s["rouge1"].fmeasure, 4),
        "rouge2": round(s["rouge2"].fmeasure, 4),
        "rougeL": round(s["rougeL"].fmeasure, 4),
    }


def compute_bertscore(prediction: str, reference: str) -> dict:
    P, R, F = bert_score([prediction], [reference], lang="en", verbose=False)
    return {
        "bertscore_precision": round(P[0].item(), 4),
        "bertscore_recall": round(R[0].item(), 4),
        "bertscore_f1": round(F[0].item(), 4),
    }


def compute_summac(prediction: str, source: str) -> float:
    model = SummaCConv(
        models=["vitc"],
        bins="percentile",
        granularity="sentence",
        nli_labels="e",
        device="cpu",
        start_file=None,
        agg="mean",
    )
    out = model.score([source], [prediction])
    return round(out["scores"][0], 4)


def compute_minicheck(prediction: str, source: str) -> dict:
    scorer = MiniCheck(model_name="roberta-large", cache_dir="./ckpts")
    sentences = nltk.sent_tokenize(prediction)
    if not sentences:
        return {"mean_prob": 0.0, "supported_ratio": 0.0, "num_claims": 0}

    docs = [source] * len(sentences)
    pred_labels, raw_probs, _, _ = scorer.score(docs=docs, claims=sentences)

    mean_prob = sum(raw_probs) / len(raw_probs)
    supported_ratio = sum(pred_labels) / len(pred_labels)

    sentence_scores = []
    for sent, label, prob in zip(sentences, pred_labels, raw_probs):
        sentence_scores.append(
            {
                "sentence": sent,
                "label": int(label),
                "prob": round(float(prob), 4),
            }
        )

    return {
        "mean_prob": round(float(mean_prob), 4),
        "supported_ratio": round(float(supported_ratio), 4),
        "num_claims": len(sentences),
        "sentence_scores": sentence_scores,
    }


def evaluate_summary(summary: str, reference: str, source: str, label: str) -> dict:
    print(f"    Computing ROUGE ({label})...")
    rouge = compute_rouge(summary, reference)

    print(f"    Computing BERTScore ({label})...")
    bertscore = compute_bertscore(summary, reference)

    print(f"    Computing SummaC ({label})...")
    summac = compute_summac(summary, source)

    print(f"    Computing MiniCheck ({label})...")
    minicheck = compute_minicheck(summary, source)

    return {
        **rouge,
        **bertscore,
        "summac": summac,
        "minicheck_mean_prob": minicheck["mean_prob"],
        "minicheck_supported_ratio": minicheck["supported_ratio"],
        "minicheck_num_claims": minicheck["num_claims"],
        "minicheck_sentence_scores": minicheck["sentence_scores"],
        "summary_words": len(summary.split()),
    }


def main():
    with open(BASELINE_RESULTS, "r", encoding="utf-8") as f:
        baseline_scores_data = json.load(f)

    with open(SELF_REFINE_PATH, "r", encoding="utf-8") as f:
        refine_data = json.load(f)

    results = {}

    for meeting_id in refine_data:
        print(f"\n{'=' * 60}")
        print(f"  Evaluating {meeting_id}")
        print(f"{'=' * 60}")

        baseline_scores = baseline_scores_data[meeting_id]
        refined_summary = refine_data[meeting_id]["summary"]
        reference = refine_data[meeting_id]["ground_truth"]
        source = refine_data[meeting_id]["transcript"]

        # Evaluate self-refined summary
        print("\n  ── Self-Refine ──")
        refine_scores = evaluate_summary(
            refined_summary, reference, source, "self-refine"
        )

        # Compute deltas
        deltas = {
            key: round(refine_scores[key] - baseline_scores[key], 4) for key in METRICS
        }

        results[meeting_id] = {
            "baseline": baseline_scores,
            "self_refine": refine_scores,
            "delta": deltas,
            "self_refine_metadata": refine_data[meeting_id].get("self_refine", {}),
        }

        print(f"\n  {'Metric':<28} {'Baseline':>10} {'Self-Refine':>12} {'Delta':>10}")
        print(f"  {'-' * 62}")
        for key in METRICS:
            b, r, d = baseline_scores[key], refine_scores[key], deltas[key]
            arrow = "↑" if d > 0 else "↓" if d < 0 else "="
            print(f"  {key:<28} {b:>10.4f} {r:>12.4f} {d:>+10.4f} {arrow}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n\n{'=' * 70}")
    print(f"  AGGREGATE COMPARISON ACROSS {len(results)} MEETINGS")
    print(f"{'=' * 70}")
    print(
        f"\n  {'Metric':<28} {'Avg Baseline':>12} {'Avg Refine':>12} {'Avg Delta':>12}"
    )
    print(f"  {'-' * 66}")

    for metric in METRICS:
        avg_b = sum(results[m]["baseline"][metric] for m in results) / len(results)
        avg_r = sum(results[m]["self_refine"][metric] for m in results) / len(results)
        avg_d = sum(results[m]["delta"][metric] for m in results) / len(results)
        arrow = "↑" if avg_d > 0 else "↓" if avg_d < 0 else "="
        print(f"  {metric:<28} {avg_b:>12.4f} {avg_r:>12.4f} {avg_d:>+12.4f} {arrow}")

    print(f"\n  Saved detailed results to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
