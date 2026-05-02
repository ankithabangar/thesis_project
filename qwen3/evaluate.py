import json
import nltk
from pathlib import Path

from rouge_score import rouge_scorer
from bert_score import score as bert_score
from summac.model_summac import SummaCConv
from minicheck.minicheck import MiniCheck


INPUT_PATH = Path("baseline_summaries.json")
OUTPUT_PATH = Path("baseline_evaluation_results.json")


def compute_rouge(
    generated_summaries: list[str], reference_summaries: list[str]
) -> list[dict]:
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    results = []
    for generated, reference in zip(generated_summaries, reference_summaries):
        scores = scorer.score(reference, generated)
        results.append(
            {
                "rouge1": round(scores["rouge1"].fmeasure, 4),
                "rouge2": round(scores["rouge2"].fmeasure, 4),
                "rougeL": round(scores["rougeL"].fmeasure, 4),
            }
        )
    return results


def compute_bertscore(
    generated_summaries: list[str], reference_summaries: list[str]
) -> list[dict]:
    P, R, F = bert_score(
        generated_summaries, reference_summaries, lang="en", verbose=False
    )
    results = []
    for p, r, f in zip(P.tolist(), R.tolist(), F.tolist()):
        results.append(
            {
                "bertscore_precision": round(p, 4),
                "bertscore_recall": round(r, 4),
                "bertscore_f1": round(f, 4),
            }
        )
    return results


def compute_summac(generated_summaries: list[str], sources: list[str]) -> list[dict]:
    model = SummaCConv(
        models=["vitc"],
        bins="percentile",
        granularity="sentence",
        nli_labels="e",
        device="cpu",
        start_file=None,
        agg="mean",
    )

    results = []
    for i, (summary, source) in enumerate(zip(generated_summaries, sources)):
        print(f"  SummaC: meeting {i + 1}/{len(generated_summaries)} ...")
        out = model.score([source], [summary])
        score = round(out["scores"][0], 4)

        print(f"    score: {score}")

        results.append({"summac_score": score})

    return results


nltk.download("punkt_tab", quiet=True)


def compute_minicheck(
    generated_summaries: list[str], sources: list[str]
) -> list[float]:
    scorer = MiniCheck(model_name="roberta-large", cache_dir="./ckpts")
    results = []
    for i, (summary, transcript) in enumerate(zip(generated_summaries, sources)):
        print(f"  MiniCheck: meeting {i + 1}/{len(generated_summaries)} ...")
        sentences = nltk.sent_tokenize(summary)
        if not sentences:
            results.append(
                {
                    "mean_prob": 0.0,
                    "supported_ratio": 0.0,
                    "num_claims": 0,
                    "sentence_scores": [],
                }
            )
            continue
        docs = [transcript] * len(sentences)
        pred_labels, raw_probs, _, _ = scorer.score(docs=docs, claims=sentences)

        sentence_scores = []
        for sent, label, prob in zip(sentences, pred_labels, raw_probs):
            sentence_scores.append(
                {
                    "sentence": sent,
                    "label": int(label),
                    "prob": round(float(prob), 4),
                }
            )

        mean_prob = sum(raw_probs) / len(raw_probs)
        supported_ratio = sum(pred_labels) / len(pred_labels)

        print(
            f"    {len(sentences)} sentences — mean_prob={mean_prob:.4f}, supported={sum(pred_labels)}/{len(pred_labels)}"
        )

        results.append(
            {
                "mean_prob": round(float(mean_prob), 4),
                "supported_ratio": round(float(supported_ratio), 4),
                "num_claims": len(sentences),
                "sentence_scores": sentence_scores,
            }
        )

    return results


def main():
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    meeting_ids = list(data.keys())
    generated_summaries = [data[m]["summary"] for m in meeting_ids]
    reference_summaries = [data[m]["ground_truth"] for m in meeting_ids]
    sources = [data[m]["transcript"] for m in meeting_ids]

    print("Computing ROUGE ...")
    rouge_scores = compute_rouge(generated_summaries, reference_summaries)

    print("Computing BERTScore ...")
    bert_scores = compute_bertscore(generated_summaries, reference_summaries)

    print("Computing SummaC ...")
    summac_scores = compute_summac(generated_summaries, sources)

    print("Computing MiniCheck ...")
    minicheck_scores = compute_minicheck(generated_summaries, sources)

    results = {}
    for i, m_id in enumerate(meeting_ids):
        mc = minicheck_scores[i]
        sc = summac_scores[i]
        results[m_id] = {
            **rouge_scores[i],
            **bert_scores[i],
            "summac_score": sc["summac_score"],
            "minicheck_mean_prob": mc["mean_prob"],
            "minicheck_supported_ratio": mc["supported_ratio"],
            "minicheck_num_claims": mc["num_claims"],
            "minicheck_sentence_scores": mc["sentence_scores"],
            "summary_words": len(generated_summaries[i].split()),
        }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    for m_id, scores in results.items():
        print(f"\n{'=' * 60}")
        print(f"  {m_id}  ({scores['summary_words']} words)")
        print(f"{'=' * 60}")
        print(f"    {'rouge1':<28} {scores['rouge1']}")
        print(f"    {'rouge2':<28} {scores['rouge2']}")
        print(f"    {'rougeL':<28} {scores['rougeL']}")
        print(f"    {'bertscore_f1':<28} {scores['bertscore_f1']}")
        print(f"    {'summac_score':<28} {scores['summac_score']}")
        print(f"    {'minicheck_mean_prob':<28} {scores['minicheck_mean_prob']}")
        print(
            f"    {'minicheck_supported_ratio':<28} {scores['minicheck_supported_ratio']}"
        )

    print(f"\nSaved results to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
