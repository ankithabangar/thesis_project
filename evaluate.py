import json
import nltk
from pathlib import Path

from rouge_score import rouge_scorer
from bert_score import score as bert_score
from summac.model_summac import SummaCConv
from minicheck.minicheck import MiniCheck
from qafacteval import QAFactEval
from alignscore import AlignScore


INPUT_PATH = Path("test_dir/baseline_summaries.json")
CLEANED_PATH = Path("data/preprocessed/cleaned_meetings.json")
OUTPUT_PATH = Path("results/llama3.1/test/baseline_evaluation_results.json")


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


def compute_summac(
    chunk_summaries_list: list[list[str]], chunks_list: list[list[str]]
) -> list[dict]:
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
    for i, (chunk_summaries, chunks) in enumerate(
        zip(chunk_summaries_list, chunks_list)
    ):
        print(
            f"  SummaC: meeting {i + 1}/{len(chunk_summaries_list)} "
            f"— {len(chunks)} chunks ..."
        )
        out = model.score(chunks, chunk_summaries)
        chunk_scores = [round(s, 4) for s in out["scores"]]
        avg_score = round(sum(chunk_scores) / len(chunk_scores), 4)

        print(f"    chunk scores: {chunk_scores}")
        print(f"    average: {avg_score}")

        results.append(
            {
                "summac_avg": avg_score,
                "summac_chunk_scores": chunk_scores,
            }
        )

    return results


def compute_qafacteval(
    chunk_summaries_list: list[list[str]],
    chunks_list: list[list[str]],
    model_folder: str = "./qafacteval_models",
) -> list[dict]:
    kwargs = {
        "cuda_device": -1,
        "use_lerc_quip": True,
        "verbose": True,
        "generation_batch_size": 32,
        "answering_batch_size": 32,
        "lerc_batch_size": 8,
    }

    metric = QAFactEval(
        lerc_quip_path=f"{model_folder}/quip-512-mocha",
        generation_model_path=f"{model_folder}/generation/model.tar.gz",
        answering_model_dir=f"{model_folder}/answering",
        lerc_model_path=f"{model_folder}/lerc/model.tar.gz",
        lerc_pretrained_model_path=f"{model_folder}/lerc/pretraining.tar.gz",
        **kwargs,
    )

    results = []
    for i, (chunk_summaries, chunks) in enumerate(
        zip(chunk_summaries_list, chunks_list)
    ):
        print(
            f"  QAFactEval: meeting {i + 1}/{len(chunk_summaries_list)} "
            f"— {len(chunks)} chunks ..."
        )
        chunk_scores = []
        for j, (chunk_summary, chunk) in enumerate(zip(chunk_summaries, chunks)):
            print(f"    chunk {j + 1}/{len(chunks)} ...")
            batch_results = metric.score_batch_qafacteval(
                [chunk], [[chunk_summary]], return_qa_pairs=True
            )
            score = batch_results[0][0]["qa-eval"]["lerc_quip"]
            chunk_scores.append(round(float(score), 4))
            print(f"      lerc_quip = {score:.4f}")

        avg_score = round(sum(chunk_scores) / len(chunk_scores), 4)
        print(f"    chunk scores: {chunk_scores}")
        print(f"    average: {avg_score}")

        results.append(
            {
                "qafacteval_avg": avg_score,
                "qafacteval_chunk_scores": chunk_scores,
            }
        )

    return results


def compute_alignscore(
    generated_summaries: list[str],
    sources: list[str],
    ckpt_path: str = "./AlignScore-large.ckpt",
) -> list[float]:
    scorer = AlignScore(
        model="roberta-large",
        batch_size=32,
        device="cpu",
        ckpt_path=ckpt_path,
        evaluation_mode="nli_sp",
    )

    scores = scorer.score(contexts=sources, claims=generated_summaries)
    results = [round(float(s), 4) for s in scores]

    for i, s in enumerate(results):
        print(f"  AlignScore: meeting {i + 1}/{len(results)} — score={s}")

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

    with open(CLEANED_PATH, "r", encoding="utf-8") as f:
        cleaned_data = json.load(f)

    meeting_ids = list(data.keys())
    generated_summaries = [data[m]["summary"] for m in meeting_ids]
    reference_summaries = [data[m]["ground_truth"] for m in meeting_ids]
    sources = [data[m]["transcript"] for m in meeting_ids]

    # Chunk-level data for SummaC and QAFactEval
    chunk_summaries_list = [data[m]["chunk_summaries"] for m in meeting_ids]
    chunks_list = [cleaned_data[m]["chunks"] for m in meeting_ids]

    print("Computing ROUGE ...")
    rouge_scores = compute_rouge(generated_summaries, reference_summaries)

    print("Computing BERTScore ...")
    bert_scores = compute_bertscore(generated_summaries, reference_summaries)

    print("Computing SummaC (chunk-level) ...")
    summac_scores = compute_summac(chunk_summaries_list, chunks_list)

    print("Computing MiniCheck ...")
    minicheck_scores = compute_minicheck(generated_summaries, sources)

    print("Computing QAFactEval (chunk-level) ...")
    qafacteval_scores = compute_qafacteval(chunk_summaries_list, chunks_list)

    print("Computing AlignScore ...")
    alignscore_scores = compute_alignscore(generated_summaries, sources)

    results = {}
    for i, m_id in enumerate(meeting_ids):
        mc = minicheck_scores[i]
        sc = summac_scores[i]
        qa = qafacteval_scores[i]
        results[m_id] = {
            **rouge_scores[i],
            **bert_scores[i],
            "summac_avg": sc["summac_avg"],
            "summac_chunk_scores": sc["summac_chunk_scores"],
            "minicheck_mean_prob": mc["mean_prob"],
            "minicheck_supported_ratio": mc["supported_ratio"],
            "minicheck_num_claims": mc["num_claims"],
            "minicheck_sentence_scores": mc["sentence_scores"],
            "qafacteval_avg": qa["qafacteval_avg"],
            "qafacteval_chunk_scores": qa["qafacteval_chunk_scores"],
            "alignscore": alignscore_scores[i],
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
        print(f"    {'summac_avg':<28} {scores['summac_avg']}")
        print(f"    {'minicheck_mean_prob':<28} {scores['minicheck_mean_prob']}")
        print(
            f"    {'minicheck_supported_ratio':<28} {scores['minicheck_supported_ratio']}"
        )
        print(f"    {'qafacteval_avg':<28} {scores['qafacteval_avg']}")
        print(f"    {'alignscore':<28} {scores['alignscore']}")

    print(f"\nSaved results to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
