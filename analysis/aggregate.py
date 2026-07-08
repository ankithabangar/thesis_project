import csv
import json
import math
import statistics
from pathlib import Path


MODELS = ("gemini", "grok", "o4mini")
RUNS = (1, 2, 3)
STAGES = ("baseline", "reflection")

METRICS = [
    ("alignscore",                    "AlignScore",              True),
    ("minicheck_ft5_mean",            "MiniCheck mean prob",     True),
    ("minicheck_ft5_supported_ratio", "MiniCheck supported %",   True),
    ("qafacteval_score",              "QAFactEval",              True),
    ("rouge1",                        "ROUGE-1",                 True),
    ("rouge2",                        "ROUGE-2",                 True),
    ("rougeL",                        "ROUGE-L",                 True),
    ("bertscore_f1",                  "BERTScore-F1",            True),
    ("summary_words",                 "Summary words",           None),
]

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ── loading ───────────────────────────────────────────────────────────────────
def load_all() -> list[dict]:
    rows = []
    missing_combos = []
    for m in MODELS:
        for r in RUNS:
            for s in STAGES:
                p = PROJECT_ROOT / f"results/{m}/run{r}/{s}_evaluation.json"
                if not p.exists():
                    missing_combos.append(f"{m} run{r} {s}")
                    continue
                data = json.loads(p.read_text(encoding="utf-8"))
                for mid, entry in data.items():
                    row = {"model": m, "run": r, "stage": s, "meeting": mid}
                    for field, _, _ in METRICS:
                        row[field] = entry.get(field)  # None if missing
                    rows.append(row)
    if missing_combos:
        print("WARNING: missing combos ->", ", ".join(missing_combos))
    return rows


# ── stats helpers ─────────────────────────────────────────────────────────────
def mean_std(values):
    vals = [v for v in values if v is not None]
    if not vals:
        return None, None, 0
    if len(vals) == 1:
        return vals[0], 0.0, 1
    return statistics.mean(vals), statistics.stdev(vals), len(vals)


def _t_two_sided_p(t: float, df: int):
    try:
        from scipy import stats
    except ImportError:
        return None
    return float(2 * (1 - stats.t.cdf(abs(t), df=df)))


def paired_t_test(base_by_meeting: dict, refl_by_meeting: dict):
    diffs = []
    for mid, base_runs in base_by_meeting.items():
        if mid not in refl_by_meeting:
            continue
        base = [v for v in base_runs if v is not None]
        refl = [v for v in refl_by_meeting[mid] if v is not None]
        if not base or not refl:
            continue
        diffs.append(statistics.mean(refl) - statistics.mean(base))
    n = len(diffs)
    if n < 2:
        return None, None, None, n
    md = statistics.mean(diffs)
    sd = statistics.stdev(diffs)
    if sd == 0:
        return md, float("inf") if md != 0 else 0.0, 0.0 if md != 0 else 1.0, n
    t = md / (sd / math.sqrt(n))
    p = _t_two_sided_p(t, df=n - 1)
    return md, t, p, n


# ── writers ───────────────────────────────────────────────────────────────────
def write_csv(rows: list[dict]) -> Path:
    path = OUT_DIR / "summary_table.csv"
    fields = ["model", "run", "stage", "meeting"] + [f for f, _, _ in METRICS]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow(row)
    print(f"wrote {path}  ({len(rows)} rows)")
    return path


def write_summary_md(rows: list[dict]) -> Path:
    path = OUT_DIR / "summary_table.md"
    lines = ["# Evaluation summary\n"]
    lines.append(
        "Each cell is mean plus/minus std across all measurements for that "
        "(model, stage) combination (20 meetings x 3 runs = up to 60 values). "
        "Missing measurements (e.g. QAFactEval before backfill completes) are "
        "excluded rather than treated as zero; the n column reports how many "
        "values contributed.\n"
    )
    for field, name, hib in METRICS:
        lines.append(f"\n## {name}\n")
        lines.append(
            "| model  | baseline (mean plus/minus std, n) | "
            "reflection (mean plus/minus std, n) | delta (reflection - baseline) |"
        )
        lines.append(
            "|--------|-----------------------------------|"
            "------------------------------------|-------------------------------|"
        )
        for m in MODELS:
            base_vals = [r[field] for r in rows if r["model"] == m and r["stage"] == "baseline"]
            refl_vals = [r[field] for r in rows if r["model"] == m and r["stage"] == "reflection"]
            bm, bs, bn = mean_std(base_vals)
            rm, rs, rn = mean_std(refl_vals)
            if bm is None or rm is None:
                lines.append(f"| {m} | — | — | — |")
                continue
            delta = rm - bm
            arrow = ""
            if hib is True:
                arrow = " up" if delta > 0 else (" down" if delta < 0 else "")
            elif hib is False:
                arrow = " down" if delta > 0 else (" up" if delta < 0 else "")
            lines.append(
                f"| {m} | {bm:.4f} +/- {bs:.4f} (n={bn}) | "
                f"{rm:.4f} +/- {rs:.4f} (n={rn}) | {delta:+.4f}{arrow} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {path}")
    return path


def write_paired_tests_md(rows: list[dict]) -> Path:
    path = OUT_DIR / "paired_tests.md"
    lines = ["# Paired t-tests: baseline vs self-reflection\n"]
    lines.append(
        "For each (model, metric): each meeting's baseline mean across the 3 "
        "runs is paired with its reflection mean, then a two-sided paired "
        "t-test is run over the 20 meetings.\n"
    )
    lines.append(
        "Significance: `*` p<0.05, `**` p<0.01, `***` p<0.001, "
        "`ns` not significant, `?` p-value unavailable (scipy not installed).\n"
    )
    lines.append(
        "**Interpretation:** a positive delta means self-reflection improved "
        "the metric versus baseline. For higher-is-better faithfulness metrics "
        "(AlignScore, MiniCheck, ROUGE, BERTScore, QAFactEval) positive is good.\n"
    )

    for field, name, hib in METRICS:
        if hib is None:
            continue  # skip pure descriptives like summary_words
        lines.append(f"\n## {name}\n")
        lines.append("| model  | delta mean | t     | p-value | N (meetings) | sig |")
        lines.append("|--------|------------|-------|---------|--------------|-----|")
        for m in MODELS:
            base_by_meeting = {}
            refl_by_meeting = {}
            for r in rows:
                if r["model"] != m:
                    continue
                bucket = base_by_meeting if r["stage"] == "baseline" else refl_by_meeting
                bucket.setdefault(r["meeting"], []).append(r[field])
            md, t, p, n = paired_t_test(base_by_meeting, refl_by_meeting)
            if md is None:
                lines.append(f"| {m} | — | — | — | {n} | — |")
                continue
            if p is None:
                p_str = "—"
                sig = "?"
            else:
                p_str = f"{p:.4f}"
                if p < 0.001:
                    sig = "***"
                elif p < 0.01:
                    sig = "**"
                elif p < 0.05:
                    sig = "*"
                else:
                    sig = "ns"
            t_str = f"{t:+.2f}" if math.isfinite(t) else "inf"
            lines.append(f"| {m} | {md:+.4f} | {t_str} | {p_str} | {n} | {sig} |")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {path}")
    return path


# ── main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    rows = load_all()
    combos = {(r["model"], r["run"], r["stage"]) for r in rows}
    print(f"loaded {len(rows)} rows across {len(combos)} combos")
    write_csv(rows)
    write_summary_md(rows)
    write_paired_tests_md(rows)


if __name__ == "__main__":
    main()
