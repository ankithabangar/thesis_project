import sys
from pathlib import Path

try:
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
except ImportError:
    print("ERROR: this script needs numpy + pandas + matplotlib.")
    print("  pip install numpy pandas matplotlib")
    sys.exit(1)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = PROJECT_ROOT / "analysis" / "summary_table.csv"
OUT_DIR = PROJECT_ROOT / "analysis" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = ["gemini", "grok", "o4mini"]
STAGES = ["baseline", "reflection"]

MODEL_COLORS = {"gemini": "#4285F4", "grok": "#4B5563", "o4mini": "#10A37F"}
STAGE_COLORS = {"baseline": "#9CA3AF", "reflection": "#3B82F6"}

FAITHFULNESS_METRICS = [
    ("alignscore",                    "AlignScore"),
    ("minicheck_ft5_mean",            "MiniCheck (mean prob)"),
    ("minicheck_ft5_supported_ratio", "MiniCheck (supported ratio)"),
    ("qafacteval_score",              "QAFactEval"),
]
OVERLAP_METRICS = [
    ("rouge1",       "ROUGE-1"),
    ("rouge2",       "ROUGE-2"),
    ("rougeL",       "ROUGE-L"),
    ("bertscore_f1", "BERTScore-F1"),
]
ALL_METRICS = FAITHFULNESS_METRICS + OVERLAP_METRICS


def load_df() -> "pd.DataFrame":
    if not CSV_PATH.exists():
        raise SystemExit(
            f"CSV not found at {CSV_PATH}\n"
            f"Run  python3 analysis/aggregate.py  first."
        )
    return pd.read_csv(CSV_PATH)


def plot_baseline_vs_reflection(df):
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for i, (field, name) in enumerate(ALL_METRICS):
        ax = axes[i]
        agg = df.groupby(["model", "stage"])[field].agg(["mean", "std"]).reset_index()
        x = np.arange(len(MODELS))
        width = 0.36
        for j, stage in enumerate(STAGES):
            heights = [agg[(agg["model"] == m) & (agg["stage"] == stage)]["mean"].values[0]
                       for m in MODELS]
            errs = [agg[(agg["model"] == m) & (agg["stage"] == stage)]["std"].values[0]
                    for m in MODELS]
            offset = (j - 0.5) * width
            ax.bar(x + offset, heights, width, yerr=errs,
                   label=stage.title(), color=STAGE_COLORS[stage],
                   capsize=3, edgecolor="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(MODELS)
        ax.set_title(name, fontsize=11)
        ax.grid(True, alpha=0.3, axis="y")
        if i == 0:
            ax.legend(loc="upper right", fontsize=9)

    fig.suptitle("Baseline vs Self-Reflection  (mean +/- std over 60 measurements = 20 meetings x 3 runs)",
                 fontsize=13, y=1.01)
    fig.tight_layout()
    out = OUT_DIR / "fig_baseline_vs_reflection.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def plot_delta_per_metric(df):
    rows = []
    for m in MODELS:
        for field, name in ALL_METRICS:
            sub = df[df["model"] == m]
            base = sub[sub["stage"] == "baseline"].groupby("meeting")[field].mean()
            refl = sub[sub["stage"] == "reflection"].groupby("meeting")[field].mean()
            paired = pd.DataFrame({"b": base, "r": refl}).dropna()
            diffs = paired["r"] - paired["b"]
            rows.append({
                "model": m, "metric": name, "field": field,
                "delta_mean": diffs.mean(),
                "delta_se": diffs.std() / np.sqrt(len(diffs)) if len(diffs) > 1 else 0.0,
                "n": len(diffs),
            })
    delta_df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(13, 6))
    x = np.arange(len(ALL_METRICS))
    width = 0.26
    for i, m in enumerate(MODELS):
        sub = delta_df[delta_df["model"] == m]
        heights = [sub[sub["field"] == f]["delta_mean"].values[0] for f, _ in ALL_METRICS]
        errs = [sub[sub["field"] == f]["delta_se"].values[0] for f, _ in ALL_METRICS]
        ax.bar(x + (i - 1) * width, heights, width, yerr=errs,
               label=m, color=MODEL_COLORS[m],
               capsize=3, edgecolor="black", linewidth=0.5)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([n for _, n in ALL_METRICS], rotation=25, ha="right")
    ax.set_ylabel("Delta (reflection - baseline)")
    ax.set_title("Effect of self-reflection per model  (positive = self-reflection improved metric)",
                 fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    out = OUT_DIR / "fig_delta_per_metric.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def plot_summary_length(df):
    fig, ax = plt.subplots(figsize=(10, 6))

    positions, labels, data, colors = [], [], [], []
    for i, m in enumerate(MODELS):
        for j, s in enumerate(STAGES):
            vals = df[(df["model"] == m) & (df["stage"] == s)]["summary_words"].dropna().values
            positions.append(i * 3 + j)
            labels.append(f"{m}\n{s}")
            data.append(vals)
            colors.append(STAGE_COLORS[s])

    bp = ax.boxplot(data, positions=positions, widths=0.7,
                    patch_artist=True, showfliers=True)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)

    ax.axhline(150, color="red", linestyle="--", alpha=0.6,
               label="prompt upper bound (150 words)")
    ax.axhline(100, color="red", linestyle=":", alpha=0.4,
               label="prompt lower bound (100 words)")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Summary word count")
    ax.set_title("Summary length distribution per (model, stage)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    out = OUT_DIR / "fig_summary_length.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def plot_paired_dots(df):
    for field, name in FAITHFULNESS_METRICS:
        fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)
        for k, m in enumerate(MODELS):
            ax = axes[k]
            sub = df[df["model"] == m]
            base = sub[sub["stage"] == "baseline"].groupby("meeting")[field].mean()
            refl = sub[sub["stage"] == "reflection"].groupby("meeting")[field].mean()
            paired = pd.DataFrame({"baseline": base, "reflection": refl}).dropna()

            n_up = n_down = 0
            for _, row in paired.iterrows():
                improved = row["reflection"] > row["baseline"]
                color = "#10B981" if improved else "#EF4444"  # green up / red down
                ax.plot([0, 1], [row["baseline"], row["reflection"]],
                        color=color, alpha=0.5, marker="o", markersize=4, linewidth=1)
                n_up += int(improved)
                n_down += int(not improved)

            # Mean lines
            mean_base = paired["baseline"].mean()
            mean_refl = paired["reflection"].mean()
            ax.plot([0, 1], [mean_base, mean_refl],
                    color="black", linewidth=2.5, marker="D", markersize=8,
                    label=f"mean ({mean_base:.3f} → {mean_refl:.3f})")

            ax.set_xticks([0, 1])
            ax.set_xticklabels(["baseline", "reflection"])
            ax.set_title(f"{m}   improved: {n_up}/{n_up + n_down}", fontsize=11)
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best", fontsize=9)

        axes[0].set_ylabel(name)
        fig.suptitle(f"{name}: baseline -> reflection per meeting", fontsize=13, y=1.02)
        fig.tight_layout()
        out = OUT_DIR / f"fig_paired_{field}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {out}")


def main():
    df = load_df()
    print(f"loaded {len(df)} rows from {CSV_PATH}")
    plot_baseline_vs_reflection(df)
    plot_delta_per_metric(df)
    plot_summary_length(df)
    plot_paired_dots(df)
    print(f"\nAll figures written to {OUT_DIR}/")


if __name__ == "__main__":
    main()
