#!/usr/bin/env python3
"""
Generate a three-panel behaviour analysis figure from the final 18 evaluation
summary CSVs for defense PPT slides.

Reads evaluation_results/{IDM,DQN,PPO}-Ego/{easy,medium,hard}/lane{2,3}_seed17/
summary_results.csv → behaviour_metrics_by_difficulty.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
EVAL_ROOT = PROJECT_ROOT / "evaluation_results"
OUT_PATH = PROJECT_ROOT / "behaviour_metrics_by_difficulty.png"

# Final experimental matrix (3 ego × 3 difficulty × 2 lanes = 18)
EGO_MODELS = ("IDM-Ego", "DQN-Ego", "PPO-Ego")
DIFFICULTIES = ("easy", "medium", "hard")
LANE_COUNTS = (2, 3)
TRAINING_SEED = 17

REQUIRED_COLUMNS = (
    "ego_model",
    "difficulty",
    "lane_count",
    "cut_in_count_mean",
    "lane_change_count_mean",
    "episode_length_mean",
)

# Panel definitions: (column, exact subplot title, y-axis label)
PANELS = (
    ("cut_in_count_mean", "Cut-ins per Episode", "Mean count"),
    ("lane_change_count_mean", "Lane-Change Actions per Episode", "Mean count"),
    ("episode_length_mean", "Episode Length", "Mean steps"),
)

DIFF_LABELS = ["Easy", "Medium", "Hard"]
DIFF_COLORS = {
    "easy": "#1f77b4",    # blue
    "medium": "#ff7f0e",  # orange
    "hard": "#d62728",    # red
}

# Horizontal jitter range for the 6 raw points within each difficulty category
JITTER_SPAN = 0.22


def expected_csv_path(ego: str, difficulty: str, lane: int) -> Path:
    return (
        EVAL_ROOT
        / ego
        / difficulty
        / f"lane{lane}_seed{TRAINING_SEED}"
        / "summary_results.csv"
    )


def load_final_18() -> pd.DataFrame:
    if not EVAL_ROOT.is_dir():
        raise FileNotFoundError(f"Evaluation root not found: {EVAL_ROOT}")

    rows = []
    missing = []

    for ego in EGO_MODELS:
        for difficulty in DIFFICULTIES:
            for lane in LANE_COUNTS:
                path = expected_csv_path(ego, difficulty, lane)
                if not path.is_file():
                    missing.append(str(path))
                    continue

                df = pd.read_csv(path)
                absent = [c for c in REQUIRED_COLUMNS if c not in df.columns]
                if absent:
                    raise ValueError(
                        f"{path}: missing required columns {absent}. "
                        f"Found: {list(df.columns)}"
                    )
                if len(df) != 1:
                    raise ValueError(
                        f"{path}: expected exactly 1 summary row, got {len(df)}"
                    )

                row = df.iloc[0]
                if str(row["ego_model"]) != ego:
                    raise ValueError(
                        f"{path}: ego_model='{row['ego_model']}', expected '{ego}'"
                    )
                if str(row["difficulty"]).lower() != difficulty:
                    raise ValueError(
                        f"{path}: difficulty='{row['difficulty']}', "
                        f"expected '{difficulty}'"
                    )
                if int(row["lane_count"]) != lane:
                    raise ValueError(
                        f"{path}: lane_count={row['lane_count']}, expected {lane}"
                    )

                values = {}
                for col in (
                    "cut_in_count_mean",
                    "lane_change_count_mean",
                    "episode_length_mean",
                ):
                    val = float(row[col])
                    if not np.isfinite(val):
                        raise ValueError(f"{path}: {col} is not finite ({val})")
                    if val < 0:
                        raise ValueError(f"{path}: {col} < 0 ({val})")
                    values[col] = val

                rows.append(
                    {
                        "ego_model": ego,
                        "difficulty": difficulty,
                        "lane_count": lane,
                        **values,
                        "source": str(path),
                    }
                )

    if missing:
        raise FileNotFoundError(
            "Missing summary CSV(s) for the final 18-group matrix:\n  "
            + "\n  ".join(missing)
        )

    if len(rows) != 18:
        raise RuntimeError(
            f"Expected exactly 18 evaluation groups, loaded {len(rows)}"
        )

    return pd.DataFrame(rows)


def difficulty_means(df: pd.DataFrame) -> pd.DataFrame:
    """Mean of each metric across the 6 experiments per difficulty."""
    return (
        df.groupby("difficulty", sort=False)[
            [
                "cut_in_count_mean",
                "lane_change_count_mean",
                "episode_length_mean",
            ]
        ]
        .mean()
        .reindex(DIFFICULTIES)
    )


def _format_mean(value: float, metric: str) -> str:
    if metric == "episode_length_mean":
        return f"{value:.1f}"
    return f"{value:.2f}"


def draw_panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    means: pd.DataFrame,
    metric: str,
    title: str,
    ylabel: str,
) -> None:
    x_positions = np.arange(len(DIFFICULTIES))

    for xi, difficulty in enumerate(DIFFICULTIES):
        subset = df[df["difficulty"] == difficulty][metric].to_numpy()
        if len(subset) != 6:
            raise RuntimeError(
                f"{metric} / {difficulty}: expected 6 points, got {len(subset)}"
            )

        color = DIFF_COLORS[difficulty]
        # Spread the 6 raw points horizontally (no lines between difficulties)
        offsets = np.linspace(-JITTER_SPAN, JITTER_SPAN, len(subset))
        ax.scatter(
            xi + offsets,
            subset,
            s=42,
            c=color,
            alpha=0.45,
            edgecolors="none",
            zorder=2,
        )

        mean_val = float(means.loc[difficulty, metric])
        # Short horizontal bar for the mean
        bar_half = 0.16
        ax.plot(
            [xi - bar_half, xi + bar_half],
            [mean_val, mean_val],
            color=color,
            linewidth=2.8,
            solid_capstyle="round",
            zorder=4,
        )
        ax.scatter(
            [xi],
            [mean_val],
            s=70,
            c=color,
            marker="D",
            edgecolors="white",
            linewidths=0.8,
            zorder=5,
        )
        ax.annotate(
            _format_mean(mean_val, metric),
            xy=(xi, mean_val),
            xytext=(10, 6),
            textcoords="offset points",
            fontsize=10,
            color=color,
            fontweight="medium",
            zorder=6,
        )

    ax.set_title(title, fontsize=12, pad=8)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(DIFF_LABELS)
    ax.set_xlim(-0.55, len(DIFFICULTIES) - 0.45)
    ax.grid(True, which="major", axis="y", color="#d0d0d0", linestyle="-", linewidth=0.8)
    ax.set_axisbelow(True)
    # Independent y-limits with a little headroom for annotations
    ymin = float(df[metric].min())
    ymax = float(df[metric].max())
    pad = max((ymax - ymin) * 0.12, 0.05 * max(abs(ymax), 1.0))
    ax.set_ylim(max(0.0, ymin - pad), ymax + pad * 1.6)


def make_figure(df: pd.DataFrame, means: pd.DataFrame, out_path: Path) -> None:
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 11,
            "ytick.labelsize": 10,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6), dpi=150)

    for ax, (metric, title, ylabel) in zip(axes, PANELS):
        draw_panel(ax, df, means, metric, title, ylabel)

    # Shared legend explaining markers (once, on the rightmost panel)
    from matplotlib.lines import Line2D

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#666666",
            markersize=7,
            alpha=0.5,
            label="Individual run (n=6)",
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            color="w",
            markerfacecolor="#333333",
            markersize=8,
            label="Difficulty mean",
        ),
    ]
    axes[-1].legend(
        handles=legend_handles,
        loc="upper right",
        frameon=True,
        fancybox=False,
        edgecolor="#cccccc",
        fontsize=9,
    )

    # No overall figure title
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    try:
        df = load_final_18()
        means = difficulty_means(df)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    print(f"Loaded {len(df)} evaluation groups from {EVAL_ROOT}")
    print("\nMean metrics by difficulty (average of 6 experiments each):")
    display = means.copy()
    display.index = [d.capitalize() for d in display.index]
    display.columns = [
        "Cut-ins / ep",
        "Lane-changes / ep",
        "Episode length",
    ]
    print(display.to_string(float_format=lambda x: f"{x:.4f}"))

    make_figure(df, means, OUT_PATH)
    print(f"\nSaved figure → {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
