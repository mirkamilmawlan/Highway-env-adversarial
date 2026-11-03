#!/usr/bin/env python3
"""
Generate dual heatmaps (median min TTC / median min distance) from the final
18 evaluation summary CSVs for defense PPT slides.

Reads evaluation_results/{IDM,DQN,PPO}-Ego/{easy,medium,hard}/lane{2,3}_seed17/
summary_results.csv → median_ttc_distance_heatmaps.png
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
OUT_PATH = PROJECT_ROOT / "median_ttc_distance_heatmaps.png"

# Final experimental matrix (3 ego × 3 difficulty × 2 lanes = 18)
EGO_MODELS = ("IDM-Ego", "DQN-Ego", "PPO-Ego")
DIFFICULTIES = ("easy", "medium", "hard")
LANE_COUNTS = (2, 3)
TRAINING_SEED = 17

REQUIRED_COLUMNS = (
    "ego_model",
    "difficulty",
    "lane_count",
    "min_ttc_median",
    "min_distance_median",
)

# Display row order (top → bottom)
ROW_ORDER = [
    ("IDM-Ego", 2),
    ("DQN-Ego", 2),
    ("PPO-Ego", 2),
    ("IDM-Ego", 3),
    ("DQN-Ego", 3),
    ("PPO-Ego", 3),
]
ROW_LABELS = [
    "IDM · Lane2",
    "DQN · Lane2",
    "PPO · Lane2",
    "IDM · Lane3",
    "DQN · Lane3",
    "PPO · Lane3",
]
COL_LABELS = ["Easy", "Medium", "Hard"]


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

                ttc = float(row["min_ttc_median"])
                dist = float(row["min_distance_median"])
                if not np.isfinite(ttc):
                    raise ValueError(f"{path}: min_ttc_median is not finite ({ttc})")
                if not np.isfinite(dist):
                    raise ValueError(
                        f"{path}: min_distance_median is not finite ({dist})"
                    )
                if ttc < 0:
                    raise ValueError(f"{path}: min_ttc_median < 0 ({ttc})")
                if dist < 0:
                    raise ValueError(f"{path}: min_distance_median < 0 ({dist})")

                rows.append(
                    {
                        "ego_model": ego,
                        "difficulty": difficulty,
                        "lane_count": lane,
                        "min_ttc_median": ttc,
                        "min_distance_median": dist,
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


def build_matrix(df: pd.DataFrame, value_col: str) -> np.ndarray:
    """Return shape (6, 3) array in the required row/column order."""
    mat = np.full((len(ROW_ORDER), len(DIFFICULTIES)), np.nan, dtype=float)
    for i, (ego, lane) in enumerate(ROW_ORDER):
        for j, difficulty in enumerate(DIFFICULTIES):
            match = df[
                (df["ego_model"] == ego)
                & (df["lane_count"] == lane)
                & (df["difficulty"] == difficulty)
            ]
            if len(match) != 1:
                raise RuntimeError(
                    f"Expected 1 row for {ego} lane{lane} {difficulty}, "
                    f"got {len(match)}"
                )
            mat[i, j] = float(match.iloc[0][value_col])
    if np.isnan(mat).any():
        raise RuntimeError(f"Matrix for {value_col} still contains NaN")
    return mat


def _cell_text_color(value: float, vmin: float, vmax: float) -> str:
    """White text on dark cells, dark text on light cells."""
    if vmax <= vmin:
        return "#111111"
    # With *_r colormaps, low values map near 0 (dark)
    norm = (value - vmin) / (vmax - vmin)
    return "#ffffff" if norm < 0.45 else "#111111"


def draw_heatmap(
    ax: plt.Axes,
    data: np.ndarray,
    *,
    title: str,
    fmt: str,
    cmap: str,
    cbar_label: str,
) -> None:
    vmin = float(np.min(data))
    vmax = float(np.max(data))
    im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)

    ax.set_xticks(np.arange(len(COL_LABELS)))
    ax.set_yticks(np.arange(len(ROW_LABELS)))
    ax.set_xticklabels(COL_LABELS)
    ax.set_yticklabels(ROW_LABELS)
    ax.tick_params(top=False, bottom=True, labeltop=False, labelbottom=True)
    ax.set_xticks(np.arange(len(COL_LABELS) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(ROW_LABELS) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            ax.text(
                j,
                i,
                format(val, fmt),
                ha="center",
                va="center",
                color=_cell_text_color(val, vmin, vmax),
                fontsize=11,
                fontweight="medium",
            )

    ax.set_title(title, fontsize=13, pad=10)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label, fontsize=11)
    cbar.ax.tick_params(labelsize=10)


def make_heatmaps(df: pd.DataFrame, out_path: Path) -> None:
    ttc = build_matrix(df, "min_ttc_median")
    dist = build_matrix(df, "min_distance_median")

    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.titlesize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )

    # *_r → low values map to dark end of the sequential colormap
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.8), dpi=150)

    draw_heatmap(
        axes[0],
        ttc,
        title="Median Minimum TTC (s)",
        fmt=".3f",
        cmap="YlOrRd_r",
        cbar_label="seconds",
    )
    draw_heatmap(
        axes[1],
        dist,
        title="Median Minimum Distance (m)",
        fmt=".2f",
        cmap="YlOrRd_r",
        cbar_label="metres",
    )

    # No overall figure title
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    try:
        df = load_final_18()
        ttc = build_matrix(df, "min_ttc_median")
        dist = build_matrix(df, "min_distance_median")
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    print(f"Loaded {len(df)} evaluation groups from {EVAL_ROOT}")
    print("\nMedian Minimum TTC (s):")
    print(
        pd.DataFrame(ttc, index=ROW_LABELS, columns=COL_LABELS)
        .map(lambda x: f"{x:.3f}")
        .to_string()
    )
    print("\nMedian Minimum Distance (m):")
    print(
        pd.DataFrame(dist, index=ROW_LABELS, columns=COL_LABELS)
        .map(lambda x: f"{x:.2f}")
        .to_string()
    )

    make_heatmaps(df, OUT_PATH)
    print(f"\nSaved figure → {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
