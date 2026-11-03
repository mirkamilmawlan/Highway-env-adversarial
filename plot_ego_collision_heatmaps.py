#!/usr/bin/env python3
"""
Generate Ego collision-rate heatmaps (Lane2 / Lane3) from the final 18
evaluation summary CSVs.

Reads evaluation_results/{IDM,DQN,PPO}-Ego/{easy,medium,hard}/lane{2,3}_seed17/
summary_results.csv →
  ego_collision_lane2_heatmap.png
  ego_collision_lane3_heatmap.png
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
OUT_LANE2 = PROJECT_ROOT / "ego_collision_lane2_heatmap.png"
OUT_LANE3 = PROJECT_ROOT / "ego_collision_lane3_heatmap.png"

EGO_MODELS = ("IDM-Ego", "DQN-Ego", "PPO-Ego")
DIFFICULTIES = ("easy", "medium", "hard")
LANE_COUNTS = (2, 3)
TRAINING_SEED = 17

REQUIRED_COLUMNS = (
    "ego_model",
    "difficulty",
    "lane_count",
    "ego_collision_rate",
)

ROW_LABELS = ["IDM-Ego", "DQN-Ego", "PPO-Ego"]
COL_LABELS = ["Easy", "Medium", "Hard"]


def expected_csv_path(ego: str, difficulty: str, lane: int) -> Path:
    return (
        EVAL_ROOT
        / ego
        / difficulty
        / f"lane{lane}_seed{TRAINING_SEED}"
        / "summary_results.csv"
    )


def rate_to_percent(value: float, source: Path) -> float:
    if not (0.0 <= value <= 1.0):
        raise ValueError(
            f"{source}: ego_collision_rate={value} outside [0, 1]; "
            "expected a fraction (e.g. 0.86 → 86%)."
        )
    return float(value) * 100.0


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

                rows.append(
                    {
                        "ego_model": ego,
                        "difficulty": difficulty,
                        "lane_count": lane,
                        "ego_collision_pct": rate_to_percent(
                            float(row["ego_collision_rate"]), path
                        ),
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


def build_lane_matrix(df: pd.DataFrame, lane: int) -> np.ndarray:
    """Return shape (3, 3): rows=Ego, cols=Easy/Medium/Hard, values in %."""
    mat = np.full((len(EGO_MODELS), len(DIFFICULTIES)), np.nan, dtype=float)
    for i, ego in enumerate(EGO_MODELS):
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
            mat[i, j] = float(match.iloc[0]["ego_collision_pct"])
    if np.isnan(mat).any():
        raise RuntimeError(f"Lane{lane} collision matrix still contains NaN")
    return mat


def _cell_text_color(value: float, vmin: float, vmax: float) -> str:
    """Dark text on yellow/light cells; white text on deep red cells."""
    if vmax <= vmin:
        return "#111111"
    # YlOrRd: high values → dark red
    norm = (value - vmin) / (vmax - vmin)
    return "#ffffff" if norm > 0.55 else "#111111"


def save_heatmap(data: np.ndarray, *, title: str, out_path: Path) -> None:
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

    # Shared 0–100% scale so Lane2 / Lane3 colours are comparable
    vmin, vmax = 0.0, 100.0

    fig, ax = plt.subplots(figsize=(6.2, 4.2), dpi=150)
    im = ax.imshow(data, aspect="auto", cmap="YlOrRd", vmin=vmin, vmax=vmax)

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
                f"{val:.0f}%",
                ha="center",
                va="center",
                color=_cell_text_color(val, vmin, vmax),
                fontsize=12,
                fontweight="medium",
            )

    ax.set_title(title, fontsize=13, pad=10)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Ego Collision Rate (%)", fontsize=11)
    cbar.ax.tick_params(labelsize=10)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    try:
        df = load_final_18()
        mat2 = build_lane_matrix(df, 2)
        mat3 = build_lane_matrix(df, 3)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    print(f"Loaded {len(df)} evaluation groups from {EVAL_ROOT}")
    print("\nLane2 Ego collision rate (%):")
    print(
        pd.DataFrame(mat2, index=ROW_LABELS, columns=COL_LABELS)
        .map(lambda x: f"{x:.0f}%")
        .to_string()
    )
    print("\nLane3 Ego collision rate (%):")
    print(
        pd.DataFrame(mat3, index=ROW_LABELS, columns=COL_LABELS)
        .map(lambda x: f"{x:.0f}%")
        .to_string()
    )

    save_heatmap(
        mat2,
        title="Ego Collision Rate — Lane 2",
        out_path=OUT_LANE2,
    )
    save_heatmap(
        mat3,
        title="Ego Collision Rate — Lane 3",
        out_path=OUT_LANE3,
    )
    print(f"\nSaved → {OUT_LANE2}")
    print(f"Saved → {OUT_LANE3}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
