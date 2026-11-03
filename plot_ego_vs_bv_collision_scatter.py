#!/usr/bin/env python3
"""
Generate a defense-slide scatter plot from the final 18 evaluation summaries.

Reads evaluation_results/{IDM,DQN,PPO}-Ego/{easy,medium,hard}/lane{2,3}_seed17/
summary_results.csv and writes ego_vs_bv_collision_scatter.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
EVAL_ROOT = PROJECT_ROOT / "evaluation_results"
OUT_PATH = PROJECT_ROOT / "ego_vs_bv_collision_scatter.png"

# Final experimental matrix (3 ego × 3 difficulty × 2 lanes = 18)
EGO_MODELS = ("IDM-Ego", "DQN-Ego", "PPO-Ego")
DIFFICULTIES = ("easy", "medium", "hard")
LANE_COUNTS = (2, 3)
TRAINING_SEED = 17

REQUIRED_COLUMNS = (
    "ego_model",
    "difficulty",
    "lane_count",
    "ego_collision_rate",
    "bv_collision_rate",
)

# Color by Ego type; marker by lane count
EGO_COLORS = {
    "IDM-Ego": "#1f77b4",  # blue
    "DQN-Ego": "#d62728",  # red
    "PPO-Ego": "#2ca02c",  # green
}
LANE_MARKERS = {
    2: "o",  # circle
    3: "s",  # square
}

def expected_csv_path(ego: str, difficulty: str, lane: int) -> Path:
    return (
        EVAL_ROOT
        / ego
        / difficulty
        / f"lane{lane}_seed{TRAINING_SEED}"
        / "summary_results.csv"
    )


def rate_to_percent(value: float, column: str, source: Path) -> float:
    """Convert collision rate in [0, 1] to percent; reject already-% values."""
    if not (0.0 <= value <= 1.0):
        raise ValueError(
            f"{source}: {column}={value} is outside [0, 1]. "
            "Expected a fraction (e.g. 0.18 → 18%), not a percentage."
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
                # Consistency checks against path metadata
                if str(row["ego_model"]) != ego:
                    raise ValueError(
                        f"{path}: ego_model column is '{row['ego_model']}', "
                        f"expected '{ego}'"
                    )
                if str(row["difficulty"]).lower() != difficulty:
                    raise ValueError(
                        f"{path}: difficulty column is '{row['difficulty']}', "
                        f"expected '{difficulty}'"
                    )
                if int(row["lane_count"]) != lane:
                    raise ValueError(
                        f"{path}: lane_count column is {row['lane_count']}, "
                        f"expected {lane}"
                    )

                ego_pct = rate_to_percent(
                    float(row["ego_collision_rate"]), "ego_collision_rate", path
                )
                bv_pct = rate_to_percent(
                    float(row["bv_collision_rate"]), "bv_collision_rate", path
                )

                rows.append(
                    {
                        "ego_model": ego,
                        "difficulty": difficulty,
                        "lane_count": lane,
                        "ego_collision_pct": ego_pct,
                        "bv_collision_pct": bv_pct,
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


def make_scatter(df: pd.DataFrame, out_path: Path) -> None:
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.labelsize": 14,
            "legend.fontsize": 11,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )

    fig, ax = plt.subplots(figsize=(8.5, 6.5), dpi=150)

    # Plot each (ego, lane) series separately for a clean legend
    for ego in EGO_MODELS:
        for lane in LANE_COUNTS:
            subset = df[(df["ego_model"] == ego) & (df["lane_count"] == lane)]
            if subset.empty:
                continue
            ax.scatter(
                subset["bv_collision_pct"],
                subset["ego_collision_pct"],
                c=EGO_COLORS[ego],
                marker=LANE_MARKERS[lane],
                s=110,
                edgecolors="white",
                linewidths=0.8,
                zorder=3,
                label=f"{ego.replace('-Ego', '')} · Lane{lane}",
            )

    ax.set_xlabel("BV Collision Rate (%)")
    ax.set_ylabel("Ego Collision Rate (%)")
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 100)
    ax.grid(True, which="major", color="#d0d0d0", linestyle="-", linewidth=0.8)
    ax.set_axisbelow(True)

    # Legend: color (Ego) + marker (Lane) via the series labels above
    ax.legend(
        loc="lower right",
        frameon=True,
        fancybox=False,
        edgecolor="#cccccc",
        framealpha=1.0,
        title="Ego · Lane",
        title_fontsize=11,
    )

    # No main title (defense slide will supply its own)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    try:
        df = load_final_18()
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    print(f"Loaded {len(df)} evaluation groups from {EVAL_ROOT}")
    print(
        df[
            [
                "ego_model",
                "difficulty",
                "lane_count",
                "bv_collision_pct",
                "ego_collision_pct",
            ]
        ]
        .sort_values(["ego_model", "lane_count", "difficulty"])
        .to_string(index=False)
    )

    # Range sanity for axes choice
    bv_max = df["bv_collision_pct"].max()
    ego_max = df["ego_collision_pct"].max()
    if bv_max > 20:
        print(
            f"[WARN] max BV collision rate is {bv_max:.1f}% > 20% axis limit; "
            "some points may be clipped.",
            file=sys.stderr,
        )
    if ego_max > 100:
        print(
            f"[WARN] max Ego collision rate is {ego_max:.1f}% > 100% axis limit.",
            file=sys.stderr,
        )

    make_scatter(df, OUT_PATH)
    print(f"\nSaved figure → {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
