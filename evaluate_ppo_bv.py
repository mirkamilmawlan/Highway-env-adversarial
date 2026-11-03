# -*- coding: utf-8 -*-
"""
Formal evaluation of a trained PPO adversarial BV (no training, no learn()).

Reuses AdvBVEnv from train_bv_ppo.py so that reward / TTC / distance / cut-in /
collision metrics match the training environment exactly. Only the evaluation
episode seeds differ from training.

Example
-------
  # Smoke test (5 episodes)
  python evaluate_ppo_bv.py \\
    --ego DQN-Ego --difficulty hard --lane-count 2 \\
    --training-seed 17 --episodes 5 --eval-seed-start 10000

  # Full evaluation (100 episodes, seeds 10000..10099)
  python evaluate_ppo_bv.py \\
    --ego DQN-Ego --difficulty hard --lane-count 2 \\
    --training-seed 17 --episodes 100 --eval-seed-start 10000
"""

from __future__ import annotations

import argparse
import csv
import os
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from stable_baselines3 import PPO

from train_bv_ppo import AdvBVEnv

warnings.filterwarnings("ignore", category=UserWarning)

# Columns written to per_episode_results.csv (exact names requested)
PER_EPISODE_FIELDS = [
    "ego_model",
    "difficulty",
    "lane_count",
    "training_seed",
    "evaluation_seed",
    "episode_reward",
    "ego_collision",
    "bv_collision",
    "min_ttc",
    "min_distance",
    "cut_in_count",
    "lane_change_count",
    "episode_length",
]


def resolve_model_zip(
    ego: str,
    difficulty: str,
    lane_count: int,
    training_seed: int,
    model_path: Optional[str],
) -> str:
    """Return path to the .zip file of the trained PPO-BV model."""
    if model_path is None:
        model_path = (
            f"PPO_BV_models/{ego}/{difficulty}"
            f"/lane{lane_count}_seed{training_seed}"
        )
    zip_path = model_path if model_path.endswith(".zip") else f"{model_path}.zip"
    if not os.path.isfile(zip_path):
        raise FileNotFoundError(f"Model not found: {zip_path}")
    return zip_path


def run_episode(
    env: AdvBVEnv,
    model: PPO,
    evaluation_seed: int,
) -> Dict[str, Any]:
    """
    Run one deterministic evaluation episode.

    Metrics (episode_reward, ego_collision, bv_collision, min_ttc, min_distance,
    cut_in_count, lane_change_count, episode_length) are taken from AdvBVEnv.info
    at episode end — the same fields populated by AdvBVEnv.step().
    """
    obs, _ = env.reset(seed=evaluation_seed)
    done = False
    info: Dict[str, Any] = {}

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _reward, terminated, truncated, info = env.step(int(action))
        done = bool(terminated or truncated)

    # Prefer AdvBVEnv's cumulative episode_reward (identical to Monitor return).
    return {
        "episode_reward": float(info.get("episode_reward", 0.0)),
        "ego_collision": int(info.get("ego_collision", 0)),
        "bv_collision": int(info.get("bv_collision", 0)),
        "min_ttc": float(info.get("min_ttc", -1.0)),
        "min_distance": float(info.get("min_distance", float("nan"))),
        "cut_in_count": int(info.get("cut_in_count", 0)),
        "lane_change_count": int(info.get("lane_change_count", 0)),
        "episode_length": int(info.get("episode_length", 0)),
    }


def write_per_episode_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=PER_EPISODE_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in PER_EPISODE_FIELDS})


def _mean(xs: List[float]) -> float:
    return float(np.mean(xs)) if xs else float("nan")


def _std(xs: List[float]) -> float:
    return float(np.std(xs, ddof=0)) if xs else float("nan")


def _median(xs: List[float]) -> float:
    return float(np.median(xs)) if xs else float("nan")


def build_summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate per-episode rows into summary statistics."""
    n = len(rows)
    rewards = [r["episode_reward"] for r in rows]
    ego_cols = [r["ego_collision"] for r in rows]
    bv_cols = [r["bv_collision"] for r in rows]
    # Exclude sentinel -1.0 (no finite TTC in that episode) from TTC stats.
    valid_ttc = [r["min_ttc"] for r in rows if r["min_ttc"] >= 0.0]
    distances = [r["min_distance"] for r in rows]
    cut_ins = [r["cut_in_count"] for r in rows]
    lane_changes = [r["lane_change_count"] for r in rows]
    lengths = [r["episode_length"] for r in rows]

    first = rows[0] if rows else {}
    return {
        "ego_model": first.get("ego_model", ""),
        "difficulty": first.get("difficulty", ""),
        "lane_count": first.get("lane_count", ""),
        "training_seed": first.get("training_seed", ""),
        "n_episodes": n,
        "eval_seed_start": first.get("evaluation_seed", ""),
        "eval_seed_end": rows[-1]["evaluation_seed"] if rows else "",
        "ego_collision_rate": float(np.mean(ego_cols)) if n else float("nan"),
        "bv_collision_rate": float(np.mean(bv_cols)) if n else float("nan"),
        "reward_mean": _mean(rewards),
        "reward_std": _std(rewards),
        "min_ttc_mean": _mean(valid_ttc),
        "min_ttc_median": _median(valid_ttc),
        "min_ttc_valid_count": len(valid_ttc),
        "min_distance_mean": _mean(distances),
        "min_distance_median": _median(distances),
        "cut_in_count_mean": _mean(cut_ins),
        "lane_change_count_mean": _mean(lane_changes),
        "episode_length_mean": _mean(lengths),
    }


def write_summary_csv(path: Path, summary: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(summary.keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(summary)


def evaluate(args: argparse.Namespace) -> None:
    model_zip = resolve_model_zip(
        ego=args.ego,
        difficulty=args.difficulty,
        lane_count=args.lane_count,
        training_seed=args.training_seed,
        model_path=args.model_path,
    )

    out_dir = Path(args.output_dir) / args.ego / args.difficulty / (
        f"lane{args.lane_count}_seed{args.training_seed}"
    )
    per_ep_path = out_dir / "per_episode_results.csv"
    summary_path = out_dir / "summary_results.csv"

    print(f"\n{'=' * 60}")
    print("  PPO-BV Formal Evaluation (deterministic, no learn)")
    print(f"  Model           : {model_zip}")
    print(f"  Ego             : {args.ego}")
    print(f"  Difficulty      : {args.difficulty}")
    print(f"  Lane count      : {args.lane_count}")
    print(f"  Training seed   : {args.training_seed}")
    print(f"  Episodes        : {args.episodes}")
    print(
        f"  Eval seeds      : {args.eval_seed_start}"
        f" .. {args.eval_seed_start + args.episodes - 1}"
    )
    print(f"  Output dir      : {out_dir}")
    print(f"{'=' * 60}\n")

    # Same AdvBVEnv / risk_level / lane_count as training; only episode seeds change.
    env = AdvBVEnv(
        ego_name=args.ego,
        risk_level=args.difficulty,
        lane_count=args.lane_count,
        seed=args.training_seed,
    )

    # Load for inference only — never call model.learn().
    model = PPO.load(model_zip, env=env)
    model_mtime_before = os.path.getmtime(model_zip)

    rows: List[Dict[str, Any]] = []
    for i in range(args.episodes):
        eval_seed = args.eval_seed_start + i
        metrics = run_episode(env, model, evaluation_seed=eval_seed)
        row = {
            "ego_model": args.ego,
            "difficulty": args.difficulty,
            "lane_count": args.lane_count,
            "training_seed": args.training_seed,
            "evaluation_seed": eval_seed,
            **metrics,
        }
        rows.append(row)
        print(
            f"[{i + 1:03d}/{args.episodes}] seed={eval_seed}  "
            f"reward={metrics['episode_reward']:.2f}  "
            f"ego_col={metrics['ego_collision']}  "
            f"bv_col={metrics['bv_collision']}  "
            f"min_ttc={metrics['min_ttc']:.4f}  "
            f"min_d={metrics['min_distance']:.2f}  "
            f"len={metrics['episode_length']}"
        )

    env.close()

    model_mtime_after = os.path.getmtime(model_zip)
    if model_mtime_after != model_mtime_before:
        raise RuntimeError(
            f"Model file was modified during evaluation: {model_zip}. "
            "Evaluation must not update the model."
        )

    write_per_episode_csv(per_ep_path, rows)
    summary = build_summary(rows)
    write_summary_csv(summary_path, summary)

    print(f"\nWrote {len(rows)} rows → {per_ep_path}")
    print(f"Wrote summary     → {summary_path}")
    print(
        f"ego_collision_rate={summary['ego_collision_rate']:.3f}  "
        f"bv_collision_rate={summary['bv_collision_rate']:.3f}  "
        f"reward_mean={summary['reward_mean']:.2f}±{summary['reward_std']:.2f}"
    )
    print("Model file unchanged (no learn / no overwrite).")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained PPO-BV model and export CSV metrics."
    )
    parser.add_argument(
        "--ego",
        type=str,
        default="DQN-Ego",
        choices=["IDM-Ego", "DQN-Ego", "A2C-Ego", "PPO-Ego"],
        help="Frozen Ego opponent used during BV training / evaluation.",
    )
    parser.add_argument(
        "--difficulty",
        type=str,
        default="hard",
        choices=["easy", "medium", "hard"],
        help="Reward difficulty (same as train_bv_ppo.py --risk_level).",
    )
    parser.add_argument(
        "--lane-count",
        type=int,
        default=2,
        dest="lane_count",
        help="Number of highway lanes (must match the trained model).",
    )
    parser.add_argument(
        "--training-seed",
        type=int,
        default=17,
        dest="training_seed",
        help="Seed used when the BV model was trained (selects model file).",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Number of evaluation episodes (default 100).",
    )
    parser.add_argument(
        "--eval-seed-start",
        type=int,
        default=10000,
        dest="eval_seed_start",
        help="First evaluation seed; episodes use start .. start+episodes-1.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        dest="model_path",
        help=(
            "Optional path to .zip (with or without extension). "
            "Default: PPO_BV_models/{ego}/{difficulty}/lane{N}_seed{S}.zip"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="evaluation_results",
        dest="output_dir",
        help="Root directory for CSV outputs.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
