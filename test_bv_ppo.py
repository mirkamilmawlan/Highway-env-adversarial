# -*- coding: utf-8 -*-
"""
Test a trained PPO adversarial BV model.

Example
-------
  # 2-lane IDM-Ego, medium model, record videos
  python test_bv_ppo.py --ego IDM-Ego --risk_level medium --lane_count 2 --seed 17 --render

  # Use a specific checkpoint
  python test_bv_ppo.py --model_path PPO_BV_models/IDM-Ego/medium/lane2_seed17_300000_steps --render
"""

import argparse
import os
import warnings

from gymnasium.wrappers import RecordVideo
from stable_baselines3 import PPO

from train_bv_ppo import AdvBVEnv

warnings.filterwarnings("ignore", category=UserWarning)


def main() -> None:
    parser = argparse.ArgumentParser(description="Test PPO adversarial BV model")
    parser.add_argument("--ego", type=str, default="IDM-Ego",
                        choices=["IDM-Ego", "DQN-Ego", "A2C-Ego", "PPO-Ego"])
    parser.add_argument("--risk_level", type=str, default="medium",
                        choices=["easy", "medium", "hard"])
    parser.add_argument("--lane_count", type=int, default=2)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--episodes", type=int, default=10,
                        help="Number of test episodes")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to .zip model (without extension). "
                             "Default: PPO_BV_models/{ego}/{risk_level}/lane{N}_seed{S}")
    parser.add_argument("--render", action="store_true",
                        help="Record videos to PPO_BV_models/.../videos/")
    args = parser.parse_args()

    if args.model_path is None:
        args.model_path = (
            f"PPO_BV_models/{args.ego}/{args.risk_level}"
            f"/lane{args.lane_count}_seed{args.seed}"
        )

    # Always load the .zip file explicitly (avoid clash with a same-named folder).
    model_zip = args.model_path if args.model_path.endswith(".zip") else f"{args.model_path}.zip"
    if not os.path.isfile(model_zip):
        raise FileNotFoundError(f"Model not found: {model_zip}")

    print(f"\n{'='*60}")
    print(f"  PPO BV Test")
    print(f"  Model     : {model_zip}")
    print(f"  Ego       : {args.ego}")
    print(f"  Lanes     : {args.lane_count}")
    print(f"  Episodes  : {args.episodes}")
    print(f"  Render    : {args.render}")
    print(f"{'='*60}\n")

    env = AdvBVEnv(
        ego_name=args.ego,
        risk_level=args.risk_level,
        lane_count=args.lane_count,
        seed=args.seed,
    )

    if args.render:
        # Use a separate "videos/" sub-folder so we never create a directory
        # with the same name as the model .zip (e.g. lane2_seed17/).
        video_dir = os.path.join(
            os.path.dirname(model_zip),
            "videos",
            f"lane{args.lane_count}_seed{args.seed}",
        )
        os.makedirs(video_dir, exist_ok=True)
        env = RecordVideo(env, video_folder=video_dir, episode_trigger=lambda e: True)
        inner = env.unwrapped._inner_env.unwrapped
        inner.set_record_video_wrapper(env)
        inner.update_metadata(video_real_time_ratio=1)
        print(f"Videos will be saved to: {video_dir}\n")

    model = PPO.load(model_zip, env=env)

    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        done = False
        total_reward = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            total_reward += reward
            done = terminated or truncated
            if args.render:
                env.render()

        print(
            f"Episode {ep}: reward={total_reward:.2f}  "
            f"ego_collision={info.get('ego_collision', 0)}  "
            f"bv_collision={info.get('bv_collision', 0)}  "
            f"min_ttc={info.get('min_ttc', -1):.4f}  "
            f"min_distance={info.get('min_distance', -1):.2f}m  "
            f"steps={info.get('episode_length', 0)}"
        )

    env.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
