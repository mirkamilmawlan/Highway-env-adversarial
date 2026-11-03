# Highway-env Adversarial

Adversarial testing for autonomous driving on top of [highway-env](https://github.com/Farama-Foundation/HighwayEnv). A reinforcement-learning (RL) **background vehicle (BV / CBV)** is trained to attack a fixed **ego** policy, so you can measure safety under dangerous interactions—not only calm traffic.

Green = Ego · Yellow = selected adversarial BV · Blue = normal IDM traffic

---

## What this project does

In standard highway simulation, background traffic is often passive. Here, the nearest vehicle ahead of the ego can become an **AdvVehicle** and learn to:

- brake hard / tailgate
- cut in
- force lane changes or collisions

| Role | Control | Notes |
|------|---------|--------|
| **Ego** | Fixed policy | IDM rule-based, or pretrained DQN / A2C / PPO |
| **Adversarial BV** | Rainbow DQN or PPO | Learns risky interactions |
| **Other BVs** | IDM | Ordinary traffic flow |

Difficulty levels (`easy` / `medium` / `hard`) reshape reward / risk terms so you can compare ego collision rate, TTC, lane changes, and cut-ins under stronger attacks.

---

## Demo animations (GIF)

GIFs live in [`image/`](image/).

### DQN-Ego

| 2 lanes · frontal pressure | 2 lanes · lateral crash |
| :---: | :---: |
| ![brake](image/DQN-Ego-2lanes-brake.gif) | ![crash](image/DQN-Ego-2lanes-crashed.gif) |
| Hard brake / pressure from the CBV | Lateral collision |

| 3 lanes · emergency lane change | 3 lanes · cut-in |
| :---: | :---: |
| ![cut1](image/DQN-Ego-3lanes-cut-in.gif) | ![cut2](image/DQN-Ego-3lanes-cut-in2.gif) |

### IDM-Ego

| 2 lanes · crash | 2 lanes · emergency lane change |
| :---: | :---: |
| ![idm-crash](image/IDM-Ego-2lanes-crashed.gif) | ![idm-cut](image/IDM-Ego-2lanes-cut-in.gif) |

| 3 lanes · cut-in | 3 lanes · surrounding crash |
| :---: | :---: |
| ![idm-cut2](image/IDM-Ego-3lanes-cut-in2.gif) | ![idm-surr](image/IDM-Ego-3lanes-surroundcrash.gif) |

### Evaluation clips (MP4)

Selected episodes in [`docs/demo/`](docs/demo/) (download to play):

| File | Scenario |
|------|----------|
| [dqn_hard_lane2_ep0.mp4](docs/demo/dqn_hard_lane2_ep0.mp4) | DQN-Ego · hard · 2 lanes |
| [dqn_hard_lane3_ep0.mp4](docs/demo/dqn_hard_lane3_ep0.mp4) | DQN-Ego · hard · 3 lanes |
| [dqn_easy_lane2_ep0.mp4](docs/demo/dqn_easy_lane2_ep0.mp4) | DQN-Ego · easy · 2 lanes |
| [idm_hard_lane2_ep0.mp4](docs/demo/idm_hard_lane2_ep0.mp4) | IDM-Ego · hard · 2 lanes |
| [ppo_hard_lane2_ep0.mp4](docs/demo/ppo_hard_lane2_ep0.mp4) | PPO-Ego · hard · 2 lanes |

More raw recordings: `PPO_BV_models/*/videos/`.

---

## Evaluation figures

Experiment matrix: **3 ego policies × 3 difficulties × 2/3 lanes**, ~100 episodes per cell (`evaluation_results/`).

### Ego collision-rate heatmaps

Stronger adversarial BVs tend to raise ego collision rates. Lane 2:

![Ego collision rate lane 2](docs/figures/ego_collision_lane2_heatmap.png)

Lane 3:

![Ego collision rate lane 3](docs/figures/ego_collision_lane3_heatmap.png)

### Behaviour metrics vs difficulty

Higher difficulty → more cut-ins and lane changes, often shorter episodes (earlier crash / termination):

![Behaviour metrics](docs/figures/behaviour_metrics_by_difficulty.png)

### TTC / distance & Ego–BV collision relationship

![Median TTC / distance](docs/figures/median_ttc_distance_heatmaps.png)

![Ego vs BV collision scatter](docs/figures/ego_vs_bv_collision_scatter.png)

### Rainbow training reward

![Mean reward](image/Mean%20Reward.png)

---

## Repository layout

```
├── Adv_main.py              # Rainbow DQN adversarial BV: train / test
├── train_bv_ppo.py          # PPO adversarial BV training (separate pipeline)
├── evaluate_ppo_bv.py       # Formal PPO BV evaluation (CSV outputs)
├── train_ego_adv.py         # Train ego (PPO/A2C) against adversarial BVs
├── config.py                # Ego paths, action map, env defaults
├── BV_Agent/                # Rainbow DQN implementation
├── Ego_Agent/               # Pretrained ego models + Trainer
├── highway_env/             # Extended env (includes highway-adv-v0)
├── BV_model/                # Rainbow BV checkpoints
├── PPO_BV_models/           # PPO BV checkpoints + evaluation videos
├── evaluation_results/      # Summary / per-episode CSVs
├── image/                   # Demo GIFs + observation sketch
├── docs/figures/            # Plots used in this README
├── docs/demo/               # Selected MP4 clips for this README
└── plot_*.py                # Rebuild heatmaps / behaviour plots from CSV
```

Observation sketch (CBV kinematics):

<p align="center">
  <img src="image/obs.png" width="360" alt="observation example" />
</p>

---

## Setup

**Python 3.10+** recommended; use a virtual environment.

```bash
git clone https://github.com/mirkamilmawlan/Highway-env-adversarial.git
cd Highway-env-adversarial

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install .
pip install "stable-baselines3[extra]"
```

Core deps (see package metadata): `gymnasium`, `numpy`, `pygame`, `matplotlib`, `pandas`, `scipy`, `moviepy`, `tensorboardX`, etc.

---

## Usage

### 1) Train / test Rainbow adversarial BV (original pipeline)

```bash
# Train (default 2 lanes; Ego: DQN-Ego / A2C-Ego / PPO-Ego / IDM-Ego)
python Adv_main.py --Ego=DQN-Ego --train
python Adv_main.py --Ego=DQN-Ego --train --lane_count=3

# Test with rendering
python Adv_main.py --Ego=IDM-Ego --test --render
python Adv_main.py --Ego=IDM-Ego --test --render --lane_count=3

# TensorBoard
tensorboard --logdir=AdvLogs
```

Optionally train your own ego:

```bash
python Ego_Agent/Trainer.py --train --ego_type DQN
```

### 2) Train PPO adversarial BV (new pipeline)

```bash
python train_bv_ppo.py --ego IDM-Ego --risk_level medium --total_timesteps 50000
python train_bv_ppo.py --ego DQN-Ego --risk_level hard --lane_count 3 --seed 17

tensorboard --logdir PPO_BV_logs/
```

### 3) Formal PPO BV evaluation

```bash
# Smoke test (5 episodes)
python evaluate_ppo_bv.py \
  --ego DQN-Ego --difficulty hard --lane-count 2 \
  --training-seed 17 --episodes 5 --eval-seed-start 10000

# Full evaluation (100 episodes)
python evaluate_ppo_bv.py \
  --ego DQN-Ego --difficulty hard --lane-count 2 \
  --training-seed 17 --episodes 100 --eval-seed-start 10000
```

Outputs go to `evaluation_results/{Ego}/{difficulty}/lane{N}_seed{S}/`.

### 4) Rebuild paper / slide figures from CSV

```bash
python plot_ego_collision_heatmaps.py
python plot_behaviour_metrics_by_difficulty.py
python plot_median_ttc_distance_heatmaps.py
python plot_ego_vs_bv_collision_scatter.py
python plot.py   # Rainbow training reward
```

### 5) Train ego under adversarial traffic

```bash
python train_ego_adv.py --help
```

---

## Adversarial BV action space

```python
Bv_Action = {
    0: "LANE_LEFT",
    1: "IDLE",
    2: "LANE_RIGHT",
    3: "FASTER",
    4: "SLOWER",
}
```

Each frame, the closest vehicle ahead of the ego is selected as the **CBV** (yellow) and driven by the RL policy with the discrete actions above.

---

## Acknowledgements

- Built on Farama **highway-env**.
- Original adversarial env extension: [CurryChen77/Highway-env-adversarial](https://github.com/CurryChen77/Highway-env-adversarial).
- This fork adds a **PPO BV train/eval pipeline**, a **multi-difficulty experiment matrix**, collision / behaviour analysis plots, and curated demo media.

---

## License

MIT (same as the upstream package declaration).
