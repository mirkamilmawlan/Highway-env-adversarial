# -*- coding: utf-8 -*-
"""
PPO-based adversarial BV (Background Vehicle) training pipeline.

This script is a NEW, SEPARATE addition to the project.
It does NOT modify or replace Adv_main.py / BV_Agent/Rainbow.py.
The original Rainbow DQN pipeline continues to work unchanged.

Architecture overview
─────────────────────
Inner env  : highway-adv-v0  (HighwayEnvAdv)
             Configured identically to Adv_main.py so the scenario is the same.
Ego        : Fixed during BV training.
             IDM-Ego  → acts autonomously via the IDM rule model (no RL model loaded).
             DQN/A2C/PPO-Ego → pre-trained SB3 model loaded from Ego_Agent/.
BV (agent) : The "selected BV" (controlled_vehicles[1]) is the PPO learner.
             Observation : CbvKinematicObservation  → shape (5,5) → flattened (25,)
             Actions     : Discrete(5) = LANE_LEFT/IDLE/LANE_RIGHT/FASTER/SLOWER

Usage examples
──────────────
  # Train against IDM ego, medium difficulty, 2 lanes, 50 000 steps
  python train_bv_ppo.py --ego IDM-Ego --risk_level medium --total_timesteps 50000

  # Hard mode against DQN ego, 3 lanes, custom seed
  python train_bv_ppo.py --ego DQN-Ego --risk_level hard --lane_count 3 --seed 42

  # Monitor with TensorBoard
  tensorboard --logdir PPO_BV_logs/
"""

import argparse
import os
import random
import warnings

import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

warnings.filterwarnings("ignore", category=UserWarning)

# Importing VehicleAction also triggers highway_env/__init__.py,
# which registers highway-adv-v0 and all other custom environments.
from highway_env.envs.common.action import VehicleAction  # noqa: E402
from config import Env_config, load_ego_agent              # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Global constants
# ─────────────────────────────────────────────────────────────────────────────

# Must match config.py's Bv_Action mapping exactly.
BV_ACTIONS = {
    0: "LANE_LEFT",
    1: "IDLE",
    2: "LANE_RIGHT",
    3: "FASTER",
    4: "SLOWER",
}

# CbvKinematicObservation default: 5 vehicles × 5 features (presence,x,y,vx,vy)
_OBS_VEHICLES = 5
_OBS_FEATURES = 5
OBS_DIM = _OBS_VEHICLES * _OBS_FEATURES  # 25 — flattened BV observation


# ─────────────────────────────────────────────────────────────────────────────
# Per-difficulty reward configuration
# ─────────────────────────────────────────────────────────────────────────────
# Each key maps to a float weight used inside _compute_reward().
# Increasing a positive weight makes the BV more aggressive for that criterion;
# increasing the magnitude of a negative weight makes it more cautious.

REWARD_CONFIGS = {
    # easy  ──  BV learns basic threat posture; heavily punished for self-crashing
    #           and reckless lane changes.  Low-TTC/proximity incentive is mild.
    "easy": {
        "ego_collision_reward":  5.0,   # bonus when ego crashes
        "bv_crash_penalty":    -15.0,   # penalty when BV self-crashes (ego safe)
        "ttc_scale":             0.05,  # weight on 1/TTC danger bonus
        "distance_scale":        0.05,  # weight on proximity bonus
        "lane_change_penalty":  -0.80,  # 加重变道惩罚，抑制疯狂左右变道
        "cut_in_reward":         1.0,   # bonus per detected cut-in on ego lane
        "step_penalty":         -0.05,  # 加大步惩罚，让BV有紧迫感别乱晃
    },
    # medium  ──  Balanced cut-in strategy. Ego collision reward dominates
    #             per-step signals; step_penalty creates urgency so BV can't park.
    #
    # Key changes (v2):
    #   1. ego_collision_reward raised to 20 — must dominate proximity rewards.
    #   2. distance_scale reduced 0.15→0.05 — cut parking exploit, keep weak signal.
    #   3. lane_change_penalty removed — was actively discouraging cut-ins.
    #   4. step_penalty raised -0.005→-0.20 — parking now costs 8 pts over 40 steps.
    #   5. cut_in_reward raised 2→5 — stronger incentive for the key behaviour.
    #   6. approach_reward/useful_lc_reward/slower_far_penalty added (cf. hard mode).
    #   7. timeout_penalty added so a passive episode always finishes with a loss.
    "medium": {
        "ego_collision_reward":  20.0,  # raised — collision must dominate
        "bv_crash_penalty":     -10.0,  # keep
        "ttc_scale":              0.15,  # slightly reduced
        "distance_scale":         0.05,  # 0.15→0.05: weak signal only, no parking
        "lane_change_penalty":    0.0,   # REMOVED — was punishing cut-ins
        "cut_in_reward":          5.0,   # 2→5: stronger incentive
        "step_penalty":          -0.20,  # -0.005→-0.20: urgency; 40 steps = -8 pts
        # Medium versions of hard-mode terms (checked via cfg.get in _compute_reward)
        "useful_lc_reward":       1.0,   # bonus for LC when within 20 m of ego
        "slower_far_penalty":    -0.20,  # penalty: SLOWER action when dist > 30 m
        "approach_reward":        0.10,  # +0.10 each step BV gets closer to ego
        "timeout_penalty":       -10.0,  # end-of-episode penalty if no ego crash
    },
    # hard  ──  Collision-dominant design. BV must cause ego to crash to win;
    #           every other reward is small and dominated by the collision bonus.
    #
    # Key design principles (v3):
    #   1. ego_collision_reward is huge (50) — only way to get a large reward.
    #   2. step_penalty is strong (-0.4/step) — parking / idling is always costly.
    #   3. distance_scale = 0 — removed entirely (was causing parking exploit).
    #   4. adjacent_lane_reward = 0 — removed entirely (same reason).
    #   5. approach_reward: dense guidance only when BV is actually getting closer.
    #   6. timeout_penalty: punish episodes that end without ego collision.
    "hard": {
        "ego_collision_reward":  50.0,  # dominant — only real path to high reward
        "bv_crash_penalty":      -8.0,  # penalise self-crash (still allows risk)
        "ttc_scale":              0.20, # small bonus only at very close range
        "distance_scale":         0.0,  # REMOVED — caused parking exploit
        "lane_change_penalty":    0.0,  # no penalty at all
        "cut_in_reward":         12.0,  # large one-time cut-in bonus
        "step_penalty":          -0.4,  # TIME PRESSURE: parking costs -16 over 40 steps
        # Hard-mode specific (checked via cfg.get in _compute_reward)
        "adjacent_lane_reward":   0.0,  # REMOVED — caused parking exploit
        "ahead_bonus":            0.0,  # REMOVED
        "useful_lc_reward":       2.0,  # bonus for lane change within 20 m of ego
        "slower_far_penalty":    -0.5,  # heavy penalty: SLOWER when dist > 30 m
        # New in v3:
        "approach_reward":        0.20, # +0.20 each step BV gets closer to ego
        "timeout_penalty":       -20.0, # applied at episode end with no ego crash
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Gymnasium wrapper  ——  AdvBVEnv
# ─────────────────────────────────────────────────────────────────────────────

class AdvBVEnv(gym.Env):
    """
    Single-agent Gymnasium wrapper for PPO training of the adversarial BV.

    The inner environment is highway-adv-v0 (HighwayEnvAdv) configured
    identically to Adv_main.py.  Only the BV is the learning agent; the
    Ego vehicle runs its pre-loaded / IDM policy with no gradient updates.

    Observation space : Box(-1, 1, shape=(25,), float32)
        Flattened CbvKinematicObservation of the selected BV.
        Row 0 = BV itself, row 1 = Ego, rows 2-4 = 3 nearest neighbours.
        Each row: [presence, x_rel, y_rel, vx_rel, vy_rel] — normalised to [-1,1].

    Action space : Discrete(5)
        0 LANE_LEFT | 1 IDLE | 2 LANE_RIGHT | 3 FASTER | 4 SLOWER
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        ego_name: str = "IDM-Ego",
        risk_level: str = "medium",
        lane_count: int = 2,
        seed: int = 0,
    ) -> None:
        super().__init__()

        if risk_level not in REWARD_CONFIGS:
            raise ValueError(
                f"risk_level must be one of {list(REWARD_CONFIGS.keys())}, "
                f"got '{risk_level}'"
            )

        self.ego_name   = ego_name
        self.risk_level = risk_level
        self.lane_count = lane_count
        self._init_seed = seed
        self._reward_cfg = REWARD_CONFIGS[risk_level]

        # ── Build inner highway-adv-v0 env ────────────────────────────────
        # Pull default config values from the same Env_config() used in
        # Adv_main.py so the scenario is identical.
        base_cfg = Env_config(ego_name)

        self._inner_env = gym.make("highway-adv-v0", render_mode="rgb_array")
        self.render_mode = "rgb_array"
        self._inner_env.configure({
            # BV and Ego both observed as Kinematics (MultiAgent)
            "observation": {
                "type": "MultiAgentObservation",
                "observation_config": [
                    {"type": "Kinematics"},               # ego's obs
                    {"type": "CbvKinematicObservation"},  # BV's obs (what PPO sees)
                ],
            },
            # ego_type is None for RL egos (MDPVehicle) or a class path for IDM
            "ego_type":          base_cfg["ego_type"],
            "lanes_count":       lane_count,
            "vehicles_count":    base_cfg["vehicle_count"],
            "controlled_vehicles": base_cfg["controlled_vehicle_count"],  # 2
            "duration":          base_cfg["simulation_time"],
            "BV_type":           base_cfg["bv_type"],
            # Ego starts at the bottom lane, selected BV one lane above
            "initial_lane_id":   lane_count - 1,
            "initial_bv_lane_id": lane_count - 2,
            "simulation_frequency": base_cfg["simulation_frequency"],
        })

        # ── Load fixed ego model (None for IDM — it acts via road.act()) ──
        # Ego model weights are frozen; they receive no gradient.
        self._ego_model = load_ego_agent(
            ego_name=ego_name,
            ego_model_path=base_cfg["Ego_model_path"],
        )

        # ── Gymnasium spaces ──────────────────────────────────────────────
        # CbvKinematicObservation normalises all values to [-1, 1].
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(len(BV_ACTIONS))

        # ── Episode-level metric accumulators ─────────────────────────────
        # These are reset at every env.reset() call.
        self._last_ego_obs: np.ndarray = np.zeros((_OBS_VEHICLES, _OBS_FEATURES),
                                                    dtype=np.float32)
        self._reset_metrics()

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _reset_metrics(self) -> None:
        """Zero all per-episode tracking variables."""
        self._ep_reward        = 0.0
        self._ego_collision    = False
        self._bv_collision     = False
        self._min_ttc          = float("inf")
        self._min_distance     = float("inf")
        self._lane_change_count = 0
        self._cut_in_count     = 0
        self._ep_length        = 0
        self._prev_bv_lane: int = 0   # filled properly in reset()
        self._prev_distance: float = float("inf")  # for approach_reward

    def _ego_vehicle(self):
        """Shortcut to the Ego vehicle object inside the inner env."""
        return self._inner_env.unwrapped.controlled_vehicles[0]

    def _bv_vehicle(self):
        """Shortcut to the currently selected BV vehicle inside the inner env."""
        return self._inner_env.unwrapped.controlled_vehicles[1]

    def _get_ego_action(self, ego_obs: np.ndarray):
        """
        Return the ego's action for this step.

        IDM-Ego : returns None.  The IDM vehicle calls its own act() via
                  road.act() inside _simulate(); passing None suppresses the
                  explicit action_type.act() call.

        RL Ego  : runs the loaded SB3 model in deterministic mode and returns
                  the discrete action integer.
        """
        if self._ego_model is None:
            return None
        obs = np.asarray(ego_obs).flatten()
        # 自动适配模型期望的 obs shape：
        #   - 旧 DQN/PPO ego 训练时 obs=(25,)   → 直接使用
        #   - 新 A2C ego 用 Trainer.py 训练，highway-v0 默认返回 (5,5) → 需要 reshape
        expected = self._ego_model.observation_space.shape
        if obs.shape != expected:
            obs = obs.reshape(expected)
        action, _ = self._ego_model.predict(obs, deterministic=True)
        return action

    def _compute_ttc_and_distance(self):
        """
        Compute approximate Time-To-Collision (TTC) and Euclidean distance
        between the BV and the Ego vehicle.

        TTC is estimated along the line connecting their centres:
            closing_speed = dot(v_bv − v_ego,  unit_vec(ego − bv))
            TTC = (distance − half_lengths) / closing_speed   if closing_speed > 0.5

        Returns (ttc, distance) as floats.
        inf TTC means the vehicles are not approaching.
        """
        ego = self._ego_vehicle()
        bv  = self._bv_vehicle()

        ego_pos = np.array(ego.position, dtype=np.float64)
        bv_pos  = np.array(bv.position,  dtype=np.float64)
        ego_vel = np.array(ego.velocity, dtype=np.float64)
        bv_vel  = np.array(bv.velocity,  dtype=np.float64)

        diff     = ego_pos - bv_pos
        distance = float(np.linalg.norm(diff))

        # Closing speed along the line of sight (positive = approaching)
        if distance > 1e-6:
            unit          = diff / distance
            closing_speed = float(np.dot(bv_vel - ego_vel, unit))
        else:
            closing_speed = 0.0

        # Physical gap (subtract half of each vehicle's length)
        gap = max(distance - (ego.LENGTH + bv.LENGTH) * 0.5, 0.1)

        ttc = gap / closing_speed if closing_speed > 0.5 else float("inf")
        return ttc, distance

    def _detect_cut_in(self, bv_lane_before: int, bv_lane_after: int) -> bool:
        """
        Return True when the BV moved INTO the ego's current lane this step
        AND is within 30 m of the ego (distinguishes a far cut-in from a
        nearby threatening one).
        """
        ego_lane = self._ego_vehicle().lane_index[2]
        if bv_lane_before == ego_lane or bv_lane_after != ego_lane:
            return False
        # Check proximity
        _, dist = self._compute_ttc_and_distance()
        return dist < 30.0

    def _compute_reward(
        self,
        action_idx: int,
        ttc: float,
        distance: float,
        cut_in: bool,
        bv_lane_before: int,
        bv_lane_after: int,
        prev_distance: float = float("inf"),
    ) -> float:
        """
        Compute the PPO reward for this step.

        The reward is composed of five terms whose magnitudes scale with
        risk_level (easy < medium < hard).  See REWARD_CONFIGS above.

        Term 1  Collision outcome
            +ego_collision_reward  if ego crashed  (BV succeeded)
            +bv_crash_penalty      if BV crashed but ego did not  (BV failed)

        Term 2  Danger proximity bonus  (only when BV is alive)
            1/TTC bonus   — rewards being on a collision course
            1/distance    — rewards following ego closely  (caps at 30 m)

        Term 3  Excessive lane-change penalty
            Applied every time the BV issues LANE_LEFT or LANE_RIGHT.
            Encourages the BV to change lanes purposefully, not randomly.

        Term 4  Cut-in bonus
            One-off bonus each time the BV cuts into the ego's lane nearby.

        Term 5  Per-step penalty  (easy/medium only)
            Small constant to encourage finishing the episode quickly.
        """
        cfg    = self._reward_cfg
        reward = 0.0

        ego = self._ego_vehicle()
        bv  = self._bv_vehicle()

        # ── Term 1: collision outcome ─────────────────────────────────────
        if ego.crashed:
            reward += cfg["ego_collision_reward"]
        if bv.crashed and not ego.crashed:
            reward += cfg["bv_crash_penalty"]

        # ── Term 2: danger proximity (only meaningful when BV is alive) ───
        if not bv.crashed:
            if ttc < 10.0:
                # 1/TTC grows as vehicles approach; cap at 1/0.5 = 2.0 to
                # avoid numerical explosion at very small TTC values.
                capped_ttc  = max(ttc, 0.5)
                reward     += cfg["ttc_scale"] * (1.0 / capped_ttc)

            if distance < 30.0:
                # Proximity bonus: linearly higher the closer the BV is.
                reward += cfg["distance_scale"] * (30.0 / max(distance, 1.0))

        # ── Term 3: lane-change penalty ───────────────────────────────────
        if action_idx in (0, 2):      # LANE_LEFT or LANE_RIGHT
            reward += cfg["lane_change_penalty"]

        # ── Term 4: cut-in bonus ──────────────────────────────────────────
        if cut_in:
            reward += cfg["cut_in_reward"]

        # ── Term 5: per-step penalty ──────────────────────────────────────
        reward += cfg["step_penalty"]

        # ── Term 6: adjacent-lane flanking reward (hard-mode) ─────────────
        # Reward BV for positioning in the lane beside ego at close range.
        # This teaches the BV to approach laterally (setup for cut-in)
        # rather than simply braking in ego's path.
        adj_lane_reward = cfg.get("adjacent_lane_reward", 0.0)
        if adj_lane_reward != 0.0 and not bv.crashed:
            ego_lane = ego.lane_index[2]
            bv_lane  = bv.lane_index[2]
            if abs(bv_lane - ego_lane) == 1 and distance < 20.0:
                reward += adj_lane_reward
                # Extra bonus when BV is slightly ahead — perfect cut-in angle
                ahead_bonus = cfg.get("ahead_bonus", 0.0)
                if ahead_bonus != 0.0 and bv.position[0] > ego.position[0]:
                    reward += ahead_bonus

        # ── Term 7: useful lane-change reward (hard-mode) ─────────────────
        # A lane change when already close to ego is the actual cut-in move;
        # reward it regardless of whether _detect_cut_in triggers.
        useful_lc = cfg.get("useful_lc_reward", 0.0)
        if useful_lc != 0.0 and action_idx in (0, 2) and distance < 20.0:
            reward += useful_lc

        # ── Term 8: passive-braking penalty (hard-mode) ───────────────────
        # Penalise SLOWER action when BV is far from ego so the BV cannot
        # exploit the "slow down and wait for ego to rear-end me" strategy.
        slower_far = cfg.get("slower_far_penalty", 0.0)
        if slower_far != 0.0 and action_idx == 4 and distance > 30.0:
            reward += slower_far

        # ── Term 9: approach reward (hard-mode) ───────────────────────────
        # Give a small bonus each step BV is actively getting CLOSER to ego.
        # This provides dense reward shaping that guides BV toward ego without
        # rewarding it for simply parking at a fixed distance.
        approach_r = cfg.get("approach_reward", 0.0)
        if approach_r != 0.0 and not bv.crashed:
            if prev_distance != float("inf") and distance < prev_distance:
                reward += approach_r

        return float(reward)

    # ── Gymnasium API ─────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        """
        Reset the inner environment and all episode-level metrics.

        The seed passed here is forwarded to the inner env so each episode
        gets a reproducible (or randomly varied) initial state.
        """
        super().reset(seed=seed)
        self._reset_metrics()

        effective_seed = seed if seed is not None else self._init_seed
        obs_tuple, info = self._inner_env.reset(seed=effective_seed)

        # obs_tuple = (ego_obs [5×5], bv_obs [5×5])
        ego_obs, bv_obs = obs_tuple[0], obs_tuple[1]
        self._last_ego_obs = ego_obs          # cache — used in next step()

        # Record starting lane of the selected BV
        self._prev_bv_lane = self._bv_vehicle().lane_index[2]

        return bv_obs.flatten().astype(np.float32), info

    def step(self, action_idx: int):
        """
        Apply action_idx for the BV, let the Ego respond, advance simulation.

        Returns
        ───────
        obs      : flattened BV kinematic observation  (25,) float32
        reward   : scalar shaped reward for this step
        terminated : True when ego crashes (episode objective reached/failed)
        truncated  : True when simulation time is up
        info       : dict with step-level and cumulative episode metrics
        """
        bv_lane_before = self._prev_bv_lane

        # ── Build joint action ────────────────────────────────────────────
        # Ego action comes from its fixed policy; BV action from PPO.
        ego_action     = self._get_ego_action(self._last_ego_obs)
        bv_action_str  = BV_ACTIONS[int(action_idx)]
        # VehicleAction is the namedtuple expected by HighwayEnvAdv._simulate()
        vehicle_action = VehicleAction(
            ego_action=ego_action,
            bv_action=bv_action_str,
        )

        # ── Step inner env ────────────────────────────────────────────────
        # When action is VehicleAction, abstract.py returns:
        #   obs = [unchanged_obs, updated_obs]
        # where each element is (ego_obs [5×5], bv_obs [5×5]) from
        # MultiAgentObservation.  We always use updated_obs (index 1) so
        # the BV state reflects the newly selected CBV after update_obs().
        obs_list, _inner_reward, terminated, truncated, info = (
            self._inner_env.step(vehicle_action)
        )
        updated_obs = obs_list[1]          # (ego_obs_new, bv_obs_new)
        ego_obs_new = updated_obs[0]       # numpy array (5, 5)
        bv_obs_new  = updated_obs[1]       # numpy array (5, 5)
        self._last_ego_obs = ego_obs_new   # update cache for next step

        # Current BV lane after the step
        bv_lane_after      = self._bv_vehicle().lane_index[2]
        self._prev_bv_lane = bv_lane_after

        # ── Compute metrics ───────────────────────────────────────────────
        ttc, distance = self._compute_ttc_and_distance()
        cut_in        = self._detect_cut_in(bv_lane_before, bv_lane_after)

        ego = self._ego_vehicle()
        bv  = self._bv_vehicle()

        if ego.crashed:
            self._ego_collision = True
        if bv.crashed and not ego.crashed:
            self._bv_collision = True
        if ttc < self._min_ttc:
            self._min_ttc = ttc
        if distance < self._min_distance:
            self._min_distance = distance
        if action_idx in (0, 2):
            self._lane_change_count += 1
        if cut_in:
            self._cut_in_count += 1
        self._ep_length += 1

        # ── Compute PPO reward ────────────────────────────────────────────
        # Pass prev_distance so _compute_reward can compute approach_reward.
        prev_dist = self._prev_distance
        reward = self._compute_reward(
            action_idx, ttc, distance, cut_in,
            bv_lane_before, bv_lane_after, prev_dist,
        )
        self._prev_distance = distance   # update for next step

        # ── Timeout penalty (hard-mode only) ─────────────────────────────
        # If the episode ends without ego crashing, penalise the BV.
        # This punishes the "park beside ego and survive" strategy.
        timeout_p = self._reward_cfg.get("timeout_penalty", 0.0)
        if timeout_p != 0.0 and (terminated or truncated) and not self._ego_collision:
            reward += timeout_p

        self._ep_reward += reward

        # ── Pack info dict ────────────────────────────────────────────────
        # These keys are read by EpisodeMetricsCallback.
        # Cumulative values reflect the whole episode so far.
        info.update({
            "episode_reward":    self._ep_reward,
            "ego_collision":     int(self._ego_collision),
            "bv_collision":      int(self._bv_collision),
            # Use -1.0 as sentinel when TTC was never finite (no close encounter)
            "min_ttc":           self._min_ttc if self._min_ttc != float("inf") else -1.0,
            "min_distance":      self._min_distance,
            "lane_change_count": self._lane_change_count,
            "cut_in_count":      self._cut_in_count,
            "episode_length":    self._ep_length,
        })

        return bv_obs_new.flatten().astype(np.float32), reward, terminated, truncated, info

    def render(self):
        return self._inner_env.render()

    def close(self):
        self._inner_env.close()


# ─────────────────────────────────────────────────────────────────────────────
# SB3 Callback  ——  episode-level metrics logging
# ─────────────────────────────────────────────────────────────────────────────

class EpisodeMetricsCallback(BaseCallback):
    """
    Logs the 8 custom metrics to TensorBoard at the END of each episode.

    SB3 uses DummyVecEnv internally, which auto-resets the env when done=True.
    The info dict at the done step contains the final cumulative values for that
    episode, so we check for dones and log at that moment.

    Logged under the TensorBoard tag  metrics/<key>.
    """

    _METRIC_KEYS = (
        "episode_reward",
        "ego_collision",
        "bv_collision",
        "min_ttc",
        "min_distance",
        "lane_change_count",
        "cut_in_count",
        "episode_length",
    )

    def __init__(self, verbose: int = 0) -> None:
        super().__init__(verbose)
        self._episode_count = 0

    def _on_step(self) -> bool:
        """Called after every env.step() by SB3."""
        dones = self.locals.get("dones", [])
        infos = self.locals.get("infos", [])

        for done, info in zip(dones, infos):
            if done:
                self._episode_count += 1
                for key in self._METRIC_KEYS:
                    if key in info:
                        self.logger.record(f"metrics/{key}", info[key])
                # Also record episode count for x-axis reference
                self.logger.record("metrics/episodes", self._episode_count)
                self.logger.dump(self.num_timesteps)

        return True  # continue training


# ─────────────────────────────────────────────────────────────────────────────
# Training entry point
# ─────────────────────────────────────────────────────────────────────────────

def auto_device() -> str:
    """Return 'mps' on Apple Silicon, 'cuda' if available, else 'cpu'."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def set_global_seeds(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train PPO-based adversarial BV agent (separate from Rainbow DQN)."
    )
    parser.add_argument(
        "--ego",
        type=str,
        default="IDM-Ego",
        choices=["IDM-Ego", "DQN-Ego", "A2C-Ego", "PPO-Ego"],
        help="Ego vehicle type to train against.",
    )
    parser.add_argument(
        "--risk_level",
        type=str,
        default="medium",
        choices=["easy", "medium", "hard"],
        help=(
            "Difficulty of the adversarial reward shaping.\n"
            "  easy   — cautious BV, heavy self-crash penalty, mild proximity incentive.\n"
            "  medium — balanced.\n"
            "  hard   — aggressive BV, strong TTC/proximity reward, light self-crash penalty."
        ),
    )
    parser.add_argument(
        "--lane_count",
        type=int,
        default=2,
        help="Number of highway lanes (2 or 3).",
    )
    parser.add_argument(
        "--total_timesteps",
        type=int,
        default=50_000,
        help="Total environment steps for PPO training.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Global random seed.",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="PPO_BV_models",
        help="Root directory for saving trained models.",
    )
    parser.add_argument(
        "--ent_coef",
        type=float,
        default=0.01,
        help=(
            "PPO entropy coefficient (exploration). "
            "Increase to 0.02-0.05 for hard mode to avoid premature convergence."
        ),
    )
    parser.add_argument(
        "--n_envs",
        type=int,
        default=4,
        help=(
            "Number of parallel environments for data collection. "
            "M1 Mac: 4–8 recommended. Speeds up training ~3-4x."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="Compute device: 'mps' (M1 GPU), 'cuda', or 'cpu'. Leave empty for auto-detect.",
    )
    args = parser.parse_args()

    # ── Reproducibility ───────────────────────────────────────────────────
    set_global_seeds(args.seed)

    # ── Build environment ─────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  PPO Adversarial BV Training")
    print(f"  Ego       : {args.ego}")
    print(f"  Risk level: {args.risk_level}")
    print(f"  Lanes     : {args.lane_count}")
    device = args.device if args.device else auto_device()
    print(f"  Steps     : {args.total_timesteps:,}")
    print(f"  Seed      : {args.seed}")
    print(f"  ent_coef  : {args.ent_coef}")
    print(f"  n_envs    : {args.n_envs}")
    print(f"  device    : {device}")
    print(f"{'='*60}\n")

    # ── A2C-Ego quality warning ────────────────────────────────────────────
    # The pre-trained A2C ego model in Ego_Agent/Ego_Agent_model/A2C/model
    # has a degenerate policy that almost always outputs SLOWER (brake)
    # regardless of the situation.  Training BV against it will produce a BV
    # that exploits this always-braking behaviour, not a genuinely hard
    # adversarial agent.  Consider using DQN-Ego or PPO-Ego instead.
    if args.ego == "A2C-Ego":
        print("[WARNING] A2C-Ego model has a degenerate policy (always SLOWER).")
        print("          The ego will not react to BV threats; trained BV will")
        print("          exploit the always-braking behaviour only.")
        print("          Recommended: use --ego DQN-Ego or --ego PPO-Ego.\n")

    # ── Build (possibly parallel) environments ────────────────────────────
    # With n_envs > 1 we run multiple independent copies of AdvBVEnv inside
    # a single DummyVecEnv.  Each copy gets a slightly different seed so the
    # initial scenarios are varied, improving generalisation.
    def _make_single_env(seed_offset: int):
        def _init():
            e = AdvBVEnv(
                ego_name   = args.ego,
                risk_level = args.risk_level,
                lane_count = args.lane_count,
                seed       = args.seed + seed_offset,
            )
            return Monitor(e)
        return _init

    if args.n_envs > 1:
        env = DummyVecEnv([_make_single_env(i) for i in range(args.n_envs)])
    else:
        env = Monitor(AdvBVEnv(
            ego_name   = args.ego,
            risk_level = args.risk_level,
            lane_count = args.lane_count,
            seed       = args.seed,
        ))

    # ── Save path ─────────────────────────────────────────────────────────
    # Layout: PPO_BV_models/{ego}/{risk_level}/lane{N}_seed{S}.zip
    model_dir  = os.path.join(args.save_path, args.ego, args.risk_level)
    model_name = f"lane{args.lane_count}_seed{args.seed}"
    model_path = os.path.join(model_dir, model_name)      # .zip appended by SB3
    os.makedirs(model_dir, exist_ok=True)

    tb_log_dir = os.path.join(
        "PPO_BV_logs", args.ego, args.risk_level
    )

    # ── PPO hyperparameters ───────────────────────────────────────────────
    # n_steps: steps collected per env per update (rollout = n_steps * n_envs).
    # batch_size: scale with n_envs so the GPU is fully utilised.
    # On M1 with n_envs=4: rollout=2048, batch_size=256 → 8 minibatches.
    n_steps    = 512
    batch_size = min(64 * max(args.n_envs, 1), 512)

    model = PPO(
        policy          = "MlpPolicy",
        env             = env,
        learning_rate   = 3e-4,
        n_steps         = n_steps,
        batch_size      = batch_size,
        n_epochs        = 10,
        gamma           = 0.99,
        gae_lambda      = 0.95,
        clip_range      = 0.2,
        ent_coef        = args.ent_coef,
        vf_coef         = 0.5,
        max_grad_norm   = 0.5,
        verbose         = 1,
        seed            = args.seed,
        device          = device,
        tensorboard_log = tb_log_dir,
    )

    # ── Callbacks ─────────────────────────────────────────────────────────
    metrics_cb = EpisodeMetricsCallback(verbose=0)

    # Save a checkpoint roughly every 20 % of training.
    # CheckpointCallback counts steps per-env, so divide by n_envs.
    ckpt_freq = max(args.total_timesteps // (5 * args.n_envs), 1_000)
    checkpoint_cb = CheckpointCallback(
        save_freq   = ckpt_freq,
        save_path   = model_dir,
        name_prefix = model_name,
        verbose     = 1,
    )

    # ── Train ─────────────────────────────────────────────────────────────
    model.learn(
        total_timesteps = args.total_timesteps,
        callback        = [metrics_cb, checkpoint_cb],
        tb_log_name     = model_name,
        reset_num_timesteps = True,
    )

    # ── Save final model ──────────────────────────────────────────────────
    model.save(model_path)
    print(f"\nFinal model saved → {model_path}.zip")
    print(f"TensorBoard logs  → {tb_log_dir}/")
    print(f"Run:  tensorboard --logdir PPO_BV_logs/\n")

    env.close()


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
