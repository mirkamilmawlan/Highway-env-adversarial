"""
train_ego_adv.py
────────────────
用预训练的 Rainbow DQN 对抗 BV 来训练 PPO / A2C Ego。

Ego 在 highway-adv-v0 中学习躲避 BV 攻击。

A2C 专项优化说明（PPO 路径完全不变）
─────────────────────────────────────
1. SubprocVecEnv（默认 n_envs=8）
   参考：SB3 RL Tips「Vectorized Environments」
   原因：DummyVecEnv 串行 → 样本高度相关；SubprocVecEnv 多进程 → 样本去相关，
         on-policy A2C 梯度估计更稳定。

2. VecNormalize(norm_obs=True, norm_reward=True, clip_obs=10)
   参考：SB3 RL Tips「Normalization」
   原因：Kinematics obs 包含位置（m）与速度（m/s），尺度不同；
         未归一化时 critic 值函数噪声高，advantage 估计不稳。
         统计保存为 vecnorm.pkl，测试时加载冻结，保证 obs 一致性。

3. RL Zoo 超参（RMSpropTFLike / net_arch=[256,256] / gamma=0.995 / ent_coef=0.02）
   参考：rl-baselines3-zoo hyperparams/a2c.yml + SB3 A2C 文档
   原因：默认 n_steps=5 / ent_coef=0 导致策略退化为确定性 SLOWER；
         新配置强制探索（ent_coef=0.02）并用更稳定的优化器。

4. 改进奖励（improved_reward=True，仅 A2C 使用）
   参考：SB3 RL Tips「Reward Shaping」+ 奖励漏洞分析（见下文）
   原因：旧奖励存在三个漏洞导致 100% SLOWER（详见 _ego_reward_improved）。

5. A2C 固定 device="cpu"
   原因：MPS 后端在 macOS 上与 SB3 A2C 存在死锁风险。

========== Reward Exploit 分析（旧 _ego_reward 中的三个漏洞）==========

旧奖励每步：
  speed_r = clip((speed-20)/15, 0, 1)
  reward  = 0.6*speed_r + 0.1 + [dist bonus/penalty] + [crash penalty]

SLOWER 策略下（speed≈15-18 m/s）：
  漏洞①  +0.1 survival（无条件）→ 40步 × 0.1 = 4.0 免费奖励
  漏洞②  speed_r≈0 → 速度奖励=0，但没有低速惩罚，停着不吃亏
  漏洞③  BV 速度比慢速 ego 快，会超越 ego → dist > 20m → +0.05/步
          最多 30步 × 0.05 = 1.5 额外奖励
  合计≈ 5.5/episode，无碰撞风险

正常驾驶（speed=30m/s）：
  speed_r = (30-20)/15 = 0.67 → 40步 × 0.6 × 0.67 ≈ 16.1
  但 A2C 早期探索随机碰撞频繁 → 每次 -8.0 → 期望收益不稳定
  → A2C 快速收敛到"安全"的 SLOWER 策略，进入局部最优

三个漏洞组合导致 SLOWER 成为稳定局部最优（每集 ~5.5 无风险 reward）。
=======================================================================

用法示例
--------
PPO（原有流程不变）:
  python -u train_ego_adv.py --algo PPO --bv_ego DQN-Ego --lanes 2 \\
      --total_timesteps 800000 --n_envs 4 --device mps --train

A2C（RL Zoo 最佳实践）:
  python -u train_ego_adv.py --algo A2C --bv_ego DQN-Ego --lanes 2 \\
      --total_timesteps 3000000 --n_envs 8 --train

测试 + 录像（A2C）:
  python -u train_ego_adv.py --algo A2C --bv_ego DQN-Ego --lanes 2 --test --render
"""

import argparse
import os
import warnings
from typing import Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, DummyVecEnv
from stable_baselines3.common.sb2_compat.rmsprop_tf_like import RMSpropTFLike

import highway_env  # noqa: F401 – registers highway envs
from BV_Agent.Rainbow import RainbowDQN
from config import Bv_Action
from highway_env.envs.common.action import VehicleAction

warnings.filterwarnings("ignore", category=UserWarning)

# ── 常量 ──────────────────────────────────────────────────────────────────────
EGO_OBS_DIM = 25          # Kinematics 5 vehicles × 5 features
BV_OBS_DIM  = 25          # CbvKinematicObservation 5×5 展平
N_ACTIONS   = len(Bv_Action)
SLOWER_IDX  = 4           # Discrete(5) 中 SLOWER 对应的动作编号


def auto_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# ── Gymnasium 包装器 ──────────────────────────────────────────────────────────
class AdvEgoEnv(gym.Env):
    """
    将 highway-adv-v0 包装成单智能体 Gymnasium 环境。

    参数
    ----
    improved_reward : bool
        False → 原有奖励函数（PPO 使用，向后兼容）
        True  → 重设计奖励（A2C 使用），消除三个 SLOWER 漏洞
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        bv_ego: str,
        lanes: int = 2,
        bv_seed: int = 1,
        render_mode: Optional[str] = None,
        improved_reward: bool = False,
    ):
        super().__init__()
        self.render_mode     = render_mode
        self._lanes          = lanes
        self._bv_ego         = bv_ego
        self._improved_reward = improved_reward

        # ── 内层对抗环境 ─────────────────────────────────────────────────────
        self._env = gym.make("highway-adv-v0", render_mode=render_mode)
        self._env.configure({
            "observation": {
                "type": "MultiAgentObservation",
                "observation_config": [
                    {
                        "type": "Kinematics",
                        "vehicles_count": 5,      # 固定5辆 → 5×5=25 特征
                        "features": ["presence", "x", "y", "vx", "vy"],
                        "absolute": False,
                        "normalize": True,
                    },
                    {"type": "CbvKinematicObservation"},
                ],
            },
            "ego_type": None,
            "lanes_count": lanes,
            "vehicles_count": 12,
            "controlled_vehicles": 2,
            "duration": 40,
            "BV_type": "highway_env.vehicle.behavior.AdvVehicle",
            "initial_lane_id": lanes - 1,
            "initial_bv_lane_id": lanes - 2,
            "simulation_frequency": 15,
        })

        # ── Rainbow DQN BV（只推理，不训练）────────────────────────────────
        self._bv_agent = RainbowDQN(
            env=self._env,
            memory_size=100,
            batch_size=32,
            target_update=100,
            obs_dim=BV_OBS_DIM,
            action_dim=N_ACTIONS,
        )
        self._bv_agent.is_test = True
        self._bv_agent.load(model_name=bv_ego, lane_count=lanes, seed=bv_seed)
        self._bv_obs_dim = BV_OBS_DIM

        self.action_space = gym.spaces.Discrete(N_ACTIONS)
        self._last_multi_obs: Optional[Tuple] = None

        # 动态检测真实 ego obs 维度
        _init_obs, _ = self._env.reset()
        _ego_obs, _  = self._split_obs(_init_obs)
        self._ego_obs_dim   = len(_ego_obs)
        self._last_multi_obs = _init_obs

        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(self._ego_obs_dim,), dtype=np.float32
        )

    # ── 辅助：从 MultiAgent obs 提取 ego / bv 观测 ─────────────────────────
    @staticmethod
    def _split_obs(multi_obs) -> Tuple[np.ndarray, np.ndarray]:
        ego_obs = np.asarray(multi_obs[0]).flatten().astype(np.float32)
        bv_obs  = np.asarray(multi_obs[1]).flatten().astype(np.float32)
        return ego_obs, bv_obs

    # ── reset ────────────────────────────────────────────────────────────────
    def reset(self, *, seed=None, options=None):
        multi_obs, info = self._env.reset(seed=seed)
        self._last_multi_obs = multi_obs
        ego_obs, _ = self._split_obs(multi_obs)
        return ego_obs, info

    # ── step ─────────────────────────────────────────────────────────────────
    def step(self, ego_action: int):
        _, bv_obs = self._split_obs(self._last_multi_obs)

        bv_idx    = int(self._bv_agent.select_action(bv_obs.reshape(1, self._bv_obs_dim)))
        bv_action = Bv_Action[bv_idx]

        combined = VehicleAction(ego_action=int(ego_action), bv_action=bv_action)
        raw_obs, _, terminated, truncated, info = self._env.step(combined)

        # highway-adv-v0 step() 返回 (former_obs, updated_obs)
        updated_obs          = raw_obs[1]
        self._last_multi_obs = updated_obs
        ego_obs, _           = self._split_obs(updated_obs)

        if self._improved_reward:
            reward = self._ego_reward_improved(int(ego_action), info, terminated or truncated)
        else:
            reward = self._ego_reward(info, terminated or truncated)

        return ego_obs, reward, terminated, truncated, info

    # ── 原有奖励函数（PPO 使用，向后兼容，完全不变）──────────────────────────
    def _ego_reward(self, info: dict, done: bool) -> float:
        env         = self._env.unwrapped
        ego_vehicle = env.controlled_vehicles[0]
        bv_vehicle  = env.controlled_vehicles[1] if len(env.controlled_vehicles) > 1 else None
        crashed     = getattr(ego_vehicle, "crashed", False)

        speed   = getattr(ego_vehicle, "speed", 0.0)
        speed_r = float(np.clip((speed - 20.0) / 15.0, 0.0, 1.0))
        reward  = 0.6 * speed_r
        reward += 0.1

        if bv_vehicle is not None:
            dx   = ego_vehicle.position[0] - bv_vehicle.position[0]
            dy   = ego_vehicle.position[1] - bv_vehicle.position[1]
            dist = float(np.sqrt(dx * dx + dy * dy))
            if dist < 10.0:
                reward -= 0.3 * (1.0 - dist / 10.0)
            elif dist > 20.0:
                reward += 0.05

        if crashed:
            reward -= 8.0

        return float(reward)

    # ── 改进奖励函数（A2C 使用，消除 SLOWER 三个漏洞）───────────────────────
    def _ego_reward_improved(self, action_idx: int, info: dict, done: bool) -> float:
        """
        修复旧奖励中三个导致 100% SLOWER 的漏洞：

        漏洞①修复：无条件 survival → Speed-gated survival
          旧：每步无条件 +0.1（40步 = 4.0 免费奖励）
          新：speed ≥ 22 m/s 才给 +0.10；speed < 18 m/s 持续扣分

        漏洞②修复：删除 dist > 20m → +0.05 奖励
          旧：减速让 BV 超过自己 → dist > 20m → 拿 bonus
          新：用 TTC-based 威胁信号替代（只有 BV 正在接近时才惩罚）

        漏洞③修复：低速无惩罚
          旧：speed ≈ 15 m/s 时速度奖励 = 0 但无额外扣分
          新：speed < 18 m/s → 每步 -0.05×(18-speed)，speed=15 → -0.15/步

        额外：SLOWER 动作本身 -0.02，使其不再是零成本默认选择
        参考：SB3 RL Tips「Reward Shaping」；Sutton & Barto step cost
        """
        env         = self._env.unwrapped
        ego_vehicle = env.controlled_vehicles[0]
        bv_vehicle  = env.controlled_vehicles[1] if len(env.controlled_vehicles) > 1 else None
        crashed     = getattr(ego_vehicle, "crashed", False)
        speed       = getattr(ego_vehicle, "speed", 0.0)

        # ① 速度奖励（保留核心，系数略调小为 0.5 给其他项留空间）
        speed_r = float(np.clip((speed - 20.0) / 15.0, 0.0, 1.0))
        reward  = 0.5 * speed_r

        # ② Speed-gated survival（漏洞①修复）
        #    只有以够快的速度规避才算"有效生存"
        if speed >= 22.0:
            reward += 0.10          # 高速生存奖励
        elif speed < 18.0:
            # 低速惩罚：speed=15 → -0.15/步 → 40步 = -6.0
            # 彻底消除"停着也能活"的动机
            reward -= 0.05 * (18.0 - speed)

        # ③ TTC-based 威胁信号（漏洞②、③修复）
        #    不再奖励"BV 已经超过自己"，改为"BV 正在接近时惩罚"
        if bv_vehicle is not None:
            dx   = ego_vehicle.position[0] - bv_vehicle.position[0]
            dy   = ego_vehicle.position[1] - bv_vehicle.position[1]
            dist = float(np.sqrt(dx * dx + dy * dy))

            if dist < 20.0:
                try:
                    # 负值 = BV 正在追近；正值 = 正在远离
                    rel_vx = float(ego_vehicle.velocity[0] - bv_vehicle.velocity[0])
                except (AttributeError, IndexError):
                    rel_vx = 0.0
                closing_speed = max(-rel_vx, 0.0)   # BV 接近速度（m/s）
                if closing_speed > 0.3:              # 过滤静态噪声
                    ttc = dist / closing_speed        # 粗略碰撞时间（秒）
                    # ttc ∈ [0, 5]：越小越危险；clip 到 [0,1] 再惩罚
                    danger = float(np.clip(1.0 - ttc / 5.0, 0.0, 1.0))
                    reward -= 0.25 * danger

        # ④ SLOWER 动作成本（参考 Sutton & Barto step cost 思想）
        #    很小但明确：避免 SLOWER 成为"免费默认动作"
        if action_idx == SLOWER_IDX:
            reward -= 0.02

        # ⑤ 碰撞惩罚（量级保持，确保 crash 仍是最坏结果）
        if crashed:
            reward -= 8.0

        return float(reward)

    # ── render / close ───────────────────────────────────────────────────────
    def render(self):
        return self._env.render()

    def close(self):
        self._env.close()

    def set_record_video_wrapper(self, wrapper):
        self._env.unwrapped.set_record_video_wrapper(wrapper)


# ── A2C 模型构建（集中管理，与 PPO 路径完全隔离）────────────────────────────
def _build_a2c_model(env, tb_log: str) -> A2C:
    """
    RL Zoo 风格 A2C 超参。
    参考：rl-baselines3-zoo / hyperparams/a2c.yml + SB3 A2C 文档。
    """
    policy_kwargs = dict(
        optimizer_class=RMSpropTFLike,       # TF A2C 官方 optimizer，eps=1e-5 更稳定
        optimizer_kwargs=dict(eps=1e-5),
        net_arch=dict(pi=[256, 256], vf=[256, 256]),  # 更大网络，提取关系特征
        ortho_init=False,                    # RL Zoo 对 MLP 的推荐，避免 tanh 饱和
    )
    return A2C(
        "MlpPolicy", env,
        device="cpu",         # 强制 cpu，避免 MPS 死锁
        learning_rate=7e-4,
        n_steps=64,           # 8 env × 64 = 512 samples/update（原来 n_steps=5 → 样本量严重不足）
        gamma=0.995,          # 长视野（50步 episode，需要更远的信用分配）
        gae_lambda=0.95,      # 标准 λ，降低 advantage 估计方差
        ent_coef=0.02,        # 熵正则（原来 0.0 → 策略很快退化为确定性 SLOWER）
        vf_coef=0.4,          # RL Zoo 标准值
        max_grad_norm=0.5,    # 梯度裁剪，防止大步更新破坏策略
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=tb_log,
    )


# ── 训练 / 测试 主函数 ────────────────────────────────────────────────────────
def main(args):
    algo    = args.algo.upper()
    AlgoCls = {"PPO": PPO, "A2C": A2C}[algo]

    # A2C 强制 cpu（MPS 死锁）；PPO 保持原来自动检测
    device = "cpu" if algo == "A2C" else (args.device if args.device else auto_device())

    save_dir     = f"Ego_Agent/Ego_Agent_model/{algo}_adv/{args.bv_ego}_lane{args.lanes}"
    save_path    = f"{save_dir}/model"
    vecnorm_path = f"{save_dir}/vecnorm.pkl"
    tb_log       = save_dir
    os.makedirs(save_dir, exist_ok=True)

    print("=" * 60)
    print(f"  Adv-Ego Training  [{algo}]")
    print(f"  BV source : {args.bv_ego}  |  Lanes: {args.lanes}")
    print(f"  Device    : {device}  |  n_envs: {args.n_envs}")
    print(f"  Steps     : {args.total_timesteps:,}")
    if algo == "A2C":
        print("  Reward    : improved (SLOWER exploit removed)")
        print("  VecEnv    : SubprocVecEnv + VecNormalize")
    print("=" * 60)

    # ── 训练 ─────────────────────────────────────────────────────────────────
    if args.train:
        if algo == "A2C":
            # ── A2C 专用路径 ────────────────────────────────────────────────
            def _make_a2c():
                return AdvEgoEnv(
                    bv_ego=args.bv_ego,
                    lanes=args.lanes,
                    bv_seed=args.bv_seed,
                    improved_reward=True,     # 启用改进奖励，消除 SLOWER 漏洞
                )

            env = make_vec_env(
                _make_a2c,
                n_envs=args.n_envs,
                seed=0,
                vec_env_cls=SubprocVecEnv,   # 多进程并行采样
            )
            env = VecNormalize(
                env,
                norm_obs=True,
                norm_reward=True,
                clip_obs=10.0,
            )
            model = _build_a2c_model(env, tb_log)

        else:
            # ── PPO 原有路径，完全不变 ──────────────────────────────────────
            def _make_ppo():
                return AdvEgoEnv(
                    bv_ego=args.bv_ego,
                    lanes=args.lanes,
                    bv_seed=args.bv_seed,
                    improved_reward=False,    # PPO 保持原奖励
                )

            env = make_vec_env(_make_ppo, n_envs=args.n_envs, seed=0)
            model = PPO(
                "MlpPolicy", env,
                device=device,
                tensorboard_log=tb_log,
                ent_coef=0.01,
                n_steps=2048,
                batch_size=64,
                verbose=1,
            )

        model.learn(total_timesteps=args.total_timesteps, log_interval=1)
        model.save(save_path)
        print(f"\n>> Model saved → {save_path}.zip")

        if algo == "A2C":
            env.save(vecnorm_path)
            print(f">> VecNormalize stats → {vecnorm_path}")

        env.close()

    # ── 测试 + 录像 ───────────────────────────────────────────────────────────
    if args.test:
        use_improved = (algo == "A2C")
        single_env = AdvEgoEnv(
            bv_ego=args.bv_ego,
            lanes=args.lanes,
            bv_seed=args.bv_seed,
            render_mode="rgb_array",
            improved_reward=use_improved,
        )

        if args.render:
            video_dir = os.path.join(save_dir, "video")
            os.makedirs(video_dir, exist_ok=True)
            single_env = RecordVideo(
                single_env, video_folder=video_dir,
                episode_trigger=lambda e: True,
            )
            single_env.unwrapped.set_record_video_wrapper(single_env)
            print(f">> Videos → {video_dir}/")

        if algo == "A2C":
            # 包装成 VecEnv 以兼容 VecNormalize
            vec_env = DummyVecEnv([lambda: single_env])
            if os.path.exists(vecnorm_path):
                vec_env = VecNormalize.load(vecnorm_path, vec_env)
                vec_env.training    = False   # 冻结 running stats
                vec_env.norm_reward = False   # 测试时不归一化 reward
                print(f">> VecNormalize loaded from {vecnorm_path}")
            else:
                print("[WARNING] vecnorm.pkl not found — obs will NOT be normalised")

            model = A2C.load(save_path, env=vec_env, device="cpu")

            crash_count = 0
            for ep in range(10):
                obs      = vec_env.reset()
                done     = [False]
                ep_r     = 0.0
                last_info = {}
                while not done[0]:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, r, done, info = vec_env.step(action)
                    ep_r += float(r[0])
                    last_info = info[0] if info else {}
                crashed = last_info.get("crashed", False)
                if crashed:
                    crash_count += 1
                print(f"  Episode {ep}: reward={ep_r:.2f}  crashed={'YES' if crashed else 'no'}")

            print(f"\n>> Ego crash rate: {crash_count}/10")
            vec_env.close()

        else:
            # PPO 测试路径，完全不变
            model = AlgoCls.load(save_path, env=single_env, device=device)
            crash_count = 0
            for ep in range(10):
                obs, _ = single_env.reset()
                done = truncated = False
                ep_r = 0.0
                while not (done or truncated):
                    action, _ = model.predict(obs, deterministic=True)
                    obs, r, done, truncated, info = single_env.step(int(action))
                    ep_r += r
                crashed = info.get("crashed", False)
                if crashed:
                    crash_count += 1
                print(f"  Episode {ep}: reward={ep_r:.2f}  crashed={'YES' if crashed else 'no'}")

            print(f"\n>> Ego crash rate: {crash_count}/10")
            single_env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train PPO/A2C ego against adversarial Rainbow DQN BV"
    )
    parser.add_argument(
        "--bv_ego", type=str, default="DQN-Ego",
        choices=["IDM-Ego", "DQN-Ego", "A2C-Ego", "PPO-Ego"],
    )
    parser.add_argument("--lanes",           type=int, default=2)
    parser.add_argument("--bv_seed",         type=int, default=1)
    parser.add_argument("--total_timesteps", type=int, default=3_000_000,
                        help="A2C 建议 ≥ 2M，推荐 3M-5M")
    parser.add_argument("--n_envs",          type=int, default=8,
                        help="A2C 建议 8（SubprocVecEnv）；PPO 用 4 即可")
    parser.add_argument("--device",          type=str, default="",
                        help="PPO 的 device。A2C 无论此参数如何都使用 cpu")
    parser.add_argument("--algo",  type=str, default="PPO",
                        choices=["PPO", "A2C"])
    parser.add_argument("--train",  action="store_true")
    parser.add_argument("--test",   action="store_true")
    parser.add_argument("--render", action="store_true", help="录制测试视频")
    args = parser.parse_args()
    main(args)
