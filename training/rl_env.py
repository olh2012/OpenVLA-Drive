"""
简化版 RL 环境，用于在 CARLA 连接不可用时快速调试强化学习策略。

特点：
- gymnasium 接口兼容 Stable-Baselines3
- 观测包含图像、离散指令 ID 以及归一化速度
- 动作为 [转向, 油门, 刹车]，范围分别为 [-1,1]、[0,1]、[0,1]
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Tuple, Optional

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover - 仅在需要 RL 时安装
    gym = None
    spaces = None

DEFAULT_COMMANDS = [
    "follow_lane",
    "turn_left",
    "turn_right",
    "stop",
    "lane_change_left",
    "lane_change_right",
]

BaseEnv = gym.Env if gym is not None else object


class OpenVLAControlEnv(BaseEnv):
    """
    轻量级驾驶环境，奖励函数基于期望横向偏移和速度，便于 RL 微调初步实验。
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        image_size: Tuple[int, int] = (224, 224),
        max_steps: int = 200,
        seed: int = 42,
    ):
        if gym is None or spaces is None:
            raise ImportError("OpenVLAControlEnv 依赖 gymnasium，请先安装: pip install gymnasium[box2d]")

        super().__init__()
        self.height, self.width = image_size
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)
        self.step_count = 0
        self.goal_lateral_offset = 0.0
        self.command_id = 0
        self._last_rgb: Optional[np.ndarray] = None

        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = spaces.Dict(
            {
                "image": spaces.Box(
                    low=0,
                    high=255,
                    shape=(3, self.height, self.width),
                    dtype=np.uint8,
                ),
                "command_id": spaces.Discrete(len(DEFAULT_COMMANDS)),
                "speed": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            }
        )

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.step_count = 0
        self.goal_lateral_offset = float(self.rng.uniform(-0.4, 0.4))
        self.command_id = int(self.rng.integers(0, len(DEFAULT_COMMANDS)))
        obs = self._sample_observation()
        self._last_rgb = obs["image"]
        return obs, {"goal_offset": self.goal_lateral_offset}

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high).astype(np.float32)
        self.step_count += 1

        lateral_error = abs(self.goal_lateral_offset - float(action[0]))
        throttle = float(action[1])
        brake = float(action[2])

        reward = 1.0 - 0.6 * lateral_error + 0.4 * throttle - 0.5 * brake
        reward -= 0.1 * max(0.0, 0.2 - throttle)
        reward = float(np.clip(reward, -1.0, 1.0))

        terminated = lateral_error > 0.75
        truncated = self.step_count >= self.max_steps

        obs = self._sample_observation()
        self._last_rgb = obs["image"]
        info = {
            "lateral_error": lateral_error,
            "goal_offset": self.goal_lateral_offset,
        }
        return obs, reward, terminated, truncated, info

    def render(self):
        if self._last_rgb is None:
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)
        return np.transpose(self._last_rgb, (1, 2, 0))

    def close(self):
        return

    # --------------------------------------------------------------------- utils
    def _sample_observation(self) -> Dict[str, np.ndarray]:
        rgb = self.rng.integers(
            low=0,
            high=255,
            size=(3, self.height, self.width),
            dtype=np.uint8,
        )
        speed = np.array([self.rng.uniform(0.0, 1.0)], dtype=np.float32)
        obs = {
            "image": rgb,
            "command_id": np.int64(self.command_id),
            "speed": speed,
        }
        return obs
