#!/usr/bin/env python3
"""
强化学习微调脚本（占位示例）

当 CARLA 不可用时，可使用 OpenVLAControlEnv + Stable-Baselines3 快速验证策略。
默认 dry-run，不会实际训练，仅演示如何搭建流水线。
"""

from __future__ import annotations

import argparse
from pathlib import Path

from training.rl_env import OpenVLAControlEnv

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
except ImportError:  # pragma: no cover - 可选依赖
    PPO = None
    DummyVecEnv = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OpenVLA 策略 RL 微调")
    parser.add_argument("--vec-envs", type=int, default=2, help="并行环境数量")
    parser.add_argument("--train-steps", type=int, default=0, help="训练步数（默认 0，仅 dry-run）")
    parser.add_argument("--max-steps", type=int, default=200, help="单个 episode 最大步数")
    parser.add_argument("--output", type=Path, default=Path("./checkpoints/ppo_openvla"))
    return parser.parse_args()


def main():
    args = parse_args()
    if PPO is None or DummyVecEnv is None:
        print("⚠️ 未检测到 stable-baselines3，请执行: pip install stable-baselines3 gymnasium[box2d]")
        return

    def env_builder():
        return OpenVLAControlEnv(max_steps=args.max_steps)

    try:
        vec_env = DummyVecEnv([env_builder for _ in range(args.vec_envs)])
    except ImportError as exc:
        print(f"⚠️ {exc}")
        return
    model = PPO("MultiInputPolicy", vec_env, verbose=1)

    if args.train_steps > 0:
        print(f"开始训练，共 {args.train_steps} 步...")
        model.learn(total_timesteps=args.train_steps)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(args.output))
        print(f"✓ 模型已保存至 {args.output}")
    else:
        print("Dry-run 完成，可通过 --train-steps N 启动正式训练。")


if __name__ == "__main__":
    main()
