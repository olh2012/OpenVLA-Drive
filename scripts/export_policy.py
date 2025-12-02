#!/usr/bin/env python3
"""
策略模型导出脚本

功能:
- 读取 Lightning/PyTorch checkpoint
- 规整 state_dict 并输出至发行目录
- 打包元数据与配置文件，方便上传至 HuggingFace / ModelScope
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
from pathlib import Path
from typing import Dict

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="导出 VLADrivingPolicy checkpoint")
    parser.add_argument("--checkpoint", type=Path, required=False, help="Lightning 或普通 PyTorch checkpoint")
    parser.add_argument("--config", type=Path, default=Path("configs/policy_config.yaml"), help="模型配置路径")
    parser.add_argument("--output-dir", type=Path, default=Path("./release/openvla_policy"), help="导出目录")
    parser.add_argument("--mock", action="store_true", help="当未提供 checkpoint 时，生成随机权重示例")
    return parser.parse_args()


def load_state_dict(ckpt_path: Path, mock: bool) -> Dict[str, torch.Tensor]:
    if ckpt_path and ckpt_path.exists():
        print(f"读取 checkpoint: {ckpt_path}")
        payload = torch.load(ckpt_path, map_location="cpu")
        state_dict = payload.get("state_dict", payload)
        cleaned = {
            k.replace("model.", ""): v for k, v in state_dict.items()
        }
        print(f"✓ 权重条目数: {len(cleaned)}")
        return cleaned

    if not mock:
        raise FileNotFoundError("未找到 checkpoint，请提供 --checkpoint 或使用 --mock 生成示例。")

    print("⚠️ 未提供 checkpoint，生成随机示例权重 (mock)。")
    dummy = {
        "vision_projection.weight": torch.randn(2048, 1024),
        "vision_projection.bias": torch.zeros(2048),
        "action_head.mlp.0.weight": torch.randn(512, 2048),
        "action_head.mlp.0.bias": torch.zeros(512),
    }
    return dummy


def write_metadata(output_dir: Path, state_dict: Dict[str, torch.Tensor], source: Path | None):
    num_params = sum(t.numel() for t in state_dict.values())
    meta = {
        "export_time": dt.datetime.utcnow().isoformat() + "Z",
        "source_checkpoint": str(source) if source else "mock",
        "num_tensors": len(state_dict),
        "num_parameters": num_params,
    }
    meta_path = output_dir / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"✓ 写入元数据: {meta_path}")


def copy_config(config_path: Path, output_dir: Path):
    if config_path.exists():
        target = output_dir / config_path.name
        shutil.copy(config_path, target)
        print(f"✓ 复制配置文件: {target}")
    else:
        print(f"⚠️ 未找到配置文件 {config_path}，跳过复制。")


def main():
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    state_dict = load_state_dict(args.checkpoint, args.mock)
    state_path = output_dir / "policy_state.pt"
    torch.save(state_dict, state_path)
    print(f"✓ State dict 已保存至 {state_path}")

    write_metadata(output_dir, state_dict, args.checkpoint)
    copy_config(args.config, output_dir)

    print("\n导出完成，可将整个目录上传至模型仓库。")


if __name__ == "__main__":
    main()
