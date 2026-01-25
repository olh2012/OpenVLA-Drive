"""
Open-loop evaluation for VLA Driving Policy.

Compute ADE/FDE on dataset splits and optionally dump metrics to JSON.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import torch
import yaml

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.carla_dataset import get_carla_vla_dataloader
from models.policy import VLADrivingPolicy


def load_policy_from_config(config_path: str, checkpoint: str, device: torch.device) -> VLADrivingPolicy:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    model_cfg = cfg.get("model", {})
    backbone_cfg = model_cfg.get("backbone", {})
    lora_cfg = model_cfg.get("lora", {})
    action_cfg = model_cfg.get("action_head", {})
    multi_task_cfg = model_cfg.get("multi_task", {})

    model = VLADrivingPolicy(
        model_name=backbone_cfg.get("model_name", "microsoft/phi-2"),
        vision_model_name=backbone_cfg.get("vision_model_name", "openai/clip-vit-base-patch32"),
        num_timesteps=action_cfg.get("num_timesteps", 10),
        action_head_hidden_dim=action_cfg.get("hidden_dim", 512),
        action_head_layers=action_cfg.get("num_layers", 3),
        use_lora=lora_cfg.get("use_lora", True),
        lora_config=lora_cfg,
        freeze_vision_tower=backbone_cfg.get("freeze_vision_tower", True),
        freeze_llm=backbone_cfg.get("freeze_llm", True),
        multi_task_config=multi_task_cfg,
    ).to(device)

    ckpt_path = Path(checkpoint)
    if ckpt_path.is_file():
        state = torch.load(ckpt_path, map_location=device)
        state_dict = state.get("state_dict", state)
        result = model.load_state_dict(
            {k.replace("model.", ""): v for k, v in state_dict.items()},
            strict=False,
        )
        print(f"✓ 权重加载完成，missing={result.missing_keys}, unexpected={result.unexpected_keys}")
    else:
        print("⚠️ 未找到 checkpoint，使用随机初始化模型")

    model.eval()
    return model


def compute_ade_fde(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
    displacement = torch.norm(pred - target, dim=-1)  # [B, T]
    ade = displacement.mean(dim=1)
    fde = displacement[:, -1]
    return {"ade": ade, "fde": fde}


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Open-loop evaluation (ADE/FDE)")
    parser.add_argument("--checkpoint", type=str, required=True, help="策略模型 checkpoint 路径")
    parser.add_argument("--config", type=str, default="configs/policy_config.yaml", help="模型配置文件")
    parser.add_argument("--data-root", type=str, default="./datasets/carla", help="数据集根目录")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-batches", type=int, default=-1, help="限制评估批次数，-1 为全量")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=str, default=None, help="输出 JSON 路径")
    return parser


def main():
    args = build_argparser().parse_args()
    device = torch.device(args.device)

    model = load_policy_from_config(args.config, args.checkpoint, device)
    num_timesteps = model.num_timesteps

    dataloader = get_carla_vla_dataloader(
        data_root=args.data_root,
        split=args.split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        num_trajectory_points=num_timesteps,
    )

    total_ade = 0.0
    total_fde = 0.0
    total_samples = 0

    for batch_idx, batch in enumerate(dataloader):
        if args.max_batches > 0 and batch_idx >= args.max_batches:
            break

        images = batch["image"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        targets = batch["trajectory"].to(device)

        with torch.no_grad():
            outputs = model(
                image_tensors=images,
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_language_output=False,
            )
        preds = outputs["trajectory"]
        metrics = compute_ade_fde(preds, targets)

        total_ade += metrics["ade"].sum().item()
        total_fde += metrics["fde"].sum().item()
        total_samples += preds.shape[0]

        if batch_idx % 20 == 0:
            print(f"batch={batch_idx:04d} | samples={total_samples}")

    total_samples = max(total_samples, 1)
    mean_ade = total_ade / total_samples
    mean_fde = total_fde / total_samples

    summary = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "split": args.split,
        "checkpoint": args.checkpoint,
        "config": args.config,
        "data_root": args.data_root,
        "num_samples": total_samples,
        "metrics": {"ade": mean_ade, "fde": mean_fde},
    }

    print("\n=== Open-loop 评估汇总 ===")
    print(f"ADE: {mean_ade:.4f}")
    print(f"FDE: {mean_fde:.4f}")

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=True)
        print(f"✓ 评估结果已写入 {output_path}")


if __name__ == "__main__":
    main()
