"""
Training Script for OpenVLA-Drive.

目前默认针对 VLADrivingPolicy（轨迹预测）进行训练。
"""

from __future__ import annotations

import argparse
import os
import sys
from copy import deepcopy

import yaml

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Conditional imports - only import if packages are available
try:
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
    from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
    LIGHTNING_AVAILABLE = True
except ImportError:
    print("Warning: PyTorch Lightning not available")
    LIGHTNING_AVAILABLE = False

from data.carla_dataset import get_carla_vla_dataloader
from training.policy_lightning_module import VLAPolicyLightningModule
from training.lightning_module import VLALightningModule  # 保留，方便后续扩展


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def setup_callbacks(config: dict):
    """Setup training callbacks."""
    callbacks = []

    checkpoint_config = config.get('checkpoint', {})
    checkpoint_callback = ModelCheckpoint(
        monitor=checkpoint_config.get('monitor', 'val_loss'),
        mode=checkpoint_config.get('mode', 'min'),
        save_top_k=checkpoint_config.get('save_top_k', 3),
        save_last=checkpoint_config.get('save_last', True),
        dirpath=checkpoint_config.get('dirpath', './checkpoints'),
        filename=checkpoint_config.get('filename', 'vla-{epoch:02d}-{val_loss:.4f}'),
    )
    callbacks.append(checkpoint_callback)

    early_stopping_config = config.get('early_stopping', {})
    if early_stopping_config:
        early_stopping = EarlyStopping(
            monitor=early_stopping_config.get('monitor', 'val_loss'),
            patience=early_stopping_config.get('patience', 10),
            mode=early_stopping_config.get('mode', 'min'),
            min_delta=early_stopping_config.get('min_delta', 0.001),
        )
        callbacks.append(early_stopping)

    return callbacks


def setup_logger(config: dict):
    """Setup experiment logger."""
    logging_config = config.get('logging', {})

    if logging_config.get('use_wandb', False):
        logger = WandbLogger(
            project=logging_config.get('project_name', 'openvla-drive'),
            name=logging_config.get('run_name'),
            save_dir=logging_config.get('save_dir', './outputs'),
        )
    elif logging_config.get('use_tensorboard', True):
        logger = TensorBoardLogger(
            save_dir=logging_config.get('log_dir', './logs'),
            name=logging_config.get('project_name', 'openvla-drive'),
        )
    else:
        logger = None

    return logger


def build_policy_dataloader(
    data_config: dict,
    split: str,
    tokenizer_name: str,
    num_timesteps: int,
):
    """Create CARLA dataloader compatible with VLADrivingPolicy."""
    dataset_cfg = data_config.get('dataset', {})
    split_cfg = data_config.get(split, {})

    if not dataset_cfg:
        raise ValueError("data_config.dataset 不可为空")

    return get_carla_vla_dataloader(
        data_root=dataset_cfg.get('data_root', './datasets/carla'),
        split=split,
        batch_size=split_cfg.get('batch_size', 8),
        num_workers=split_cfg.get('num_workers', 4),
        shuffle=split_cfg.get('shuffle', split == 'train'),
        image_size=tuple(dataset_cfg.get('image_size', [224, 224])),
        tokenizer_name=tokenizer_name,
        num_trajectory_points=num_timesteps,
        max_text_length=data_config.get('language', {}).get('max_length', 128),
    )


def apply_policy_training_overrides(
    training_config: dict,
    data_config: dict,
    policy_training_cfg: dict,
):
    """Align batch size / LR / epochs with policy config when可用."""
    if not policy_training_cfg:
        return

    batch_size = policy_training_cfg.get('batch_size')
    if batch_size:
        data_config.setdefault('train', {})['batch_size'] = batch_size
        data_config.setdefault('val', {})['batch_size'] = batch_size

    lr = policy_training_cfg.get('learning_rate')
    if lr:
        training_config.setdefault('optimizer', {})['lr'] = lr

    max_epochs = policy_training_cfg.get('max_epochs')
    if max_epochs:
        training_config.setdefault('trainer', {})['max_epochs'] = max_epochs


def main():
    parser = argparse.ArgumentParser(description="Train OpenVLA-Drive models")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training_config.yaml",
        help="Training hyper-parameter config",
    )
    parser.add_argument(
        "--model_config",
        type=str,
        default="configs/model_config.yaml",
        help="(VLA 模型) 配置文件；policy 模式会使用 --policy_config",
    )
    parser.add_argument(
        "--data_config",
        type=str,
        default="configs/data_config.yaml",
        help="Data/DataLoader config",
    )
    parser.add_argument(
        "--policy_config",
        type=str,
        default="configs/policy_config.yaml",
        help="VLADrivingPolicy 配置文件",
    )
    parser.add_argument(
        "--module",
        choices=["policy", "vla"],
        default="policy",
        help="选择训练模块（默认：policy 轨迹预测）",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Checkpoint path for resuming",
    )

    args = parser.parse_args()

    if not LIGHTNING_AVAILABLE:
        raise ImportError("未检测到 PyTorch Lightning，请先安装 pytorch-lightning>=2.1")

    print("=" * 60)
    print(f"OpenVLA-Drive Training ({args.module})")
    print("=" * 60)

    print("Loading configs...")
    training_config = load_config(args.config)
    data_config = load_config(args.data_config)

    if args.module == "policy":
        policy_cfg = load_config(args.policy_config)
        model_config = policy_cfg.get('model', {})
        apply_policy_training_overrides(
            training_config,
            data_config,
            policy_cfg.get('training', {}),
        )
    else:
        model_config = load_config(args.model_config)
        print("⚠️ 经典 VLA 模式目前未集成对应数据管线，如需使用请自行扩展。")

    print(f"Training config: {args.config}")
    print(f"Data config:     {args.data_config}")
    if args.module == "policy":
        print(f"Policy config:   {args.policy_config}")
    else:
        print(f"Model config:    {args.model_config}")

    seed = training_config.get('seed', 42)
    pl.seed_everything(seed)
    print(f"Random seed: {seed}")

    print("\nSetting up data loaders...")
    train_loader = val_loader = None

    try:
        if args.module == "policy":
            tokenizer_name = model_config.get('backbone', {}).get('model_name', 'microsoft/phi-2')
            num_timesteps = model_config.get('action_head', {}).get('num_timesteps', 10)

            train_loader = build_policy_dataloader(
                data_config=deepcopy(data_config),
                split='train',
                tokenizer_name=tokenizer_name,
                num_timesteps=num_timesteps,
            )
            val_loader = build_policy_dataloader(
                data_config=deepcopy(data_config),
                split='val',
                tokenizer_name=tokenizer_name,
                num_timesteps=num_timesteps,
            )
        else:
            raise NotImplementedError("当前脚本尚未实现传统 VLA 模式的数据管线。")

        print(f"Train samples: {len(train_loader.dataset)}")
        print(f"Val samples:   {len(val_loader.dataset)}")
    except Exception as exc:
        print(f"Warning: Could not load data - {exc}")

    print("\nInitializing model...")
    if args.module == "policy":
        model = VLAPolicyLightningModule(
            model_config=model_config,
            optimizer_config=training_config.get('optimizer', {}),
            scheduler_config=training_config.get('scheduler', {}),
            loss_config=training_config.get('loss', {}),
        )
    else:
        model = VLALightningModule(
            model_config=model_config,
            optimizer_config=training_config['optimizer'],
            scheduler_config=training_config['scheduler'],
            loss_config=training_config['loss'],
        )

    callbacks = setup_callbacks(training_config)
    logger = setup_logger(training_config)

    print("\nSetting up trainer...")
    trainer_config = training_config.get('trainer', {})
    trainer = pl.Trainer(
        max_epochs=trainer_config.get('max_epochs', 100),
        accelerator=trainer_config.get('accelerator', 'gpu'),
        devices=trainer_config.get('devices', 1),
        precision=trainer_config.get('precision', '16-mixed'),
        gradient_clip_val=trainer_config.get('gradient_clip_val', 1.0),
        accumulate_grad_batches=trainer_config.get('accumulate_grad_batches', 1),
        log_every_n_steps=trainer_config.get('log_every_n_steps', 50),
        val_check_interval=trainer_config.get('val_check_interval', 1.0),
        callbacks=callbacks,
        logger=logger,
        enable_checkpointing=trainer_config.get('enable_checkpointing', True),
        enable_progress_bar=trainer_config.get('enable_progress_bar', True),
        deterministic=training_config.get('deterministic', False),
    )

    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60 + "\n")

    if train_loader is None or val_loader is None:
        print("✗ 数据加载失败，终止训练。请检查 datasets/carla 是否存在。")
        return

    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=args.resume,
    )

    print("\n" + "=" * 60)
    print("Training completed!")
    print("=" * 60)
    print(f"Best model checkpoint: {callbacks[0].best_model_path}")


if __name__ == "__main__":
    main()
