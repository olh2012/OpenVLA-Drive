"""
Training Script for VLA Model

This script handles the complete training pipeline using PyTorch Lightning.
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import yaml
from omegaconf import OmegaConf

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

from data.carla_dataset import get_carla_dataloader
from training.lightning_module import VLALightningModule


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_callbacks(config: dict):
    """Setup training callbacks."""
    callbacks = []
    
    # Checkpoint callback
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
    
    # Early stopping callback
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


def main():
    parser = argparse.ArgumentParser(description="Train VLA Model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training_config.yaml",
        help="Path to training config file",
    )
    parser.add_argument(
        "--model_config",
        type=str,
        default="configs/model_config.yaml",
        help="Path to model config file",
    )
    parser.add_argument(
        "--data_config",
        type=str,
        default="configs/data_config.yaml",
        help="Path to data config file",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from",
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("VLA Model Training")
    print("=" * 60)
    
    # Load configurations
    print(f"Loading configs...")
    training_config = load_config(args.config)
    model_config = load_config(args.model_config)
    data_config = load_config(args.data_config)
    
    print(f"Training config: {args.config}")
    print(f"Model config: {args.model_config}")
    print(f"Data config: {args.data_config}")
    
    # Set random seed
    seed = training_config.get('seed', 42)
    pl.seed_everything(seed)
    print(f"Random seed: {seed}")
    
    # Create data loaders
    print("\nSetting up data loaders...")
    
    try:
        train_loader = get_carla_dataloader(
            data_root=data_config['dataset']['data_root'],
            split='train',
            batch_size=data_config['train']['batch_size'],
            num_workers=data_config['train']['num_workers'],
            shuffle=data_config['train']['shuffle'],
            image_size=tuple(data_config['dataset']['image_size']),
        )
        
        val_loader = get_carla_dataloader(
            data_root=data_config['dataset']['data_root'],
            split='val',
            batch_size=data_config['val']['batch_size'],
            num_workers=data_config['val']['num_workers'],
            shuffle=data_config['val']['shuffle'],
            image_size=tuple(data_config['dataset']['image_size']),
        )
        
        print(f"Train samples: {len(train_loader.dataset)}")
        print(f"Val samples: {len(val_loader.dataset)}")
    except Exception as e:
        print(f"Warning: Could not load data - {e}")
        print("Proceeding without data loaders (for testing purposes)")
        train_loader = None
        val_loader = None
    
    # Create model
    print("\nInitializing model...")
    model = VLALightningModule(
        model_config=model_config,
        optimizer_config=training_config['optimizer'],
        scheduler_config=training_config['scheduler'],
        loss_config=training_config['loss'],
    )
    
    # Setup callbacks and logger
    callbacks = setup_callbacks(training_config)
    logger = setup_logger(training_config)
    
    # Create trainer
    print("\nSetting up trainer...")
    trainer_config = training_config['trainer']
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
    
    # Start training
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60 + "\n")
    
    if train_loader is not None and val_loader is not None:
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
    else:
        print("\nSkipping training - no data available")
        print("Please prepare CARLA dataset first")


if __name__ == "__main__":
    main()
