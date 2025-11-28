"""
PyTorch Lightning Module for VLA Model Training

This module wraps the VLA model with training, validation, and optimization logic.
"""

from typing import Any, Dict, Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.vla_model import VLAModel


class VLALightningModule(pl.LightningModule):
    """
    Lightning module for training VLA model.
    """
    
    def __init__(
        self,
        model_config: Dict[str, Any],
        optimizer_config: Dict[str, Any],
        scheduler_config: Dict[str, Any],
        loss_config: Dict[str, Any],
    ):
        super().__init__()
        
        # Save hyperparameters
        self.save_hyperparameters()
        
        # Build model
        self.model = VLAModel(
            vision_config=model_config.get('vision_encoder'),
            language_config=model_config.get('language_encoder'),
            fusion_config=model_config.get('fusion'),
            decoder_config=model_config.get('action_decoder'),
        )
        
        # Loss weights
        self.steering_weight = loss_config.get('steering_weight', 1.0)
        self.throttle_weight = loss_config.get('throttle_weight', 0.5)
        self.brake_weight = loss_config.get('brake_weight', 0.5)
        self.loss_type = loss_config.get('loss_type', 'mse')
        
        # Metrics
        self.train_losses = []
        self.val_losses = []
    
    def forward(self, images: torch.Tensor, instructions: list) -> Dict[str, torch.Tensor]:
        """Forward pass."""
        return self.model(images, instructions)
    
    def compute_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute weighted loss for each action dimension.
        
        Args:
            predictions: Dict with 'steering', 'throttle', 'brake' [B, 1]
            targets: [B, 3] ground truth actions
        
        Returns:
            Dict with individual and total losses
        """
        # Select loss function
        if self.loss_type == 'mse':
            loss_fn = F.mse_loss
        elif self.loss_type == 'l1':
            loss_fn = F.l1_loss
        elif self.loss_type == 'smooth_l1':
            loss_fn = F.smooth_l1_loss
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        # Compute individual losses
        steering_loss = loss_fn(predictions['steering'], targets[:, 0:1])
        throttle_loss = loss_fn(predictions['throttle'], targets[:, 1:2])
        brake_loss = loss_fn(predictions['brake'], targets[:, 2:3])
        
        # Weighted total loss
        total_loss = (
            self.steering_weight * steering_loss +
            self.throttle_weight * throttle_loss +
            self.brake_weight * brake_loss
        )
        
        return {
            'loss': total_loss,
            'steering_loss': steering_loss,
            'throttle_loss': throttle_loss,
            'brake_loss': brake_loss,
        }
    
    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Training step."""
        images = batch['image']
        instructions = batch['instruction']
        actions = batch['actions']
        
        # Forward pass
        predictions = self(images, instructions)
        
        # Compute loss
        losses = self.compute_loss(predictions, actions)
        
        # Log metrics
        self.log('train_loss', losses['loss'], prog_bar=True)
        self.log('train_steering_loss', losses['steering_loss'])
        self.log('train_throttle_loss', losses['throttle_loss'])
        self.log('train_brake_loss', losses['brake_loss'])
        
        return losses['loss']
    
    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Validation step."""
        images = batch['image']
        instructions = batch['instruction']
        actions = batch['actions']
        
        # Forward pass
        predictions = self(images, instructions)
        
        # Compute loss
        losses = self.compute_loss(predictions, actions)
        
        # Log metrics
        self.log('val_loss', losses['loss'], prog_bar=True)
        self.log('val_steering_loss', losses['steering_loss'])
        self.log('val_throttle_loss', losses['throttle_loss'])
        self.log('val_brake_loss', losses['brake_loss'])
        
        return losses['loss']
    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        # Optimizer
        optimizer_config = self.hparams.optimizer_config
        optimizer = AdamW(
            self.parameters(),
            lr=optimizer_config.get('lr', 3e-4),
            weight_decay=optimizer_config.get('weight_decay', 0.01),
            betas=optimizer_config.get('betas', (0.9, 0.999)),
            eps=optimizer_config.get('eps', 1e-8),
        )
        
        # Scheduler
        scheduler_config = self.hparams.scheduler_config
        max_epochs = self.trainer.max_epochs
        
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=max_epochs,
            eta_min=scheduler_config.get('min_lr', 1e-6),
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1,
            }
        }
    
    def predict_step(
        self,
        batch: Dict[str, Any],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> Dict[str, torch.Tensor]:
        """Prediction step for inference."""
        images = batch['image']
        instructions = batch['instruction']
        
        predictions = self(images, instructions)
        
        return predictions
