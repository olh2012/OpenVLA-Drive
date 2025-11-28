"""
PyTorch Lightning Module for VLA Driving Policy Training

This module wraps the VLADrivingPolicy with training, validation, and optimization logic.
Optimized for LoRA-based fine-tuning of pre-trained VLMs.
"""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import pytorch_lightning as pl
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
    LIGHTNING_AVAILABLE = True
except ImportError:
    LIGHTNING_AVAILABLE = False
    print("Warning: PyTorch Lightning not available")

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.policy import VLADrivingPolicy


if LIGHTNING_AVAILABLE:
    class VLAPolicyLightningModule(pl.LightningModule):
        """
        Lightning module for training VLA Driving Policy with LoRA.
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
            
            # Extract configurations
            backbone_config = model_config.get('backbone', {})
            lora_config = model_config.get('lora', {})
            action_config = model_config.get('action_head', {})
            
            # Build VLA Driving Policy
            self.model = VLADrivingPolicy(
                model_name=backbone_config.get('model_name', 'microsoft/phi-2'),
                vision_model_name=backbone_config.get('vision_model_name', 'openai/clip-vit-base-patch32'),
                num_timesteps=action_config.get('num_timesteps', 10),
                action_head_hidden_dim=action_config.get('hidden_dim', 512),
                action_head_layers=action_config.get('num_layers', 3),
                use_lora=lora_config.get('use_lora', True),
                lora_config=lora_config,
                freeze_vision_tower=backbone_config.get('freeze_vision_tower', True),
                freeze_llm=backbone_config.get('freeze_llm', True),
            )
            
            # Loss configuration
            self.trajectory_weight = loss_config.get('trajectory_weight', 1.0)
            self.loss_type = loss_config.get('loss_type', 'smooth_l1')
            
            # Metrics tracking
            self.train_losses = []
            self.val_losses = []
        
        def forward(
            self,
            image_tensors: torch.Tensor,
            input_ids: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
        ) -> Dict[str, torch.Tensor]:
            """Forward pass."""
            return self.model(
                image_tensors=image_tensors,
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_language_output=False,
            )
        
        def compute_trajectory_loss(
            self,
            predicted_trajectory: torch.Tensor,
            target_trajectory: torch.Tensor,
        ) -> torch.Tensor:
            """
            Compute trajectory prediction loss.
            
            Args:
                predicted_trajectory: [B, T, 2] predicted waypoints
                target_trajectory: [B, T, 2] ground truth waypoints
            
            Returns:
                loss: Scalar loss value
            """
            # Select loss function
            if self.loss_type == 'mse':
                loss = F.mse_loss(predicted_trajectory, target_trajectory)
            elif self.loss_type == 'l1':
                loss = F.l1_loss(predicted_trajectory, target_trajectory)
            elif self.loss_type == 'smooth_l1':
                loss = F.smooth_l1_loss(predicted_trajectory, target_trajectory)
            else:
                raise ValueError(f"Unknown loss type: {self.loss_type}")
            
            return loss * self.trajectory_weight
        
        def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
            """Training step."""
            images = batch['image']
            input_ids = batch['input_ids']
            attention_mask = batch.get('attention_mask', None)
            target_trajectory = batch['trajectory']  # [B, T, 2]
            
            # Forward pass
            outputs = self(
                image_tensors=images,
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            
            predicted_trajectory = outputs['trajectory']
            
            # Compute loss
            loss = self.compute_trajectory_loss(predicted_trajectory, target_trajectory)
            
            # Compute metrics
            with torch.no_grad():
                # Average Displacement Error (ADE)
                ade = torch.mean(torch.norm(predicted_trajectory - target_trajectory, dim=-1))
                
                # Final Displacement Error (FDE)
                fde = torch.mean(torch.norm(predicted_trajectory[:, -1] - target_trajectory[:, -1], dim=-1))
            
            # Log metrics
            self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
            self.log('train_ade', ade, prog_bar=True, on_step=False, on_epoch=True)
            self.log('train_fde', fde, prog_bar=False, on_step=False, on_epoch=True)
            
            return loss
        
        def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
            """Validation step."""
            images = batch['image']
            input_ids = batch['input_ids']
            attention_mask = batch.get('attention_mask', None)
            target_trajectory = batch['trajectory']
            
            # Forward pass
            outputs = self(
                image_tensors=images,
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            
            predicted_trajectory = outputs['trajectory']
            
            # Compute loss
            loss = self.compute_trajectory_loss(predicted_trajectory, target_trajectory)
            
            # Compute metrics
            with torch.no_grad():
                ade = torch.mean(torch.norm(predicted_trajectory - target_trajectory, dim=-1))
                fde = torch.mean(torch.norm(predicted_trajectory[:, -1] - target_trajectory[:, -1], dim=-1))
            
            # Log metrics
            self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
            self.log('val_ade', ade, prog_bar=True, on_step=False, on_epoch=True)
            self.log('val_fde', fde, prog_bar=False, on_step=False, on_epoch=True)
            
            return loss
        
        def configure_optimizers(self):
            """Configure optimizer and learning rate scheduler."""
            # Get only trainable parameters (LoRA adapters + Action Head)
            trainable_params = self.model.get_trainable_parameters()
            
            print(f"\nOptimizing {len(trainable_params)} parameter groups")
            
            # Optimizer
            optimizer_config = self.hparams.optimizer_config
            optimizer = AdamW(
                trainable_params.values(),
                lr=optimizer_config.get('lr', 2e-4),
                weight_decay=optimizer_config.get('weight_decay', 0.01),
                betas=optimizer_config.get('betas', (0.9, 0.999)),
                eps=optimizer_config.get('eps', 1e-8),
            )
            
            # Scheduler
            scheduler_config = self.hparams.scheduler_config
            scheduler_name = scheduler_config.get('name', 'cosine')
            
            if scheduler_name == 'cosine':
                max_epochs = self.trainer.max_epochs
                scheduler = CosineAnnealingLR(
                    optimizer,
                    T_max=max_epochs,
                    eta_min=scheduler_config.get('min_lr', 1e-6),
                )
                interval = 'epoch'
            elif scheduler_name == 'onecycle':
                # Estimate total steps
                steps_per_epoch = len(self.trainer.datamodule.train_dataloader()) if hasattr(self.trainer, 'datamodule') else 1000
                total_steps = self.trainer.max_epochs * steps_per_epoch
                
                scheduler = OneCycleLR(
                    optimizer,
                    max_lr=optimizer_config.get('lr', 2e-4),
                    total_steps=total_steps,
                    pct_start=scheduler_config.get('warmup_pct', 0.1),
                    anneal_strategy='cos',
                )
                interval = 'step'
            else:
                raise ValueError(f"Unknown scheduler: {scheduler_name}")
            
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': interval,
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
            instructions = batch.get('instruction', None)
            
            if instructions is not None:
                # Use text instructions
                trajectory = self.model.predict_trajectory(
                    image_tensors=images,
                    text_instructions=instructions
                )
            else:
                # Use pre-tokenized inputs
                input_ids = batch['input_ids']
                attention_mask = batch.get('attention_mask', None)
                
                outputs = self(
                    image_tensors=images,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                trajectory = outputs['trajectory']
            
            return {'trajectory': trajectory}
        
        def on_train_epoch_end(self):
            """Called at the end of training epoch."""
            # Print trainable parameters summary periodically
            if self.current_epoch % 10 == 0:
                print(f"\nEpoch {self.current_epoch} - Trainable Parameters:")
                self.model.print_trainable_parameters()
else:
    # Dummy class when Lightning is not available
    class VLAPolicyLightningModule:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch Lightning is not installed. Please install it to use this module.")
