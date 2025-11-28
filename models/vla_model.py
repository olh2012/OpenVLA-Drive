"""
Vision-Language-Action (VLA) Model for Autonomous Driving

This module implements the core VLA architecture that combines:
- Vision Encoder: Processes camera images
- Language Encoder: Processes text instructions
- Fusion Module: Combines vision and language features
- Action Decoder: Outputs driving control actions
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class VisionEncoder(nn.Module):
    """Vision encoder for processing RGB images."""
    
    def __init__(
        self,
        model_type: str = "resnet50",
        pretrained: bool = True,
        output_dim: int = 2048,
    ):
        super().__init__()
        self.model_type = model_type
        self.output_dim = output_dim
        
        # Load pretrained vision model
        if model_type == "resnet50":
            from torchvision.models import resnet50, ResNet50_Weights
            weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            backbone = resnet50(weights=weights)
            # Remove classification head
            self.encoder = nn.Sequential(*list(backbone.children())[:-1])
            self.feat_dim = 2048
        else:
            raise NotImplementedError(f"Vision encoder {model_type} not implemented")
        
        # Projection layer
        self.projection = nn.Linear(self.feat_dim, output_dim)
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: [B, C, H, W]
        Returns:
            features: [B, output_dim]
        """
        features = self.encoder(images)
        features = features.flatten(1)
        features = self.projection(features)
        return features


class LanguageEncoder(nn.Module):
    """Language encoder for processing text instructions."""
    
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        output_dim: int = 768,
        max_length: int = 128,
    ):
        super().__init__()
        self.model_name = model_name
        self.output_dim = output_dim
        self.max_length = max_length
        
        # Load pretrained language model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        
        # Get hidden size from model config
        self.hidden_size = self.encoder.config.hidden_size
        
        # Projection layer
        self.projection = nn.Linear(self.hidden_size, output_dim)
    
    def forward(self, texts: list) -> torch.Tensor:
        """
        Args:
            texts: List of text strings
        Returns:
            features: [B, output_dim]
        """
        # Tokenize
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Move to same device as model
        device = next(self.encoder.parameters()).device
        encoded = {k: v.to(device) for k, v in encoded.items()}
        
        # Encode
        outputs = self.encoder(**encoded)
        
        # Use [CLS] token embedding
        features = outputs.last_hidden_state[:, 0, :]
        features = self.projection(features)
        
        return features


class CrossAttentionFusion(nn.Module):
    """Cross-attention fusion module for vision and language."""
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        num_layers: int = 4,
    ):
        super().__init__()
        
        # Cross-attention layers
        self.layers = nn.ModuleList([
            nn.MultiheadAttention(dim, num_heads, batch_first=True)
            for _ in range(num_layers)
        ])
        
        # Layer norms
        self.norms = nn.ModuleList([
            nn.LayerNorm(dim) for _ in range(num_layers)
        ])
    
    def forward(
        self,
        vision_feats: torch.Tensor,
        language_feats: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            vision_feats: [B, vision_dim]
            language_feats: [B, lang_dim]
        Returns:
            fused_feats: [B, dim]
        """
        # Add sequence dimension
        vision_feats = vision_feats.unsqueeze(1)  # [B, 1, dim]
        language_feats = language_feats.unsqueeze(1)  # [B, 1, dim]
        
        x = vision_feats
        for attn, norm in zip(self.layers, self.norms):
            # Cross-attention: query from vision, key/value from language
            attn_out, _ = attn(x, language_feats, language_feats)
            x = norm(x + attn_out)
        
        # Remove sequence dimension
        fused_feats = x.squeeze(1)
        
        return fused_feats


class ActionDecoder(nn.Module):
    """Decoder for outputting driving control actions."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        num_layers: int = 3,
        action_dim: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        layers = []
        in_dim = input_dim
        
        for _ in range(num_layers - 1):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(in_dim, action_dim))
        
        self.decoder = nn.Sequential(*layers)
        
        # Action heads
        self.steering_head = nn.Tanh()  # [-1, 1]
        self.throttle_head = nn.Sigmoid()  # [0, 1]
        self.brake_head = nn.Sigmoid()  # [0, 1]
    
    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            features: [B, input_dim]
        Returns:
            actions: Dict with 'steering', 'throttle', 'brake' [B, 1] each
        """
        x = self.decoder(features)
        
        steering = self.steering_head(x[:, 0:1])
        throttle = self.throttle_head(x[:, 1:2])
        brake = self.brake_head(x[:, 2:3])
        
        return {
            'steering': steering,
            'throttle': throttle,
            'brake': brake,
        }


class VLAModel(nn.Module):
    """
    Complete Vision-Language-Action model for autonomous driving.
    """
    
    def __init__(
        self,
        vision_config: Optional[Dict] = None,
        language_config: Optional[Dict] = None,
        fusion_config: Optional[Dict] = None,
        decoder_config: Optional[Dict] = None,
    ):
        super().__init__()
        
        # Default configs
        vision_config = vision_config or {}
        language_config = language_config or {}
        fusion_config = fusion_config or {}
        decoder_config = decoder_config or {}
        
        # Build components
        self.vision_encoder = VisionEncoder(**vision_config)
        self.language_encoder = LanguageEncoder(**language_config)
        
        # Fusion module
        fusion_dim = vision_config.get('output_dim', 2048)
        self.fusion = CrossAttentionFusion(
            dim=fusion_dim,
            **fusion_config
        )
        
        # Action decoder
        self.action_decoder = ActionDecoder(
            input_dim=fusion_dim,
            **decoder_config
        )
    
    def forward(
        self,
        images: torch.Tensor,
        instructions: list,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of VLA model.
        
        Args:
            images: [B, C, H, W] RGB images
            instructions: List of text instructions (length B)
        
        Returns:
            actions: Dict with 'steering', 'throttle', 'brake'
        """
        # Encode vision and language
        vision_feats = self.vision_encoder(images)
        language_feats = self.language_encoder(instructions)
        
        # Fuse features
        fused_feats = self.fusion(vision_feats, language_feats)
        
        # Decode actions
        actions = self.action_decoder(fused_feats)
        
        return actions
