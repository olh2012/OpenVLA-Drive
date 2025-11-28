"""
VLA Driving Policy with Pre-trained VLM Backbone

This module implements a vision-language-action policy for autonomous driving
using pre-trained models (e.g., LLaVA, Phi-3-Vision) with LoRA adaptation.
"""

from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import AutoModel, AutoProcessor, AutoTokenizer, CLIPVisionModel
from peft import LoraConfig, get_peft_model, TaskType


class ActionHead(nn.Module):
    """
    Multi-Layer Perceptron for trajectory prediction.
    
    Predicts waypoints (x, y) for T future timesteps.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        num_layers: int = 3,
        num_timesteps: int = 10,
        dropout: float = 0.1,
    ):
        """
        Args:
            input_dim: Dimension of input features (LLM hidden size)
            hidden_dim: Dimension of hidden layers
            num_layers: Number of MLP layers
            num_timesteps: Number of future timesteps to predict (T)
            dropout: Dropout probability
        """
        super().__init__()
        
        self.num_timesteps = num_timesteps
        self.output_dim = num_timesteps * 2  # T x (x, y)
        
        # Build MLP layers
        layers = []
        in_dim = input_dim
        
        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(in_dim, self.output_dim))
        
        self.mlp = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            features: [B, input_dim] LLM hidden states
        
        Returns:
            waypoints: [B, T, 2] predicted (x, y) waypoints
        """
        # Project to trajectory
        trajectory_flat = self.mlp(features)  # [B, T*2]
        
        # Reshape to [B, T, 2]
        batch_size = trajectory_flat.shape[0]
        waypoints = trajectory_flat.view(batch_size, self.num_timesteps, 2)
        
        return waypoints


class VLADrivingPolicy(nn.Module):
    """
    Vision-Language-Action Driving Policy.
    
    Integrates pre-trained VLM (e.g., LLaVA, Phi-3-Vision) with LoRA adapters
    and adds an action head for trajectory prediction.
    """
    
    def __init__(
        self,
        model_name: str = "llava-hf/llava-1.5-7b-hf",
        vision_model_name: Optional[str] = "openai/clip-vit-large-patch14",
        num_timesteps: int = 10,
        action_head_hidden_dim: int = 512,
        action_head_layers: int = 3,
        use_lora: bool = True,
        lora_config: Optional[Dict] = None,
        freeze_vision_tower: bool = True,
        freeze_llm: bool = True,
    ):
        """
        Args:
            model_name: HuggingFace model name for the VLM backbone
            vision_model_name: Vision encoder model name (CLIP)
            num_timesteps: Number of future waypoints to predict (T)
            action_head_hidden_dim: Hidden dimension for action head MLP
            action_head_layers: Number of layers in action head
            use_lora: Whether to use LoRA for efficient fine-tuning
            lora_config: Custom LoRA configuration
            freeze_vision_tower: Whether to freeze vision encoder
            freeze_llm: Whether to freeze LLM (except LoRA adapters)
        """
        super().__init__()
        
        self.model_name = model_name
        self.num_timesteps = num_timesteps
        self.use_lora = use_lora
        
        # Load vision encoder (CLIP)
        print(f"Loading vision encoder: {vision_model_name}")
        self.vision_tower = CLIPVisionModel.from_pretrained(vision_model_name)
        self.vision_hidden_size = self.vision_tower.config.hidden_size
        
        if freeze_vision_tower:
            print("Freezing vision tower parameters")
            for param in self.vision_tower.parameters():
                param.requires_grad = False
        
        # Load VLM backbone (e.g., LLaVA, Phi-3-Vision)
        print(f"Loading VLM backbone: {model_name}")
        try:
            self.llm_backbone = AutoModel.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )
        except Exception as e:
            print(f"Warning: Could not load {model_name}, using fallback model")
            # Fallback to a smaller model for testing
            self.llm_backbone = AutoModel.from_pretrained(
                "microsoft/phi-2",
                torch_dtype=torch.float16,
                trust_remote_code=True,
            )
        
        # Get LLM hidden size
        if hasattr(self.llm_backbone.config, 'hidden_size'):
            self.llm_hidden_size = self.llm_backbone.config.hidden_size
        elif hasattr(self.llm_backbone.config, 'd_model'):
            self.llm_hidden_size = self.llm_backbone.config.d_model
        else:
            self.llm_hidden_size = 2048  # Default fallback
        
        # Apply LoRA if requested
        if use_lora:
            print("Applying LoRA configuration")
            if lora_config is None:
                # Default LoRA config
                lora_config = {
                    'r': 16,
                    'lora_alpha': 32,
                    'target_modules': ['q_proj', 'v_proj', 'k_proj', 'o_proj'],
                    'lora_dropout': 0.05,
                    'bias': 'none',
                }
            
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=lora_config.get('r', 16),
                lora_alpha=lora_config.get('lora_alpha', 32),
                lora_dropout=lora_config.get('lora_dropout', 0.05),
                target_modules=lora_config.get('target_modules', ['q_proj', 'v_proj']),
                bias=lora_config.get('bias', 'none'),
            )
            
            self.llm_backbone = get_peft_model(self.llm_backbone, peft_config)
            self.llm_backbone.print_trainable_parameters()
        elif freeze_llm:
            # Freeze LLM if not using LoRA
            print("Freezing LLM backbone parameters")
            for param in self.llm_backbone.parameters():
                param.requires_grad = False
        
        # Vision-Language projection layer
        self.vision_projection = nn.Linear(
            self.vision_hidden_size,
            self.llm_hidden_size
        )
        
        # Action head for trajectory prediction
        print(f"Initializing action head (T={num_timesteps})")
        self.action_head = ActionHead(
            input_dim=self.llm_hidden_size,
            hidden_dim=action_head_hidden_dim,
            num_layers=action_head_layers,
            num_timesteps=num_timesteps,
            dropout=0.1,
        )
        
        # Load processor/tokenizer
        try:
            self.processor = AutoProcessor.from_pretrained(model_name)
        except:
            self.processor = None
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name if "phi" not in model_name.lower() else "microsoft/phi-2",
                trust_remote_code=True
            )
    
    def encode_images(self, image_tensors: torch.Tensor) -> torch.Tensor:
        """
        Encode images using vision tower.
        
        Args:
            image_tensors: [B, C, H, W] image tensors
        
        Returns:
            vision_features: [B, hidden_size] vision features
        """
        with torch.no_grad() if not self.vision_tower.training else torch.enable_grad():
            # Get vision features
            vision_outputs = self.vision_tower(pixel_values=image_tensors)
            
            # Use pooled output or mean pooling
            if hasattr(vision_outputs, 'pooler_output'):
                vision_features = vision_outputs.pooler_output
            else:
                # Mean pooling over spatial dimensions
                vision_features = vision_outputs.last_hidden_state.mean(dim=1)
        
        # Project to LLM dimension
        vision_features = self.vision_projection(vision_features)
        
        return vision_features
    
    def forward(
        self,
        image_tensors: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_language_output: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the VLA driving policy.
        
        Args:
            image_tensors: [B, C, H, W] preprocessed images
            input_ids: [B, seq_len] tokenized text instructions
            attention_mask: [B, seq_len] attention mask for text
            return_language_output: Whether to return language generation output
        
        Returns:
            Dict containing:
                - 'trajectory': [B, T, 2] predicted waypoints
                - 'logits': [B, seq_len, vocab_size] language model logits (optional)
                - 'hidden_states': [B, hidden_size] last hidden state
        """
        batch_size = image_tensors.shape[0]
        device = image_tensors.device
        
        # Encode images
        vision_features = self.encode_images(image_tensors)  # [B, hidden_size]
        
        # Get LLM outputs
        # Note: Different VLMs have different interfaces
        # This is a simplified version - you may need to adapt for specific models
        try:
            llm_outputs = self.llm_backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
        except:
            # Fallback for models with different interfaces
            llm_outputs = self.llm_backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )
        
        # Extract last hidden state
        if hasattr(llm_outputs, 'hidden_states') and llm_outputs.hidden_states is not None:
            # Use the last layer's hidden state
            last_hidden_state = llm_outputs.hidden_states[-1]
        elif hasattr(llm_outputs, 'last_hidden_state'):
            last_hidden_state = llm_outputs.last_hidden_state
        else:
            raise AttributeError("Could not extract hidden states from LLM output")
        
        # Pool the sequence: use [CLS] token or mean pooling
        if attention_mask is not None:
            # Mean pooling with attention mask
            mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            sum_hidden = torch.sum(last_hidden_state * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            pooled_features = sum_hidden / sum_mask
        else:
            # Simple mean pooling
            pooled_features = last_hidden_state.mean(dim=1)
        
        # Fuse vision and language features
        fused_features = pooled_features + vision_features  # [B, hidden_size]
        
        # Predict trajectory using action head
        trajectory = self.action_head(fused_features)  # [B, T, 2]
        
        # Prepare output
        output = {
            'trajectory': trajectory,
            'hidden_states': fused_features,
        }
        
        # Optionally include language model logits
        if return_language_output and hasattr(llm_outputs, 'logits'):
            output['logits'] = llm_outputs.logits
        
        return output
    
    def predict_trajectory(
        self,
        image_tensors: torch.Tensor,
        text_instructions: list,
    ) -> torch.Tensor:
        """
        Convenience method for trajectory prediction during inference.
        
        Args:
            image_tensors: [B, C, H, W] images
            text_instructions: List of text instructions (length B)
        
        Returns:
            trajectory: [B, T, 2] predicted waypoints
        """
        # Tokenize text
        if self.processor is not None:
            encoding = self.processor(
                text=text_instructions,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
        else:
            encoding = self.tokenizer(
                text_instructions,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
        
        input_ids = encoding['input_ids'].to(image_tensors.device)
        attention_mask = encoding.get('attention_mask', None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(image_tensors.device)
        
        # Forward pass
        with torch.no_grad():
            outputs = self.forward(
                image_tensors=image_tensors,
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_language_output=False,
            )
        
        return outputs['trajectory']
    
    def get_trainable_parameters(self) -> Dict[str, torch.Tensor]:
        """
        Get only trainable parameters.
        
        Returns:
            Dict of parameter names to tensors
        """
        trainable_params = {}
        for name, param in self.named_parameters():
            if param.requires_grad:
                trainable_params[name] = param
        
        return trainable_params
    
    def print_trainable_parameters(self):
        """Print trainable parameter statistics."""
        trainable_params = 0
        all_params = 0
        
        for name, param in self.named_parameters():
            num_params = param.numel()
            all_params += num_params
            if param.requires_grad:
                trainable_params += num_params
                print(f"  Trainable: {name} - {num_params:,} params")
        
        print(f"\nTrainable params: {trainable_params:,} || "
              f"All params: {all_params:,} || "
              f"Trainable %: {100 * trainable_params / all_params:.2f}%")
