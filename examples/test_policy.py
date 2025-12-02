#!/usr/bin/env python3
"""
Example script demonstrating VLADrivingPolicy usage.

This script shows how to:
1. Initialize the VLA driving policy with pre-trained VLM
2. Prepare inputs (images and text instructions)
3. Run inference to get trajectory predictions
4. Analyze trainable parameters (LoRA + Action Head only)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import yaml
from PIL import Image
import numpy as np

from models.policy import VLADrivingPolicy


def create_dummy_data(batch_size=2, image_size=(224, 224)):
    """Create dummy data for testing."""
    # Create random images
    images = torch.randn(batch_size, 3, *image_size)
    
    # Example text instructions
    instructions = [
        "Follow the lane and maintain safe distance from the vehicle ahead",
        "Turn left at the next intersection and stop at the traffic light",
    ]
    
    # Dummy ground truth waypoints (for training)
    gt_waypoints = torch.randn(batch_size, 10, 2) * 10  # [B, T, 2]
    
    return images, instructions, gt_waypoints


def main():
    print("=" * 70)
    print("VLA Driving Policy Example")
    print("=" * 70)
    print()
    
    # Load configuration
    config_path = "configs/policy_config.yaml"
    if os.path.exists(config_path):
        print(f"Loading config from {config_path}")
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        model_config = config.get('model', {})
        lora_config = model_config.get('lora', {})
        action_config = model_config.get('action_head', {})
        backbone_config = model_config.get('backbone', {})
        multi_task_config = model_config.get('multi_task', {})
    else:
        print("Config file not found, using default configuration")
        lora_config = {'r': 16, 'lora_alpha': 32, 'lora_dropout': 0.05}
        action_config = {'num_timesteps': 10, 'hidden_dim': 512}
        backbone_config = {
            'model_name': 'microsoft/phi-2',  # Smaller model for testing
            'vision_model_name': 'openai/clip-vit-base-patch32',
        }
        multi_task_config = {'enabled': True}
    
    # Initialize model
    print("\nInitializing VLADrivingPolicy...")
    print("-" * 70)
    
    model = VLADrivingPolicy(
        model_name=backbone_config.get('model_name', 'microsoft/phi-2'),
        vision_model_name=backbone_config.get('vision_model_name', 'openai/clip-vit-base-patch32'),
        num_timesteps=action_config.get('num_timesteps', 10),
        action_head_hidden_dim=action_config.get('hidden_dim', 512),
        action_head_layers=action_config.get('num_layers', 3),
        use_lora=lora_config.get('use_lora', True),
        lora_config=lora_config,
        freeze_vision_tower=backbone_config.get('freeze_vision_tower', True),
        freeze_llm=backbone_config.get('freeze_llm', True),
        multi_task_config=multi_task_config,
    )
    
    print("\n" + "=" * 70)
    print("Model Architecture Summary")
    print("=" * 70)
    print(f"Vision Tower: {backbone_config.get('vision_model_name')}")
    print(f"LLM Backbone: {backbone_config.get('model_name')}")
    print(f"Using LoRA: {lora_config.get('use_lora', True)}")
    print(f"Trajectory Timesteps: {action_config.get('num_timesteps', 10)}")
    print()
    
    # Print trainable parameters
    print("=" * 70)
    print("Trainable Parameters Analysis")
    print("=" * 70)
    model.print_trainable_parameters()
    print()
    
    # Set to evaluation mode
    model.eval()
    
    # Create dummy input data
    print("=" * 70)
    print("Running Inference Example")
    print("=" * 70)
    
    images, instructions, gt_waypoints = create_dummy_data(batch_size=2)
    
    print(f"\nInput:")
    print(f"  Images shape: {images.shape}")
    print(f"  Instructions: {len(instructions)} samples")
    print(f"    - '{instructions[0]}'")
    print(f"    - '{instructions[1]}'")
    
    # Run inference
    print("\nRunning forward pass...")
    
    try:
        with torch.no_grad():
            # Method 1: Using predict_trajectory with auxiliary outputs
            outputs = model.predict_trajectory(
                image_tensors=images,
                text_instructions=instructions,
                return_aux=True,
            )
            trajectory = outputs['trajectory']
            
            print(f"\nOutput:")
            print(f"  Predicted trajectory shape: {trajectory.shape}")
            print(f"  Expected shape: [batch_size={len(instructions)}, "
                  f"timesteps={action_config.get('num_timesteps', 10)}, coords=2]")
            
            # Show sample predictions
            print(f"\nSample trajectory (first batch):")
            print(f"  Waypoint 0: x={trajectory[0, 0, 0].item():.3f}, "
                  f"y={trajectory[0, 0, 1].item():.3f}")
            print(f"  Waypoint 1: x={trajectory[0, 1, 0].item():.3f}, "
                  f"y={trajectory[0, 1, 1].item():.3f}")
            print(f"  ...")
            print(f"  Waypoint {action_config.get('num_timesteps', 10)-1}: "
                  f"x={trajectory[0, -1, 0].item():.3f}, "
                  f"y={trajectory[0, -1, 1].item():.3f}")
            
            if 'multi_task' in outputs:
                mt = outputs['multi_task']
                nav_labels = outputs.get('navigation_labels', [])
                print("\nMulti-task heads:")
                if 'navigation_logits' in mt:
                    probs = torch.softmax(mt['navigation_logits'][0], dim=-1)
                    top_idx = torch.argmax(probs).item()
                    label = nav_labels[top_idx] if nav_labels else f"class_{top_idx}"
                    print(f"  - Navigation: {label} ({probs[top_idx].item():.2f})")
                if 'lane_offset' in mt:
                    print(f"  - Lane offset: {mt['lane_offset'][0, 0].item():+.3f}")
                if 'obstacle_score' in mt:
                    print(f"  - Obstacle score: {mt['obstacle_score'][0, 0].item():.3f}")
            
            print("\n✓ Inference successful!")
            
    except Exception as e:
        print(f"\n✗ Error during inference: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Training example
    print("\n" + "=" * 70)
    print("Training Example (Forward + Backward)")
    print("=" * 70)
    
    model.train()
    
    # Create optimizer (only for trainable parameters)
    optimizer = torch.optim.AdamW(
        model.get_trainable_parameters().values(),
        lr=2e-4,
        weight_decay=0.01
    )
    
    print(f"Optimizer parameters: {len(list(optimizer.param_groups[0]['params']))}")
    
    try:
        # Tokenize instructions
        if model.processor is not None:
            encoding = model.processor(
                text=instructions,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
        else:
            encoding = model.tokenizer(
                instructions,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
        
        input_ids = encoding['input_ids']
        attention_mask = encoding.get('attention_mask', None)
        
        # Forward pass
        outputs = model(
            image_tensors=images,
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_language_output=False,
        )
        
        predicted_trajectory = outputs['trajectory']
        
        # Compute loss (simple MSE for demonstration)
        loss = torch.nn.functional.mse_loss(predicted_trajectory, gt_waypoints)
        
        print(f"\nTraining step:")
        print(f"  Predicted shape: {predicted_trajectory.shape}")
        print(f"  Ground truth shape: {gt_waypoints.shape}")
        print(f"  Loss: {loss.item():.4f}")
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print("  ✓ Backward pass successful")
        print("  ✓ Only LoRA adapters + Action Head were updated")
        
    except Exception as e:
        print(f"\n✗ Error during training: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print("✓ Model initialization successful")
    print("✓ LoRA configuration applied")
    print("✓ Action head integrated")
    print("✓ Inference working")
    print("✓ Training loop working")
    print("\nThe model is ready for training on CARLA driving data!")
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
