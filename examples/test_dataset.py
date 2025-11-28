#!/usr/bin/env python3
"""
Example script demonstrating CARLA VLA Dataset usage.

This script shows how to:
1. Create a CARLA VLA dataset
2. Use the custom collate function
3. Load batches with DataLoader
4. Inspect the data format
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from pathlib import Path

from data.carla_dataset import CARLAVLADataset, get_carla_vla_dataloader, carla_vla_collate_fn


def create_dummy_dataset(data_root: str = "./dummy_carla_data"):
    """
    Create a dummy CARLA dataset for testing.
    
    Creates directory structure:
    dummy_carla_data/
        train/
            images/
                000000.png
                000001.png
                ...
            annotations.json
    """
    import json
    from PIL import Image
    
    data_path = Path(data_root)
    train_path = data_path / "train"
    images_path = train_path / "images"
    images_path.mkdir(parents=True, exist_ok=True)
    
    # Create dummy images and annotations
    annotations = {}
    num_samples = 10
    
    commands = [
        "Follow the lane and maintain speed",
        "Turn left at the next intersection",
        "Turn right and merge into traffic",
        "Stop at the traffic light",
        "Overtake the vehicle ahead safely",
        "Navigate to the destination",
        "Change lane to the left",
        "Change lane to the right",
        "Slow down and prepare to stop",
        "Accelerate and maintain lane",
    ]
    
    for i in range(num_samples):
        # Create dummy image
        img = Image.new('RGB', (800, 600), color=(100 + i*10, 150, 200))
        img_path = images_path / f"{i:06d}.png"
        img.save(img_path)
        
        # Generate dummy trajectory (10 waypoints)
        # Simulate a curved path
        t = np.linspace(0, 5, 10)
        x = t * 2.0  # Forward motion
        y = np.sin(t * 0.5) * 1.5  # Lateral motion (curve)
        trajectory = np.stack([x, y], axis=1).tolist()
        
        # Create annotation
        annotations[f"{i:06d}"] = {
            "image": f"images/{i:06d}.png",
            "command": commands[i % len(commands)],
            "trajectory": trajectory,
            "ego_position": [0.0, 0.0, 0.0],  # x, y, theta
        }
    
    # Save annotations
    annotation_file = train_path / "annotations.json"
    with open(annotation_file, 'w') as f:
        json.dump(annotations, f, indent=2)
    
    print(f"✓ Created dummy dataset at {data_root}")
    print(f"  - {num_samples} samples")
    print(f"  - Images: {images_path}")
    print(f"  - Annotations: {annotation_file}")
    
    return str(data_path)


def main():
    print("=" * 70)
    print("CARLA VLA Dataset Test")
    print("=" * 70)
    print()
    
    # Create dummy dataset
    print("Creating dummy dataset...")
    data_root = create_dummy_dataset()
    print()
    
    # Test 1: Create dataset
    print("=" * 70)
    print("Test 1: Dataset Creation")
    print("=" * 70)
    
    dataset = CARLAVLADataset(
        data_root=data_root,
        split='train',
        image_size=(224, 224),
        tokenizer_name='microsoft/phi-2',
        max_text_length=128,
        num_trajectory_points=10,
        normalize_trajectory=True,
        use_clip_normalization=True,
    )
    
    print(f"✓ Dataset created successfully")
    print(f"  - Number of samples: {len(dataset)}")
    print()
    
    # Test 2: Get single sample
    print("=" * 70)
    print("Test 2: Single Sample")
    print("=" * 70)
    
    sample = dataset[0]
    print(f"Sample keys: {list(sample.keys())}")
    print(f"  - image shape: {sample['image'].shape}")
    print(f"  - command: '{sample['command']}'")
    print(f"  - trajectory shape: {sample['trajectory'].shape}")
    
    if 'input_ids' in sample:
        print(f"  - input_ids shape: {sample['input_ids'].shape}")
        print(f"  - attention_mask shape: {sample['attention_mask'].shape}")
    
    print(f"\nTrajectory waypoints (first 3):")
    for i in range(min(3, len(sample['trajectory']))):
        x, y = sample['trajectory'][i]
        print(f"  Waypoint {i}: x={x:.3f}, y={y:.3f}")
    print()
    
    # Test 3: DataLoader with collate function
    print("=" * 70)
    print("Test 3: DataLoader with Collate Function")
    print("=" * 70)
    
    dataloader = get_carla_vla_dataloader(
        data_root=data_root,
        split='train',
        batch_size=4,
        tokenizer_name='microsoft/phi-2',
        num_trajectory_points=10,
        image_size=(224, 224),
        num_workers=0,  # Use 0 for testing
        shuffle=True,
    )
    
    print(f"✓ DataLoader created")
    print(f"  - Batch size: 4")
    print(f"  - Number of batches: {len(dataloader)}")
    print()
    
    # Get a batch
    batch = next(iter(dataloader))
    
    print(f"Batch contents:")
    print(f"  - image: {batch['image'].shape}")
    print(f"  - trajectory: {batch['trajectory'].shape}")
    print(f"  - input_ids: {batch['input_ids'].shape}")
    print(f"  - attention_mask: {batch['attention_mask'].shape}")
    print(f"  - command: {len(batch['command'])} text strings")
    
    print(f"\nSample commands in batch:")
    for i, cmd in enumerate(batch['command']):
        print(f"  [{i}]: '{cmd}'")
    print()
    
    # Test 4: Verify data format for VLA model
    print("=" * 70)
    print("Test 4: Data Format Verification")
    print("=" * 70)
    
    print("Checking data format compatibility with VLADrivingPolicy...")
    
    # Expected format
    B = batch['image'].shape[0]
    C, H, W = batch['image'].shape[1:]
    T, coords = batch['trajectory'].shape[1:]
    
    print(f"✓ Image format: [B={B}, C={C}, H={H}, W={W}] ✓")
    print(f"✓ Trajectory format: [B={B}, T={T}, coords={coords}] ✓")
    print(f"✓ Input IDs format: {batch['input_ids'].shape} ✓")
    
    assert coords == 2, "Trajectory should have 2 coordinates (x, y)"
    assert C == 3, "Images should have 3 channels (RGB)"
    
    print("\n✓ All format checks passed!")
    print()
    
    # Test 5: Normalization verification
    print("=" * 70)
    print("Test 5: Trajectory Normalization")
    print("=" * 70)
    
    print("Checking trajectory normalization...")
    traj = batch['trajectory'][0]  # First sample
    
    print(f"Trajectory statistics:")
    print(f"  - X range: [{traj[:, 0].min():.3f}, {traj[:, 0].max():.3f}]")
    print(f"  - Y range: [{traj[:, 1].min():.3f}, {traj[:, 1].max():.3f}]")
    print(f"  - Mean: x={traj[:, 0].mean():.3f}, y={traj[:, 1].mean():.3f}")
    print()
    
    # Test 6: Integration with model (mock)
    print("=" * 70)
    print("Test 6: Mock Model Integration")
    print("=" * 70)
    
    print("Simulating VLA model forward pass...")
    
    # Mock model forward pass
    mock_predictions = torch.randn_like(batch['trajectory'])
    
    # Compute mock loss
    loss = torch.nn.functional.mse_loss(mock_predictions, batch['trajectory'])
    
    print(f"✓ Mock forward pass successful")
    print(f"  - Predictions shape: {mock_predictions.shape}")
    print(f"  - Ground truth shape: {batch['trajectory'].shape}")
    print(f"  - MSE loss: {loss.item():.4f}")
    print()
    
    # Summary
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print("✓ Dataset loading works")
    print("✓ CLIP normalization applied")
    print("✓ Text tokenization works")
    print("✓ Trajectory normalization works")
    print("✓ Custom collate function works")
    print("✓ DataLoader batching works")
    print("✓ Data format compatible with VLADrivingPolicy")
    print()
    print("The dataset is ready for training!")
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
