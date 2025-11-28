"""
CARLA Dataset Loader for VLA Training

This module handles loading and preprocessing of CARLA driving data.
Optimized for VLA Driving Policy with trajectory prediction.
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import AutoTokenizer, AutoProcessor


class CARLAVLADataset(Dataset):
    """
    Dataset for CARLA VLA Driving Policy Training.
    
    Expected data format:
    - Front-view RGB images
    - Navigation commands (text)
    - Ground truth trajectory (list of x,y waypoints)
    
    Directory structure:
    data_root/
        {split}/
            images/
                000000.png
                000001.png
                ...
            annotations.json  # or separate JSON files per sample
    
    Annotation format:
    {
        "000000": {
            "image": "images/000000.png",
            "command": "Follow the lane",
            "trajectory": [[x0, y0], [x1, y1], ...],
            "ego_position": [x, y, theta]  # optional, for normalization
        },
        ...
    }
    """
    
    def __init__(
        self,
        data_root: str,
        split: str = "train",
        image_size: Tuple[int, int] = (224, 224),
        tokenizer_name: str = "microsoft/phi-2",
        max_text_length: int = 128,
        num_trajectory_points: int = 10,
        normalize_trajectory: bool = True,
        use_clip_normalization: bool = True,
        transform: Optional[transforms.Compose] = None,
    ):
        """
        Args:
            data_root: Path to the dataset root directory
            split: Dataset split ('train', 'val', 'test')
            image_size: Target image size (H, W) for CLIP encoder
            tokenizer_name: HuggingFace tokenizer name
            max_text_length: Maximum text token length
            num_trajectory_points: Number of trajectory waypoints (T)
            normalize_trajectory: Whether to normalize trajectory to ego frame
            use_clip_normalization: Use CLIP normalization for images
            transform: Optional custom image transformations
        """
        self.data_root = Path(data_root)
        self.split = split
        self.image_size = image_size
        self.max_text_length = max_text_length
        self.num_trajectory_points = num_trajectory_points
        self.normalize_trajectory = normalize_trajectory
        
        # Load tokenizer
        print(f"Loading tokenizer: {tokenizer_name}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_name,
                trust_remote_code=True
            )
            # Set pad token if not exists
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        except Exception as e:
            print(f"Warning: Could not load tokenizer {tokenizer_name}: {e}")
            self.tokenizer = None
        
        # Image transforms for CLIP encoder
        if transform is None:
            if use_clip_normalization:
                # CLIP normalization
                self.transform = transforms.Compose([
                    transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
                    transforms.CenterCrop(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.48145466, 0.4578275, 0.40821073],
                        std=[0.26862954, 0.26130258, 0.27577711]
                    ),
                ])
            else:
                # Standard ImageNet normalization
                self.transform = transforms.Compose([
                    transforms.Resize(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]
                    ),
                ])
        else:
            self.transform = transform
        
        # Load dataset annotations
        self.samples = self._load_annotations()
    
    def _load_annotations(self) -> List[Dict[str, Any]]:
        """
        Load dataset annotations.
        
        Supports multiple formats:
        1. Single annotations.json file
        2. Individual JSON files per sample
        3. CSV format
        """
        split_dir = self.data_root / self.split
        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")
        
        samples = []
        
        # Try loading from annotations.json
        annotation_file = split_dir / "annotations.json"
        if annotation_file.exists():
            print(f"Loading annotations from {annotation_file}")
            with open(annotation_file, 'r') as f:
                annotations = json.load(f)
            
            # Convert to list of samples
            for sample_id, data in annotations.items():
                sample = {
                    'id': sample_id,
                    'image_path': split_dir / data['image'],
                    'command': data['command'],
                    'trajectory': np.array(data['trajectory'], dtype=np.float32),
                }
                
                # Optional: ego position for normalization
                if 'ego_position' in data:
                    sample['ego_position'] = np.array(data['ego_position'], dtype=np.float32)
                
                samples.append(sample)
        
        else:
            # Try loading from individual JSON files
            json_files = sorted(split_dir.glob('*.json'))
            if json_files:
                print(f"Loading {len(json_files)} individual annotation files")
                for json_file in json_files:
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                    
                    sample = {
                        'id': json_file.stem,
                        'image_path': split_dir / data['image'],
                        'command': data['command'],
                        'trajectory': np.array(data['trajectory'], dtype=np.float32),
                    }
                    
                    if 'ego_position' in data:
                        sample['ego_position'] = np.array(data['ego_position'], dtype=np.float32)
                    
                    samples.append(sample)
            else:
                print(f"Warning: No annotation files found in {split_dir}")
                print("Creating dummy samples for testing...")
                # Create dummy samples for testing
                image_files = sorted(split_dir.glob('images/*.png')) + sorted(split_dir.glob('images/*.jpg'))
                for i, img_path in enumerate(image_files[:10]):
                    samples.append({
                        'id': f'{i:06d}',
                        'image_path': img_path,
                        'command': 'Follow the lane',
                        'trajectory': np.random.randn(self.num_trajectory_points, 2).astype(np.float32),
                    })
        
        print(f"Loaded {len(samples)} samples for split '{self.split}'")
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str]]:
        """
        Get a single sample.
        
        Returns:
            Dictionary containing:
                - 'image': [C, H, W] preprocessed image tensor
                - 'command': str, navigation command text
                - 'trajectory': [T, 2] normalized trajectory waypoints
                - 'input_ids': [seq_len] tokenized text (if tokenizer available)
                - 'attention_mask': [seq_len] attention mask (if tokenizer available)
        """
        sample = self.samples[idx]
        
        # Load and preprocess image
        image = Image.open(sample['image_path']).convert('RGB')
        image = self.transform(image)
        
        # Get text command
        command = sample['command']
        
        # Process trajectory
        trajectory = sample['trajectory'].copy()
        
        # Normalize trajectory if needed
        if self.normalize_trajectory:
            trajectory = self._normalize_trajectory(
                trajectory,
                ego_position=sample.get('ego_position', None)
            )
        
        # Ensure trajectory has exactly num_trajectory_points
        trajectory = self._resample_trajectory(trajectory, self.num_trajectory_points)
        
        # Convert to tensor
        trajectory = torch.from_numpy(trajectory).float()  # [T, 2]
        
        # Prepare output
        output = {
            'image': image,
            'command': command,
            'trajectory': trajectory,
        }
        
        # Tokenize text if tokenizer is available
        if self.tokenizer is not None:
            encoding = self.tokenizer(
                command,
                max_length=self.max_text_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            output['input_ids'] = encoding['input_ids'].squeeze(0)
            output['attention_mask'] = encoding['attention_mask'].squeeze(0)
        
        return output
    
    def _normalize_trajectory(
        self,
        trajectory: np.ndarray,
        ego_position: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Normalize trajectory to ego-vehicle frame.
        
        Args:
            trajectory: [N, 2] absolute world coordinates (x, y)
            ego_position: [3] ego vehicle position (x, y, theta) in world frame
        
        Returns:
            normalized_trajectory: [N, 2] relative coordinates in ego frame
        """
        if ego_position is None:
            # If no ego position, assume first point is ego position
            # and normalize relative to it
            if len(trajectory) > 0:
                offset = trajectory[0].copy()
                trajectory = trajectory - offset
            return trajectory
        
        # Extract ego position and orientation
        ego_x, ego_y = ego_position[0], ego_position[1]
        ego_theta = ego_position[2] if len(ego_position) > 2 else 0.0
        
        # Translate to ego position
        trajectory_translated = trajectory - np.array([ego_x, ego_y])
        
        # Rotate to ego frame
        cos_theta = np.cos(-ego_theta)
        sin_theta = np.sin(-ego_theta)
        rotation_matrix = np.array([
            [cos_theta, -sin_theta],
            [sin_theta, cos_theta]
        ])
        
        trajectory_normalized = trajectory_translated @ rotation_matrix.T
        
        return trajectory_normalized.astype(np.float32)
    
    def _resample_trajectory(
        self,
        trajectory: np.ndarray,
        num_points: int
    ) -> np.ndarray:
        """
        Resample trajectory to have exactly num_points waypoints.
        
        Args:
            trajectory: [N, 2] original trajectory
            num_points: Target number of points (T)
        
        Returns:
            resampled: [num_points, 2] resampled trajectory
        """
        if len(trajectory) == num_points:
            return trajectory
        
        if len(trajectory) < num_points:
            # Upsample: linear interpolation
            indices = np.linspace(0, len(trajectory) - 1, num_points)
            resampled = np.zeros((num_points, 2), dtype=np.float32)
            
            for i, idx in enumerate(indices):
                idx_floor = int(np.floor(idx))
                idx_ceil = min(int(np.ceil(idx)), len(trajectory) - 1)
                
                if idx_floor == idx_ceil:
                    resampled[i] = trajectory[idx_floor]
                else:
                    alpha = idx - idx_floor
                    resampled[i] = (1 - alpha) * trajectory[idx_floor] + alpha * trajectory[idx_ceil]
        else:
            # Downsample: uniformly sample
            indices = np.linspace(0, len(trajectory) - 1, num_points, dtype=int)
            resampled = trajectory[indices]
        
        return resampled


def carla_vla_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for CARLA VLA dataset.
    
    Handles variable-length text by padding to the longest sequence in the batch.
    
    Args:
        batch: List of sample dictionaries from __getitem__
    
    Returns:
        Batched dictionary with:
            - 'image': [B, C, H, W]
            - 'trajectory': [B, T, 2]
            - 'input_ids': [B, max_seq_len]
            - 'attention_mask': [B, max_seq_len]
            - 'command': List of strings (length B)
    """
    # Stack images and trajectories
    images = torch.stack([item['image'] for item in batch])
    trajectories = torch.stack([item['trajectory'] for item in batch])
    commands = [item['command'] for item in batch]
    
    # Handle tokenized text if available
    if 'input_ids' in batch[0]:
        # All items should have the same length if padded in __getitem__
        input_ids = torch.stack([item['input_ids'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        
        return {
            'image': images,
            'trajectory': trajectories,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'command': commands,
        }
    else:
        # No tokenization in dataset, return raw commands
        return {
            'image': images,
            'trajectory': trajectories,
            'command': commands,
        }


def get_carla_vla_dataloader(
    data_root: str,
    split: str,
    batch_size: int,
    tokenizer_name: str = "microsoft/phi-2",
    num_trajectory_points: int = 10,
    image_size: Tuple[int, int] = (224, 224),
    max_text_length: int = 128,
    num_workers: int = 4,
    shuffle: bool = True,
    use_clip_normalization: bool = True,
    **kwargs
) -> DataLoader:
    """
    Create DataLoader for CARLA VLA dataset.
    
    Args:
        data_root: Path to dataset root
        split: 'train', 'val', or 'test'
        batch_size: Batch size
        tokenizer_name: HuggingFace tokenizer name
        num_trajectory_points: Number of waypoints to predict (T)
        image_size: Target image size for CLIP
        max_text_length: Maximum text sequence length
        num_workers: Number of data loading workers
        shuffle: Whether to shuffle data
        use_clip_normalization: Use CLIP image normalization
        **kwargs: Additional arguments for CARLAVLADataset
    
    Returns:
        DataLoader instance
    """
    dataset = CARLAVLADataset(
        data_root=data_root,
        split=split,
        image_size=image_size,
        tokenizer_name=tokenizer_name,
        max_text_length=max_text_length,
        num_trajectory_points=num_trajectory_points,
        use_clip_normalization=use_clip_normalization,
        **kwargs
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=carla_vla_collate_fn,
        pin_memory=True,
        drop_last=True if split == 'train' else False,
    )
    
    return dataloader


# Backward compatibility: keep old name
CARLADataset = CARLAVLADataset
