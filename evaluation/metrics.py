"""
Evaluation Metrics for Autonomous Driving

This module provides metrics for evaluating driving performance.
"""

from typing import Dict, List

import numpy as np
import torch


def compute_driving_metrics(
    predictions: Dict[str, torch.Tensor],
    ground_truth: torch.Tensor,
) -> Dict[str, float]:
    """
    Compute driving-specific metrics.
    
    Args:
        predictions: Dict with 'steering', 'throttle', 'brake' predictions
        ground_truth: [B, 3] ground truth actions
    
    Returns:
        Dict of metric names to values
    """
    # Convert to numpy
    pred_steering = predictions['steering'].detach().cpu().numpy().flatten()
    pred_throttle = predictions['throttle'].detach().cpu().numpy().flatten()
    pred_brake = predictions['brake'].detach().cpu().numpy().flatten()
    
    gt_steering = ground_truth[:, 0].detach().cpu().numpy()
    gt_throttle = ground_truth[:, 1].detach().cpu().numpy()
    gt_brake = ground_truth[:, 2].detach().cpu().numpy()
    
    # Compute MAE and MSE for each action
    metrics = {
        'steering_mae': np.mean(np.abs(pred_steering - gt_steering)),
        'steering_mse': np.mean((pred_steering - gt_steering) ** 2),
        'throttle_mae': np.mean(np.abs(pred_throttle - gt_throttle)),
        'throttle_mse': np.mean((pred_throttle - gt_throttle) ** 2),
        'brake_mae': np.mean(np.abs(pred_brake - gt_brake)),
        'brake_mse': np.mean((pred_brake - gt_brake) ** 2),
    }
    
    return metrics


def compute_route_completion(
    waypoints_reached: int,
    total_waypoints: int,
) -> float:
    """
    Compute route completion percentage.
    
    Args:
        waypoints_reached: Number of waypoints successfully reached
        total_waypoints: Total number of waypoints in route
    
    Returns:
        Completion percentage [0, 100]
    """
    return (waypoints_reached / total_waypoints) * 100.0


def compute_infraction_score(
    collisions: int,
    red_light_violations: int,
    off_road_frames: int,
    total_frames: int,
) -> Dict[str, float]:
    """
    Compute infraction-based metrics.
    
    Args:
        collisions: Number of collision events
        red_light_violations: Number of red light violations
        off_road_frames: Number of frames where vehicle was off-road
        total_frames: Total number of simulation frames
    
    Returns:
        Dict of infraction metrics
    """
    return {
        'collision_count': collisions,
        'red_light_violations': red_light_violations,
        'off_road_percentage': (off_road_frames / total_frames) * 100.0,
        'infraction_score': collisions + red_light_violations + (off_road_frames / total_frames),
    }
