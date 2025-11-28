"""
OpenVLA-Drive: Vision-Language-Action Models for Autonomous Driving

A research project exploring VLA models in CARLA simulator.
"""

__version__ = "0.1.0"
__author__ = "欧林海"
__email__ = "franka907@126.com"

from models.vla_model import VLAModel
from data.carla_dataset import CARLADataset

__all__ = ['VLAModel', 'CARLADataset']
