#!/usr/bin/env python3
"""
Quick verification script to check if OpenVLA-Drive is set up correctly.
"""

import sys
import importlib
from pathlib import Path


def check_import(module_name: str, package_name: str = None) -> bool:
    """Check if a module can be imported."""
    try:
        importlib.import_module(module_name)
        print(f"✓ {package_name or module_name}")
        return True
    except ImportError as e:
        print(f"✗ {package_name or module_name} - {e}")
        return False


def check_class_in_module(module_name: str, class_name: str, display_name: str = None) -> bool:
    """Check if a class can be imported from a module."""
    try:
        module = importlib.import_module(module_name)
        if hasattr(module, class_name):
            print(f"✓ {display_name or f'{module_name}.{class_name}'}")
            return True
        else:
            print(f"✗ {display_name or f'{module_name}.{class_name}'} - Class not found")
            return False
    except ImportError as e:
        print(f"✗ {display_name or f'{module_name}.{class_name}'} - {e}")
        return False


def check_file_exists(file_path: str, display_name: str = None) -> bool:
    """Check if a file exists."""
    path = Path(file_path)
    if path.exists():
        print(f"✓ {display_name or file_path}")
        return True
    else:
        print(f"✗ {display_name or file_path} - File not found")
        return False


def main():
    print("=" * 60)
    print("OpenVLA-Drive Environment Check")
    print("=" * 60)
    print()
    
    # Core dependencies
    print("Core Dependencies:")
    print("-" * 60)
    deps = [
        ("torch", "PyTorch"),
        ("torchvision", "TorchVision"),
        ("transformers", "HuggingFace Transformers"),
        ("numpy", "NumPy"),
        ("PIL", "Pillow"),
    ]
    
    core_ok = True
    for module, name in deps:
        if not check_import(module, name):
            core_ok = False
    
    print()
    
    # Optional dependencies
    print("Optional Dependencies:")
    print("-" * 60)
    optional_deps = [
        ("pytorch_lightning", "PyTorch Lightning"),
        ("cv2", "OpenCV"),
        ("h5py", "H5PY"),
        ("carla", "CARLA Python API"),
        ("omegaconf", "OmegaConf"),
        ("wandb", "Weights & Biases"),
        ("peft", "PEFT (LoRA)"),
    ]
    
    for module, name in optional_deps:
        check_import(module, name)
    
    print()
    
    # Check project modules
    print("Project Modules:")
    print("-" * 60)
    project_modules = [
        ("models.vla_model", "VLA Model"),
        ("data.carla_dataset", "CARLA Dataset"),
        ("training.lightning_module", "VLA Lightning Module"),
        ("training.policy_lightning_module", "Policy Lightning Module"),
        ("training.rl_env", "RL Environment"),
        ("evaluation.metrics", "Evaluation Metrics"),
    ]
    
    project_ok = True
    for module, name in project_modules:
        if not check_import(module, name):
            project_ok = False
    
    print()
    
    # Check policy-specific components
    print("Policy & RL Components:")
    print("-" * 60)
    policy_ok = True
    
    # Check VLADrivingPolicy class
    if not check_class_in_module("models.policy", "VLADrivingPolicy", "VLADrivingPolicy"):
        policy_ok = False
    
    # Check ActionHead class
    if not check_class_in_module("models.policy", "ActionHead", "ActionHead"):
        policy_ok = False
    
    # Check RL environment
    if not check_class_in_module("training.rl_env", "OpenVLAControlEnv", "OpenVLAControlEnv"):
        policy_ok = False
    
    # Check RL fine-tune script
    if not check_import("scripts.rl_finetune", "RL Fine-tune Script"):
        policy_ok = False
    
    print()
    
    # Check RL dependencies (optional)
    print("RL Dependencies (Optional):")
    print("-" * 60)
    rl_deps = [
        ("gymnasium", "Gymnasium"),
        ("stable_baselines3", "Stable-Baselines3"),
    ]
    
    for module, name in rl_deps:
        check_import(module, name)
    
    print()
    
    # Check configuration files
    print("Configuration Files:")
    print("-" * 60)
    config_files = [
        ("configs/training_config.yaml", "Training Config"),
        ("configs/policy_config.yaml", "Policy Config"),
        ("configs/data_config.yaml", "Data Config"),
        ("configs/model_config.yaml", "Model Config"),
    ]
    
    config_ok = True
    for file_path, name in config_files:
        if not check_file_exists(file_path, name):
            config_ok = False
    
    print()
    print("=" * 60)
    
    if core_ok and project_ok and policy_ok and config_ok:
        print("✓ Environment is ready!")
        print()
        print("Next steps:")
        print("1. Install CARLA 0.9.15 from https://github.com/carla-simulator/carla/releases")
        print("2. Collect data: python scripts/collect_carla_data.py")
        print("3. Train policy: python scripts/train.py --module policy")
        print("4. (Optional) RL fine-tune: python scripts/rl_finetune.py")
        print("5. Evaluate: python evaluation/closed_loop_sim.py")
    else:
        print("✗ Some dependencies or components are missing")
        if not core_ok or not project_ok:
            print("Run: pip install -r requirements.txt")
        if not policy_ok:
            print("Some policy/RL components failed to import. Check the error messages above.")
        if not config_ok:
            print("Some configuration files are missing. Check the configs/ directory.")
        return 1
    
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
