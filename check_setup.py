#!/usr/bin/env python3
"""
Quick verification script to check if OpenVLA-Drive is set up correctly.
"""

import sys
import importlib


def check_import(module_name: str, package_name: str = None) -> bool:
    """Check if a module can be imported."""
    try:
        importlib.import_module(module_name)
        print(f"✓ {package_name or module_name}")
        return True
    except ImportError as e:
        print(f"✗ {package_name or module_name} - {e}")
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
    ]
    
    for module, name in optional_deps:
        check_import(module, name)
    
    print()
    
    # Check project modules
    print("Project Modules:")
    print("-" * 60)
    project_modules = [
        "models.vla_model",
        "data.carla_dataset",
        "training.lightning_module",
        "training.policy_lightning_module",
        "training.rl_env",
        "evaluation.metrics",
    ]
    
    project_ok = True
    for module in project_modules:
        if not check_import(module):
            project_ok = False
    
    print()
    print("=" * 60)
    
    if core_ok and project_ok:
        print("✓ Environment is ready!")
        print()
        print("Next steps:")
        print("1. Install CARLA 0.9.15 from https://github.com/carla-simulator/carla/releases")
        print("2. Collect data: python scripts/collect_carla_data.py")
        print("3. Train policy: python scripts/train.py --module policy")
        print("4. (Optional) RL fine-tune: python scripts/rl_finetune.py")
        print("5. Evaluate: python evaluation/closed_loop_sim.py")
    else:
        print("✗ Some dependencies are missing")
        print("Run: pip install -r requirements.txt")
        return 1
    
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
