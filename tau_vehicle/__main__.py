"""
Main entry point for the tau_vehicle package.

This module provides command-line interface for training and inference
of vehicle type recognition models.

Usage:
    python -m tau_vehicle <command> [options]

Commands:
    train       Train a vehicle classification model
    infer       Run inference on images
    export      Export checkpoint to different formats (pt, ONNX, TensorRT)

Examples:
    python3 -m tau_vehicle train
    python3 -m tau_vehicle infer conf/infer/baseline.yaml
    python3 -m tau_vehicle export to_engine checkpoints/best.ckpt
"""

import sys

import fire
from hydra import compose, initialize

from tau_vehicle.src.export import ModelExporter
from tau_vehicle.src.infer import main as infer_py
from tau_vehicle.src.train import train as train_py


def load_hydra_config(config_dir: str, config_name: str):
    with initialize(version_base=None, config_path=config_dir):
        config = compose(config_name)
    return config


def train(config_name: str = "baseline", config_dir: str = "../conf"):
    """
    Train a vehicle classification model.

    Args:
        config_name: Name of the configuration file
        config_dir: Directory containing configuration files
    """
    config = load_hydra_config(config_dir, config_name)
    return train_py(config)


def infer(*args, **kwargs):
    """
    Run inference on images.

    Args:
        config_path: Path to YAML configuration file for inference
        image_dir: Directory with images to classify (optional)
        output_path: Output YAML file path (default: predictions.yaml)
    Modes:
        1) If image_dir is provided, saves predictions for all images in image_dir to output_path.
        2) Otherwise start interactive mode. One prediction for each input.
    """
    return infer_py(*args, **kwargs)


def main():
    """Main entry point using Fire for CLI."""

    # Create Fire CLI with available commands
    commands = {
        "train": train,
        "infer": infer,
        "export": ModelExporter,
    }

    try:
        fire.Fire(commands)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
