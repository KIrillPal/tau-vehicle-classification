from pathlib import Path
from typing import Optional, Union

import cv2
import fire
import numpy as np
import onnxruntime as ort
import torch
import yaml
from omegaconf import DictConfig, OmegaConf
from PIL import Image

from tau_vehicle.src.data.transforms import create_infer_transforms


class ModelClassifier:
    def __init__(self, config: DictConfig):
        """
        Initialize classifier based on configuration

        Args:
            config: OmegaConf configuration dictionary
        """
        self.config = config
        self.model_path = Path(config.path)
        self.img_size = config.imgsz
        self.classes = list(config.classes)

        # Create image transforms
        self.transform = create_infer_transforms(self.img_size, config.transforms)

        # Load model based on format
        if self.model_path.suffix == ".pt":
            self._load_torch_model()
        elif self.model_path.suffix == ".onnx":
            self._load_onnx_model()
        else:
            raise ValueError(f"Unsupported model format: {self.model_path.suffix}")

    def _load_torch_model(self):
        """Load PyTorch model"""
        # Load model file
        model_data = torch.load(self.model_path, weights_only=False, map_location="cpu")

        if isinstance(model_data, torch.jit.ScriptModule):
            self.model = model_data
        elif isinstance(model_data, dict) and "state_dict" in model_data:
            raise ValueError(
                "Checkpoint contains state_dict only. "
                "Please export to TorchScript format first."
            )
        elif isinstance(model_data, torch.nn.Module):
            # Full PyTorch model
            self.model = model_data
        else:
            raise ValueError("Unsupported PyTorch checkpoint format")

        self.model.eval()
        print(f"Loaded PyTorch model: {self.model_path}")

    def _load_onnx_model(self):
        """Load ONNX model"""
        # Create ONNX Runtime session
        self.sess = ort.InferenceSession(str(self.model_path))
        self.input_name = self.sess.get_inputs()[0].name
        print(f"Loaded ONNX model: {self.model_path}")

    def preprocess_image(self, image_path: str) -> Union[torch.Tensor, np.ndarray]:
        """
        Preprocess image for inference

        Args:
            image_path: Path to image file

        Returns:
            Preprocessed tensor (for PyTorch) or array (for ONNX)
        """
        # Read image
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")

        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # PyTorch model - use torchvision transforms
        if self.model_path.suffix == ".pt":
            img_pil = Image.fromarray(img)
            return self.transform(img_pil).unsqueeze(0)  # Add batch dimension

        # ONNX model - manual transformations
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = img.astype(np.float32) / 255.0

        # Normalization
        mean = np.array(self.config.transforms.normalize.mean, dtype=np.float32)
        std = np.array(self.config.transforms.normalize.std, dtype=np.float32)
        img = (img - mean) / std

        # Change dimension order: HWC -> CHW
        img = np.transpose(img, (2, 0, 1))
        return np.expand_dims(img, axis=0)  # Add batch dimension

    def predict(self, image_path: str) -> str:
        """
        Predict class for a single image

        Args:
            image_path: Path to image file

        Returns:
            Predicted class name
        """
        try:
            # Preprocess image
            input_data = self.preprocess_image(image_path)

            # Run inference
            if self.model_path.suffix == ".pt":
                with torch.no_grad():
                    outputs = self.model(input_data)

                    # Handle different output types
                    if hasattr(
                        outputs, "logits"
                    ):  # For models like timm that return objects
                        outputs = outputs.logits
                    elif hasattr(outputs, "prediction"):  # Other custom output formats
                        outputs = outputs.prediction

                    pred = torch.argmax(outputs, dim=1).item()
            else:  # ONNX
                outputs = self.sess.run(None, {self.input_name: input_data})
                pred = np.argmax(outputs[0])

            # Return class name
            return self.classes[pred]

        except Exception as e:
            return f"Error processing {image_path}: {str(e)}"

    def process_directory(self, image_dir: str, output_yaml: str = "predictions.yaml"):
        """
        Process all images in a directory and save results

        Args:
            image_dir: Path to directory with images
            output_yaml: Output YAML file path
        """
        results = {}
        image_dir = Path(image_dir)

        # Supported image extensions
        img_exts = [".jpg", ".jpeg", ".png", ".bmp"]

        # Process all images in directory
        image_paths = sorted(list(image_dir.iterdir()))
        total = len(image_paths)
        for i, img_path in enumerate(image_paths):
            if img_path.suffix.lower() in img_exts:
                class_name = self.predict(str(img_path))
                results[img_path.name] = class_name
                print(f"[{i:4}/{total:4}] {img_path.name}: {class_name}")

        # Save results to YAML
        with open(output_yaml, "w", encoding="utf-8") as f:
            yaml.dump(results, f, allow_unicode=True)

        print(f"\nResults saved to {output_yaml}")

    def interactive_mode(self):
        """Interactive mode for image classification"""
        print("\nInteractive mode - enter image paths")
        print("Type 'q' or 'quit' to exit")

        while True:
            img_path = input("\nEnter image path: ").strip()

            # Check for exit command
            if img_path.lower() in ["q", "quit", "exit"]:
                print("Exiting interactive mode")
                break

            # Check file exists
            if not Path(img_path).exists():
                print("File not found, please try again")
                continue

            # Run prediction
            class_name = self.predict(img_path)
            print(f"Predicted class: {class_name}")


def main(
    config_path: str,
    image_dir: Optional[str] = None,
    output_path: str = "predictions.yaml",
):
    """
    Inference.

    Args:
        config_path: Path to YAML configuration file for inference
        image_dir: Directory with images to classify (optional)
        output_path: Output YAML file path (default: predictions.yaml)
    Modes:
        1) If image_dir is provided, saves predictions for all images in image_dir to output_path.
        2) Otherwise start interactive mode. One prediction for each input.
    """
    # Load configuration with OmegaConf
    config = OmegaConf.load(config_path)

    # Initialize classifier
    classifier = ModelClassifier(config)

    # Process based on mode
    if image_dir:
        classifier.process_directory(image_dir, output_path)
    else:
        classifier.interactive_mode()


if __name__ == "__main__":
    fire.Fire(main)
