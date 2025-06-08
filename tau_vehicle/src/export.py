from pathlib import Path

import fire
import onnx
import tensorrt as trt
import torch
from hydra import compose, initialize

from tau_vehicle.src.model.huggingface import HuggingFaceModel


def load_hydra_config(config_dir: str, config_name: str):
    with initialize(version_base=None, config_path=config_dir):
        config = compose(config_name)
    return config


class ModelExporter:
    """
    Export PyTorch .ckpt models to various formats.
    Supported formats: .pt, .onnx, .engine (TensorRT)
    """

    def __init__(self):
        self.logger = trt.Logger(trt.Logger.WARNING)

    def to_pt(
        self,
        ckpt_path: str,
        output_path: str = None,
        config_name: str = "baseline",
        config_dir: str = "../../conf",
    ):
        """
        Export .ckpt to .pt format
        Args:
            ckpt_path: Path to input .ckpt file
            output_path: Optional output path (defaults to same directory as input)
        """
        ckpt_path = Path(ckpt_path)
        if not output_path:
            output_path = ckpt_path.with_suffix(".pt")
        checkpoint = torch.load(ckpt_path, weights_only=False)

        config = load_hydra_config(config_dir, config_name)

        model = HuggingFaceModel(config.model, config.hyp)
        model.load_state_dict(checkpoint["state_dict"])

        torch.save(model.model, output_path)

        print(f"Successfully exported to {output_path}")

    def to_onnx(
        self,
        ckpt_path: str,
        output_path: str = None,
        config_name: str = "baseline",
        config_dir: str = "../../conf",
        input_names: list = ["input"],
        output_names: list = ["output"],
        dynamic_axes: dict = None,
        opset_version: int = 11,
    ):
        """
        Export .ckpt to ONNX format
        Args:
            ckpt_path: Path to input .ckpt file
            output_path: Optional output path
            input_names: List of input names
            output_names: List of output names
            dynamic_axes: Dict specifying dynamic axes
            opset_version: ONNX opset version
        """

        # First export to .pt
        pt_path = Path(ckpt_path).with_suffix(".pt")
        if not pt_path.exists():
            self.to_pt(ckpt_path, pt_path, config_name, config_dir)

        ckpt_path = Path(ckpt_path)
        if not output_path:
            output_path = ckpt_path.with_suffix(".onnx")

        model = torch.load(pt_path, weights_only=False)
        if not isinstance(model, torch.nn.Module):
            raise ValueError(
                "Model in .ckpt file must be a PyTorch Module for ONNX export"
            )
        model.eval()

        config = load_hydra_config(config_dir, config_name)
        input_shape = (1, 3, config.hyp.imgsz, config.hyp.imgsz)

        dummy_input = torch.randn(*input_shape)

        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opset_version,
        )

        # Verify the ONNX model
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)

        print(f"Successfully exported to {output_path}")

    def to_engine(
        self,
        ckpt_path: str,
        output_path: str = None,
        config_name: str = "baseline",
        config_dir: str = "../../conf",
        fp16_mode: bool = False,
    ):
        """
        Export .ckpt to TensorRT engine format
        Args:
            ckpt_path: Path to input .ckpt file
            output_path: Optional output path
            input_shape: Model input shape
            fp16_mode: Enable FP16 precision
            max_workspace_size: Maximum workspace size in bytes
        """
        # First export to ONNX
        onnx_path = Path(ckpt_path).with_suffix(".onnx")
        if not onnx_path.exists():
            self.to_onnx(ckpt_path, onnx_path, config_name, config_dir)

        if not output_path:
            output_path = Path(ckpt_path).with_suffix(".engine")

        # Build TensorRT engine
        builder = trt.Builder(self.logger)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        parser = trt.OnnxParser(network, self.logger)

        with open(onnx_path, "rb") as model:
            if not parser.parse(model.read()):
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                raise ValueError("ONNX parser failed")

        config = builder.create_builder_config()
        if fp16_mode and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)

        engine = builder.build_serialized_network(network, config)
        with open(output_path, "wb") as f:
            f.write(engine)

        print(f"Successfully exported to {output_path}")


if __name__ == "__main__":
    fire.Fire(ModelExporter)
