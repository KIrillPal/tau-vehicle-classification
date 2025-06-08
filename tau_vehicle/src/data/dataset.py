from pathlib import Path
from typing import Callable, Dict, Optional

from omegaconf import DictConfig
from PIL import Image
from torch.utils.data import Dataset


class VehicleDataset(Dataset):
    """Vehicle dataset that works with pre-split train/val/test directories."""

    def __init__(
        self,
        data_config: DictConfig,
        split: str = "train",
        transform: Optional[Callable] = None,
    ):
        self.root_dir = Path(data_config.dir) / split
        self.transform = transform

        if "classes" in data_config:
            self.classes = data_config.classes
        else:
            self.classes = sorted(
                [d.name for d in self.root_dir.iterdir() if d.is_dir()]
            )

        self.samples = []

        if not data_config.split[split]:
            return  # split is disabled

        for cls_idx, cls_name in enumerate(self.classes):
            cls_dir = self.root_dir / cls_name
            self.samples.extend(
                [
                    (img_path.absolute(), cls_idx)
                    for img_path in cls_dir.iterdir()
                    if img_path.suffix.lower() in (".png", ".jpg", ".jpeg")
                ]
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        return self.transform(img) if self.transform else img, label

    @property
    def class_distribution(self) -> Dict[str, int]:
        """Return count of samples per class."""
        dist = {}
        for cls in self.classes:
            cls_dir = self.root_dir / cls
            if cls_dir.exists():
                dist[cls] = len(list(cls_dir.iterdir()))
            else:
                dist[cls] = 0
        return dist
