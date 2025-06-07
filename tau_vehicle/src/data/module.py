from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from typing import Optional
from pathlib import Path
from omegaconf import DictConfig

from tau_vehicle.src.data.dataset import VehicleDataset
from tau_vehicle.src.data.transforms import create_transforms


class VehicleDataModule(LightningDataModule):
    def __init__(self, config : DictConfig):
        super().__init__()
        self.data_dir = config.data.dir
        self.batch_size = config.hyp.batch
        self.transforms = create_transforms(config.hyp.imgsz, config.hyp.augment)
        self.config = config
        
    def setup(self, stage: Optional[str] = None):
        if stage in ('fit', None):
            self.train_ds = VehicleDataset(self.config.data, 'train', self.transforms['train'])
            self.val_ds = VehicleDataset(self.config.data, 'val', self.transforms['val'])
            self.num_classes = len(self.train_ds.classes)
        
        if stage in ('test', None):
            self.test_ds = VehicleDataset(self.config.data, 'test', self.transforms['test'])
    
    def _create_dataloader(self, dataset, shuffle=False):
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.config.data.loader.num_workers,
            pin_memory=self.config.data.loader.pin_memory,
            persistent_workers=True
        )
    
    def train_dataloader(self):
        return self._create_dataloader(self.train_ds, shuffle=self.config.data.loader.shuffle)
    
    def val_dataloader(self):
        return self._create_dataloader(self.val_ds, shuffle=False)
    
    def test_dataloader(self):
        return self._create_dataloader(self.test_ds, shuffle=False)