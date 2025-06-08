from typing import Any, Dict

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from torch.nn import functional as F
from torchmetrics import Accuracy
from transformers import AutoConfig, AutoModelForImageClassification


class HuggingFaceModel(pl.LightningModule):
    """PyTorch Lightning module for HuggingFace image classification models."""

    def __init__(self, model_config: DictConfig, hyp_config: DictConfig):
        """
        Args:
            model_config : DictConfig - model initialization config
        """
        super().__init__()
        self.save_hyperparameters()

        # Initialize model
        self.num_classes = model_config.num_classes
        self.model = self._init_model(model_config)

        # Metrics
        self.train_acc = Accuracy(task="multiclass", num_classes=self.num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=self.num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=self.num_classes)

        self.hyp_config = hyp_config

    def _init_model(self, model_config: DictConfig):
        """Initialize HuggingFace model with config."""
        auto_config = AutoConfig.from_pretrained(
            model_config.source,
            num_labels=self.num_classes,
            ignore_mismatched_sizes=True,
        )

        if model_config.pretrained:
            model = AutoModelForImageClassification.from_pretrained(
                model_config.source, config=auto_config, ignore_mismatched_sizes=True
            )
        else:
            model = AutoModelForImageClassification.from_config(auto_config)

        return model

    def forward(self, x):
        return self.model(x).logits

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)

        self.train_acc(logits, y)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", self.train_acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)

        self.val_acc(logits, y)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_acc, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)

        self.test_acc(logits, y)
        self.log("test_loss", loss)
        self.log("test_acc", self.test_acc)
        return loss

    def _load_element_class(
        self, element_config: DictConfig, possible_classes: Dict[str, Any]
    ):
        if element_config.name not in possible_classes:
            raise ValueError(
                f"Unknown element '{element_config.name}'. Parsing is not implemented"
            )
        element_class = possible_classes[element_config.name]
        element_params = dict(element_config)
        element_params.pop("name")
        return element_class, element_params

    def configure_optimizers(self):
        optimizers = {
            "AdamW": torch.optim.AdamW,
            "Adam": torch.optim.Adam,
            "SGD": torch.optim.SGD,
        }

        optim_class, optim_params = self._load_element_class(
            self.hyp_config.optimizer, optimizers
        )
        optimizer = optim_class(self.parameters(), **optim_params)

        schedulers = {
            "CosineAnnealingLR": torch.optim.lr_scheduler.CosineAnnealingLR,
            "StepLR": torch.optim.lr_scheduler.StepLR,
            "ExponentialLR": torch.optim.lr_scheduler.ExponentialLR,
            "OneCycleLR": torch.optim.lr_scheduler.OneCycleLR,
            "MultiStepLR": torch.optim.lr_scheduler.MultiStepLR,
        }
        scheduler_class, scheduler_params = self._load_element_class(
            self.hyp_config.scheduler, schedulers
        )
        scheduler = scheduler_class(
            optimizer, T_max=self.hyp_config.epochs, **scheduler_params
        )

        return [optimizer], [scheduler]
