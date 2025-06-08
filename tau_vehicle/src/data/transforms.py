import torchvision.transforms as T
from typing import Dict
from omegaconf import DictConfig

def create_transforms(img_size: int, aug_config : DictConfig) -> Dict[str, T.Compose]:
    """Get standard transforms for train/val/test splits."""
    common = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(aug_config.normalize.mean, aug_config.normalize.std)
    ])

    train_augs = [
            T.RandomRotation(aug_config.rotation),
            T.ColorJitter(
                brightness=aug_config.brightness,
                contrast=aug_config.contrast, 
                saturation=aug_config.saturation
            ),
    ]

    if aug_config.flip.horizontal:
        train_augs.append(T.RandomHorizontalFlip())
    if aug_config.flip.vertical:
        train_augs.append(T.RandomVerticalFlip())

    train_augs.extend([*common.transforms])
    
    return {
        'train': T.Compose(train_augs),
        'val': common,
        'test': common
    }
    
    
def create_infer_transforms(img_size: int, transform_config : DictConfig) -> Dict[str, T.Compose]:
    """Get standard transforms for train/val/test splits."""
    common = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(transform_config.normalize.mean, transform_config.normalize.std)
    ])
    return common