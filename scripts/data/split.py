from pathlib import Path
from sklearn.model_selection import train_test_split
import fire
import shutil


def split_dataset(
    input_dir: Path | str,
    output_dir: Path | str,
    val_ratio: float = 0.10,
    test_ratio: float = 0.15,
    random_state: int = 42
):
    """Split dataset into train/val/test with directory structure.
    
    Args:
        input_dir: Path to input dataset directory
        output_dir: Path where to create split dataset
        val_ratio: Proportion for validation set (default: 0.15)
        test_ratio: Proportion for test set (default: 0.15)
        random_state: Random seed for reproducibility (default: 42)
    """
    assert val_ratio + test_ratio < 1, "Tried to grab more than 100% data for test and val."
    
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    # Create output directories
    splits = {
        'train': output_dir / 'train',
        'val': output_dir / 'val', 
        'test': output_dir / 'test'
    }
    for split_dir in splits.values():
        split_dir.mkdir(parents=True, exist_ok=True)

    # Process each class
    for cls_dir in [d for d in input_dir.iterdir() if d.is_dir()]:
        cls = cls_dir.name
        images = [f for f in cls_dir.iterdir() if f.suffix.lower() in ('.png', '.jpg', '.jpeg')]
        
        # Split images
        train, temp = train_test_split(
            images, 
            test_size=test_ratio + val_ratio, 
            random_state=random_state
        )
        relative_ratio = test_ratio/(val_ratio+test_ratio)
        val, test = train_test_split(
            temp, 
            test_size=relative_ratio, 
            random_state=random_state
        )
        
        # Copy files to respective split directories
        for split, imgs in zip(splits.keys(), [train, val, test]):
            dest_dir = splits[split] / cls
            dest_dir.mkdir(exist_ok=True)
            for img in imgs:
                shutil.copy2(img, dest_dir / img.name)
        
        print(f"{cls}: {len(train)} train, {len(val)} val, {len(test)} test")


if __name__ == "__main__":
    fire.Fire(split_dataset)
