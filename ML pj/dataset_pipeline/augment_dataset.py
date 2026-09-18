import os
import cv2
import albumentations as A
import numpy as np
from pathlib import Path

# Designed to mimic field conditions: bad lighting, motion blur from cheap phone cameras, Jpeg artifacts
field_augmenter = A.Compose([
    # Simulate variable farm lighting and strong sun shadows
    A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.3, p=0.7),
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=30, val_shift_limit=20, p=0.5),
    A.RandomGamma(gamma_limit=(80, 120), p=0.4),
    
    # Simulate cheap phone camera artifacts
    A.ImageCompression(quality_lower=30, quality_upper=80, p=0.5), # Standard JPEG artifacting
    A.MotionBlur(blur_limit=7, p=0.4), # Hand shake
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.4), # Low-light sensor noise
    A.Defocus(radius=(2, 6), alias_blur=(0.1, 0.5), p=0.3), # Bad autofocus

    # Geometry (Leaves are rarely perfectly centered/flat)
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=45, p=0.8),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
])

def run_augmentation(input_dir: str, output_dir: str, target_count_per_class: int = 450):
    """
    Augments under-represented classes to reach `target_count_per_class`.
    Simulates real field conditions out of clean lab images.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    if not input_path.exists():
        print(f"Input directory {input_dir} not found.")
        return
        
    for cls_dir in [d for d in input_path.iterdir() if d.is_dir()]:
        cls_name = cls_dir.name
        out_cls_dir = output_path / cls_name
        out_cls_dir.mkdir(parents=True, exist_ok=True)
        
        images = list(cls_dir.glob("*.jpg")) + list(cls_dir.glob("*.png")) + list(cls_dir.glob("*.jpeg"))
        if not images:
            continue
            
        print(f"[{cls_name}] Starting augmentation. Base images: {len(images)} -> Target: {target_count_per_class}")
        
        # 1. Copy all original images to output directly
        for img_path in images:
            img = cv2.imread(str(img_path))
            if img is not None:
                cv2.imwrite(str(out_cls_dir / img_path.name), img)

        current_count = len(images)
        augment_count = target_count_per_class - current_count
        
        if augment_count <= 0:
            print(f"  - No augmentation needed, already has {current_count} images.")
            continue
            
        # 2. Generate augmented images
        gen_count = 0
        while gen_count < augment_count:
            # Pick a random base image
            img_path = str(np.random.choice(images))
            img = cv2.imread(img_path)
            if img is None:
                continue
                
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            augmented = field_augmenter(image=img)["image"]
            augmented = cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR)
            
            aug_filename = f"aug_{gen_count}_{Path(img_path).name}"
            cv2.imwrite(str(out_cls_dir / aug_filename), augmented)
            gen_count += 1
            
        print(f"  - Added {gen_count} augmented field-condition images.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Augment dataset")
    parser.add_argument("--input", type=str, default="dataset_raw", help="Path to raw dataset")
    parser.add_argument("--output", type=str, default="dataset_augmented", help="Path for output augmented dataset")
    parser.add_argument("--target", type=int, default=450, help="Target count per class")
    args = parser.parse_args()
    
    run_augmentation(args.input, args.output, args.target)
