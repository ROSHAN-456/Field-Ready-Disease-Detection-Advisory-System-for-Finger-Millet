import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split

def create_stratified_split(dataset_dir: str, out_base_dir: str, train_pct=0.7, val_pct=0.15, test_pct=0.15):
    """
    Splits the dataset into train/val/test maintaining exact class distributions.
    """
    assert abs((train_pct + val_pct + test_pct) - 1.0) < 1e-5, "Percentages must sum to 1.0"
    
    in_dir = Path(dataset_dir)
    out_dir = Path(out_base_dir)
    
    print(f"Splitting dataset: {dataset_dir} -> {out_base_dir}")
    print(f"Ratios: Train ({train_pct}), Val ({val_pct}), Test ({test_pct})")
    
    # Create subset directories
    for subset in ['train', 'val', 'test']:
        (out_dir / subset).mkdir(parents=True, exist_ok=True)
    
    classes = [d.name for d in in_dir.iterdir() if d.is_dir()]
    
    for cls in classes:
        cls_path = in_dir / cls
        images = [f for f in cls_path.iterdir() if f.is_file() and f.suffix.lower() in ['.jpg', '.png', '.jpeg']]
        
        # Needs at least 3 images for a train/val/test split
        if len(images) < 3:
            print(f"WARNING: Class {cls} has only {len(images)} images. Skipping.")
            continue
            
        # Create class folders in splits
        (out_dir / 'train' / cls).mkdir(exist_ok=True)
        (out_dir / 'val' / cls).mkdir(exist_ok=True)
        (out_dir / 'test' / cls).mkdir(exist_ok=True)
        
        # We need stratified labels. Though doing it per-class iteratively achieves the exact same thing natively.
        # Temp ratio for the first split (train vs val+test)
        rem_pct = val_pct + test_pct
        train_imgs, rem_imgs = train_test_split(images, train_size=train_pct, random_state=42)
        
        # Second split (val vs test)
        rel_val_pct = val_pct / rem_pct
        val_imgs, test_imgs = train_test_split(rem_imgs, train_size=rel_val_pct, random_state=42)
        
        print(f"[{cls} ({len(images)})] -> Train: {len(train_imgs)}, Val: {len(val_imgs)}, Test: {len(test_imgs)}")
        
        # Copy files
        def copy_files(img_list, split_name):
            for i in img_list:
                shutil.copy2(i, out_dir / split_name / cls / i.name)
                
        copy_files(train_imgs, 'train')
        copy_files(val_imgs, 'val')
        copy_files(test_imgs, 'test')

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Split dataset")
    parser.add_argument("--input", type=str, default="dataset_augmented", help="Path to augmented dataset")
    parser.add_argument("--output", type=str, default="dataset_split", help="Path for train/val/test split output")
    args = parser.parse_args()
    
    create_stratified_split(args.input, args.output)
