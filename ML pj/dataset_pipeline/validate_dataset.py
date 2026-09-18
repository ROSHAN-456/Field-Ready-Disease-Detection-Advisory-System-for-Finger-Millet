import os
import sys
import hashlib
from collections import defaultdict
from PIL import Image

def validate_dataset(data_dir: str):
    """
    Validates a dataset folder against the expected schema.
    - Checks for corrupt images
    - Identifies exact duplicates using SHA-256
    - Computes class balance
    """
    print(f"--- Validating Dataset at {data_dir} ---")
    
    if not os.path.exists(data_dir):
        print(f"Directory {data_dir} does not exist.")
        sys.exit(1)

    expected_classes = {
        'healthy_leaf',
        'blast_mild', 'blast_moderate', 'blast_severe',
        'leaf_blight_mild', 'leaf_blight_moderate', 'leaf_blight_severe',
        'rust_mild', 'rust_moderate', 'rust_severe',
        'smut_mild', 'smut_moderate', 'smut_severe',
        'stress_nutrient_deficiency', 'stress_pest_damage', 'stress_sunburn'
    }

    classes = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    
    missing_classes = expected_classes - set(classes)
    if missing_classes:
        print(f"[ERROR] Missing required classes: {missing_classes}")
        sys.exit(1)
        
    extra_classes = set(classes) - expected_classes
    if extra_classes:
        print(f"[WARNING] Found extra classes not in schema: {extra_classes}. These will be ignored.")
        # Filter strictly to expected
        classes = list(expected_classes)
    
    corrupt_files = []
    hash_map = defaultdict(list)
    class_counts = {}
    total_images = 0

    print("Checking for corrupt files and duplicates...")
    for cls in classes:
        cls_dir = os.path.join(data_dir, cls)
        files = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        class_counts[cls] = len(files)
        
        if len(files) < 10:
            print(f"[ERROR] Class '{cls}' has less than 10 images ({len(files)}). Please add more raw data.")
            sys.exit(1)
        
        for file in files:
            file_path = os.path.join(cls_dir, file)
            total_images += 1
            
            # 1. Check for corruption
            try:
                with Image.open(file_path) as img:
                    img.verify() # Verify it's an image
            except Exception as e:
                corrupt_files.append((file_path, str(e)))
                continue
                
            # 2. Check for exact duplicates
            with open(file_path, "rb") as f:
                file_hash = hashlib.sha256(f.read()).hexdigest()
                hash_map[file_hash].append(file_path)

    # Summarize Corruption
    print(f"\n[Corruption Check]")
    if corrupt_files:
        print(f"FAILED: Found {len(corrupt_files)} corrupt files!")
        for f, err in corrupt_files[:5]:
            print(f" - {f}: {err}")
    else:
        print("PASSED: No corrupt images detected.")

    # Summarize Duplicates
    print(f"\n[Duplicate Check]")
    duplicates = {h: paths for h, paths in hash_map.items() if len(paths) > 1}
    if duplicates:
        print(f"WARNING: Found {len(duplicates)} sets of duplicated files.")
        for h, paths in list(duplicates.items())[:3]:
            print(f" - Duplicate set: {paths}")
    else:
        print("PASSED: No exact duplicates detected.")

    # Summarize Balance
    print(f"\n[Class Balance Check]")
    print(f"Total valid images analyzed: {total_images}")
    if total_images == 0:
        return
        
    avg_per_class = total_images / len(classes)
    for cls, count in class_counts.items():
        ratio = count / avg_per_class
        status = "OK"
        if ratio < 0.5: status = "UNDER-REPRESENTED"
        if ratio > 1.5: status = "OVER-REPRESENTED"
        print(f" - {cls}: {count} images ({status})")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Validate dataset")
    parser.add_argument("--dataset", type=str, default="dataset_augmented", help="Path to raw or augmented dataset")
    args = parser.parse_args()
    
    validate_dataset(args.dataset)
