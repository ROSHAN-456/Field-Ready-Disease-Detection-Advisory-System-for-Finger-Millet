"""
generate_synthetic_dataset.py — Creates a small synthetic dataset for pipeline testing.

Generates simple leaf-like images with synthetic disease patterns so we can
validate the entire training → evaluation → export pipeline end-to-end
before real data arrives.

Each class gets configurable number of images (default 30 per class).
"""

import os
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from pathlib import Path

# All 16 classes from our schema
CLASSES = [
    "healthy_leaf",
    "blast_mild", "blast_moderate", "blast_severe",
    "leaf_blight_mild", "leaf_blight_moderate", "leaf_blight_severe",
    "rust_mild", "rust_moderate", "rust_severe",
    "smut_mild", "smut_moderate", "smut_severe",
    "stress_nutrient_deficiency", "stress_pest_damage", "stress_sunburn",
]

# Color palettes for different conditions
LEAF_GREENS = [(34, 139, 34), (0, 128, 0), (85, 107, 47), (60, 120, 50)]
DISEASE_COLORS = {
    "blast": [(101, 67, 33), (139, 90, 43), (80, 60, 30)],          # Brown spots
    "leaf_blight": [(160, 82, 45), (178, 134, 61), (120, 80, 40)],  # Tan/yellow patches
    "rust": [(178, 102, 0), (204, 85, 0), (210, 120, 40)],          # Orange pustules
    "smut": [(40, 40, 40), (60, 60, 60), (80, 60, 80)],             # Dark sooty patches
}
STRESS_COLORS = {
    "stress_nutrient_deficiency": [(200, 200, 60), (180, 180, 40)],  # Yellowing/chlorosis
    "stress_pest_damage": [(100, 80, 60)],                            # Brown holes
    "stress_sunburn": [(220, 200, 150), (240, 220, 180)],            # Bleached/scorched
}


def generate_leaf_base(size=224):
    """Create a base leaf-colored image with some natural variation."""
    bg_color = random.choice([(80, 50, 30), (100, 90, 70), (60, 80, 60), (120, 100, 80)])
    # Create a larger image for drawing the leaf so we can rotate it without cropping
    canvas_size = int(size * 1.5)
    img = Image.new("RGBA", (canvas_size, canvas_size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Draw a leaf shape (elongated ellipse)
    leaf_color = random.choice(LEAF_GREENS)
    # Add slight color variation
    leaf_color = tuple(c + random.randint(-15, 15) for c in leaf_color)
    leaf_color = tuple(max(0, min(255, c)) for c in leaf_color)

    cx, cy = canvas_size // 2, canvas_size // 2
    # Greater shape variation
    rx, ry = random.randint(70, 110), random.randint(30, 60)
    draw.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=leaf_color + (255,))

    # Add some vein lines
    vein_color = tuple(max(0, c - 30) for c in leaf_color)
    draw.line([(cx - rx + 10, cy), (cx + rx - 10, cy)], fill=vein_color + (255,), width=2)
    for i in range(random.randint(4, 8)):
        offset = random.randint(10, 40) * (1 if i % 2 == 0 else -1)
        draw.line([(cx, cy), (cx + offset, cy + random.randint(-ry + 5, ry - 5))],
                  fill=vein_color + (255,), width=1)
                  
    # Rotate the leaf for orientation variation
    angle = random.randint(-60, 60)
    img = img.rotate(angle, resample=Image.BICUBIC, expand=False)
    
    # Composite onto background
    bg = Image.new("RGB", (canvas_size, canvas_size), bg_color)
    bg.paste(img, (0, 0), img)
    
    # Crop center
    left = (canvas_size - size) // 2
    top = (canvas_size - size) // 2
    bg = bg.crop((left, top, left + size, top + size))

    return bg


def add_disease_spots(img, disease_type, severity):
    """Add synthetic disease lesion spots."""
    draw = ImageDraw.Draw(img)
    size = img.size[0]

    # Number and size of spots scale heavily with severity for distinction
    severity_scale = {
        "mild": (random.randint(1, 4), 4, 8), 
        "moderate": (random.randint(7, 12), 8, 14), 
        "severe": (random.randint(15, 30), 12, 25)
    }
    n_spots, min_r, max_r = severity_scale.get(severity, (3, 6, 10))

    colors = DISEASE_COLORS.get(disease_type, [(100, 80, 60)])

    # For severe, we also add some general leaf discoloration
    if severity == "severe":
        overlay = Image.new('RGBA', img.size, (200, 200, 50, 40)) # Yellowish tint
        img = Image.alpha_composite(img.convert('RGBA'), overlay).convert('RGB')
        draw = ImageDraw.Draw(img)

    for _ in range(n_spots):
        # Allow spots to appear anywhere, not just the center
        x = random.randint(size // 5, 4 * size // 5)
        y = random.randint(size // 5, 4 * size // 5)
        r = random.randint(min_r, max_r)
        
        # Shape variation: oval vs circle
        rx = r
        ry = r * random.uniform(0.6, 1.4)
        
        color = random.choice(colors)
        color = tuple(max(0, min(255, c + random.randint(-20, 20))) for c in color)
        
        # Severe spots might have a darker center or a halo
        if severity == "severe" and random.random() > 0.5:
            # Halo
            draw.ellipse([x - rx - 2, y - ry - 2, x + rx + 2, y + ry + 2], fill=(200, 200, 100))
        
        draw.ellipse([x - rx, y - ry, x + rx, y + ry], fill=color)

    # Add varying blur for realism
    blur_amount = {"mild": 0.3, "moderate": 0.6, "severe": 1.2}
    img = img.filter(ImageFilter.GaussianBlur(radius=blur_amount.get(severity, 0.5)))

    return img


def add_stress_pattern(img, stress_type):
    """Add synthetic stress patterns."""
    draw = ImageDraw.Draw(img)
    size = img.size[0]

    if stress_type == "stress_nutrient_deficiency":
        # Yellowing patches
        colors = STRESS_COLORS[stress_type]
        for _ in range(random.randint(3, 8)):
            x = random.randint(size // 4, 3 * size // 4)
            y = random.randint(size // 4, 3 * size // 4)
            r = random.randint(15, 30)
            draw.ellipse([x - r, y - r, x + r, y + r],
                         fill=random.choice(colors))

    elif stress_type == "stress_pest_damage":
        # Small irregular holes
        for _ in range(random.randint(5, 15)):
            x = random.randint(size // 4, 3 * size // 4)
            y = random.randint(size // 4, 3 * size // 4)
            r = random.randint(2, 6)
            draw.ellipse([x - r, y - r, x + r, y + r],
                         fill=random.choice(STRESS_COLORS[stress_type]))

    elif stress_type == "stress_sunburn":
        # Bleached leaf edge regions
        colors = STRESS_COLORS[stress_type]
        for _ in range(random.randint(2, 5)):
            x = random.randint(size // 3, 2 * size // 3)
            y = random.randint(size // 3, 2 * size // 3)
            r = random.randint(20, 40)
            draw.ellipse([x - r, y - r, x + r, y + r],
                         fill=random.choice(colors))

    return img


def generate_dataset(output_dir: str, images_per_class: int = 30):
    """Generate the full synthetic dataset."""
    out = Path(output_dir)

    print(f"Generating synthetic dataset: {images_per_class} images × {len(CLASSES)} classes")
    print(f"Output: {output_dir}")
    print(f"Total images: {images_per_class * len(CLASSES)}\n")

    for cls in CLASSES:
        cls_dir = out / cls
        cls_dir.mkdir(parents=True, exist_ok=True)

        for i in range(images_per_class):
            img = generate_leaf_base()

            if cls == "healthy_leaf":
                pass  # Clean leaf, no modifications
            elif cls.startswith("stress_"):
                img = add_stress_pattern(img, cls)
            else:
                # Disease class: extract disease type and severity
                parts = cls.rsplit("_", 1)
                severity = parts[-1]  # mild/moderate/severe
                disease_type = cls.replace(f"_{severity}", "")
                img = add_disease_spots(img, disease_type, severity)

            # Random quality degradation (simulating phone cameras)
            if random.random() > 0.5:
                img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 1.0)))

            img.save(str(cls_dir / f"{cls}_{i:04d}.jpg"), "JPEG",
                     quality=random.randint(60, 95))

        print(f"  [OK] {cls}: {images_per_class} images")

    print(f"\n[DONE] Total: {images_per_class * len(CLASSES)} images in {len(CLASSES)} classes")


def generate_split_dataset(base_output: str, images_per_class: int = 150):
    """Generate just the raw dataset. Splitting will be handled by split_dataset.py."""
    # We no longer split here directly, to fit the standard pipeline flow
    generate_dataset(base_output, images_per_class)
    print()


def generate_benchmark_data(output_dir: str, images_per_class: int = 10):
    """Generate lab and field benchmark sets for domain shift testing."""
    lab_dir = os.path.join(output_dir, "lab")
    field_dir = os.path.join(output_dir, "field")

    print("=" * 50)
    print("Generating LAB benchmark images (clean)...")
    print("=" * 50)
    generate_dataset(lab_dir, images_per_class)

    print("\n" + "=" * 50)
    print("Generating FIELD benchmark images (noisy)...")
    print("=" * 50)
    # Field images get extra degradation
    generate_dataset(field_dir, images_per_class)

    # Apply extra noise/blur to field images
    field_path = Path(field_dir)
    for img_file in field_path.rglob("*.jpg"):
        img = Image.open(img_file)
        # Extra blur + noise
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 2.0)))
        # Brightness shift
        arr = np.array(img, dtype=np.float32)
        arr = arr * random.uniform(0.6, 1.4)
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)
        img.save(str(img_file), "JPEG", quality=random.randint(40, 70))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate synthetic dataset for pipeline testing")
    parser.add_argument("--output", type=str, default="../dataset_pipeline/dataset_raw",
                        help="Output directory for raw dataset")
    parser.add_argument("--benchmark_output", type=str, default="./benchmark_data",
                        help="Output directory for domain shift benchmark data")
    parser.add_argument("--images_per_class", type=int, default=150,
                        help="Number of base images per class")
    parser.add_argument("--benchmark_images", type=int, default=10,
                        help="Number of images per class for benchmark sets")
    args = parser.parse_args()

    print("=" * 60)
    print("  GENERATING SYNTHETIC DATASET FOR PIPELINE TESTING")
    print("=" * 60)
    print()

    # 1. Generate raw dataset
    generate_split_dataset(args.output, args.images_per_class)

    # 2. Generate benchmark data
    print("\n")
    generate_benchmark_data(args.benchmark_output, args.benchmark_images)

    print("\n\n[DONE] All synthetic data generated successfully!")
    print(f"   Dataset: {args.output}")
    print(f"   Benchmark: {args.benchmark_output}")
