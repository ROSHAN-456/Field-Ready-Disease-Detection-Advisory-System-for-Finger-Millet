# Real Dataset Collection Checklist

To swap out the synthetic data and begin training on real Finger Millet field images, ensure your images are dumped precisely following this structure. The `run_pipeline.py` orchestrator will handle all validation, augmentation, and splitting automatically as long as the inputs match this layout.

## 🎯 Target Quantity
For robust transfer learning, aim for **300 - 500 images per folder**. 
*(The pipeline's augmentation script will automatically generate synthetic permutations for any folder under this limit to reach 500 automatically, but raw data is always superior).*

## 📁 Required Folder Layout

Drop all collected raw images into a master folder (e.g., `d:\ML pj\real_dataset_raw\`).
It **must** contain exactly these 16 sub-folders:

### 1. Healthy
- [ ] `healthy_leaf/` - Normal, green leaves taken in bright and shaded environments.

### 2. Finger Millet Diseases (3 Stages)
*The model uses these names to automatically extract severity targets!*
- [ ] `blast_mild/` - Early blast (small spindle-shaped lesions)
- [ ] `blast_moderate/`
- [ ] `blast_severe/` - Leaf heavily withered/greyed by blast
- [ ] `leaf_blight_mild/`
- [ ] `leaf_blight_moderate/`
- [ ] `leaf_blight_severe/`
- [ ] `rust_mild/`
- [ ] `rust_moderate/`
- [ ] `rust_severe/`
- [ ] `smut_mild/`
- [ ] `smut_moderate/`
- [ ] `smut_severe/`

### 3. Non-Disease Environmental Stressors (Look-Alikes)
*Crucial to prevent the app assigning agro-chemicals when a plant just needs water or fertilizer.*
- [ ] `stress_nutrient_deficiency/` - Yellowing, chlorosis, lacking nitrogen.
- [ ] `stress_pest_damage/` - Clean holes, insect chewing.
- [ ] `stress_sunburn/` - Burnt, crispy margins from excessive heat/drought.

---

### Benchmark Folders (Optional but highly recommended)
To generate the Domain Shift Benchmark properly, create a separate directory (e.g., `d:\ML pj\real_benchmark\`) containing TWO subfolders:
- `lab/` - (Containing the same 16 subfolders as above) with clean, controlled, white-background test images.
- `field/` - (Containing the same 16 subfolders) with messy, real-world, obscured field test images.
