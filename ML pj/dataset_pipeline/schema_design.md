# Finger Millet & Related Crops Dataset Schema

## Overview
To ensure the app correctly detects both the disease type and its severity while minimizing false positives from non-disease stress factors, the training dataset folder structure must be heavily curated.

## Root Directory: `dataset/raw/`

The dataset should be structurally flattened into discrete class folders to allow seamless iteration for image generators. Each folder name represents a distinct classification class output.

### 1. Healthy
*   `healthy_leaf/` - Completely healthy leaves under varied lighting.

### 2. Finger Millet Diseases (Class + 3 Severities)
*   `blast_mild/` - Early stage: Small brown/grey lesions.
*   `blast_moderate/` - Mid stage: Elliptical/spindle lesions with grey centers.
*   `blast_severe/` - Late stage: Lesions coalesce, leaf dries up/withers.

*   `leaf_blight_mild/`
*   `leaf_blight_moderate/`
*   `leaf_blight_severe/`

*   `rust_mild/`
*   `rust_moderate/`
*   `rust_severe/`

*   `smut_mild/`
*   `smut_moderate/`
*   `smut_severe/`

### 3. Non-Disease Stress Factors (Look-alikes)
These act as negative classes to explicitly teach the model *not* to predict a disease when it's an environmental stressor.
*   `stress_nutrient_deficiency/` - Yellowing, chlorosis, specifically Nitrogen/Iron deficiency.
*   `stress_pest_damage/` - Holes, chewing marks, insect webbing.
*   `stress_sunburn/` - Scorched leaf margins, bleaching.

## Validation Expectations
- **Balance**: Ideally max 20% variance between class counts. Realistically, severe cases might be fewer. Data augmentation will be applied locally on minority classes.
- **Artifacts**: Ensure images include shadows, farm soil background, and sun glares. Do not only use uniform white-background lab images.
