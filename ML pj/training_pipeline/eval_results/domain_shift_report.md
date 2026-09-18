# Domain Shift Benchmark Report

Compares model performance on **lab-condition** images (clean, controlled lighting)
vs **field-condition** images (cluttered backgrounds, phone cameras, farm lighting).

A significant gap indicates the model has not generalized beyond clean data.

## disease_head

| Metric | Lab | Field | Gap |
|---|---|---|---|
| Accuracy | 0.2437 | 0.1625 | +0.0812 |
| Weighted F1 | 0.1565 | 0.1152 | +0.0413 |
| Macro F1 | 0.1565 | 0.1152 | +0.0413 |

**Verdict:** [ACCEPTABLE] Moderate domain shift (5-10% gap). More field augmentation recommended.

### Per-Class F1 Comparison
| Class | Lab F1 | Field F1 | Gap |
|---|---|---|---|
| blast_mild | 0.000 | 0.000 | +0.000 |
| blast_moderate | 0.000 | 0.000 | +0.000 |
| blast_severe | 0.133 | 0.190 | -0.057 |
| healthy_leaf | 0.000 | 0.000 | +0.000 |
| leaf_blight_mild | 0.000 | 0.000 | +0.000 |
| leaf_blight_moderate | 0.154 | 0.000 | +0.154 [WARN] |
| leaf_blight_severe | 0.000 | 0.091 | -0.091 |
| rust_mild | 0.184 | 0.174 | +0.010 |
| rust_moderate | 0.000 | 0.000 | +0.000 |
| rust_severe | 0.391 | 0.231 | +0.161 [WARN] |
| smut_mild | 0.000 | 0.000 | +0.000 |
| smut_moderate | 0.000 | 0.000 | +0.000 |
| smut_severe | 0.000 | 0.000 | +0.000 |
| stress_nutrient_deficiency | 0.690 | 0.333 | +0.356 [WARN] |
| stress_pest_damage | 0.000 | 0.000 | +0.000 |
| stress_sunburn | 0.952 | 0.824 | +0.129 |

## severity_head

| Metric | Lab | Field | Gap |
|---|---|---|---|
| Accuracy | 0.5687 | 0.4750 | +0.0938 |
| Weighted F1 | 0.4826 | 0.3988 | +0.0838 |
| Macro F1 | 0.4826 | 0.3988 | +0.0838 |

**Verdict:** [ACCEPTABLE] Moderate domain shift (5-10% gap). More field augmentation recommended.

### Per-Class F1 Comparison
| Class | Lab F1 | Field F1 | Gap |
|---|---|---|---|
| mild | 0.615 | 0.538 | +0.077 |
| moderate | 0.000 | 0.000 | +0.000 |
| none | 0.515 | 0.407 | +0.108 |
| severe | 0.800 | 0.650 | +0.150 [WARN] |

## stress_head

| Metric | Lab | Field | Gap |
|---|---|---|---|
| Accuracy | 0.8438 | 0.8000 | +0.0437 |
| Weighted F1 | 0.8064 | 0.7455 | +0.0609 |
| Macro F1 | 0.5286 | 0.4438 | +0.0848 |

**Verdict:** [EXCELLENT] Minimal domain shift (<5% accuracy gap)

### Per-Class F1 Comparison
| Class | Lab F1 | Field F1 | Gap |
|---|---|---|---|
| disease | 0.905 | 0.881 | +0.023 |
| healthy | 0.000 | 0.000 | +0.000 |
| stress | 0.681 | 0.450 | +0.231 [WARN] |
