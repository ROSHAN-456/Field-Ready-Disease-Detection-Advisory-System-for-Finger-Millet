# Evaluation Report — test set
**Model:** `./checkpoints/best_model.keras`
**Samples:** 80

## disease_head
```
                            precision    recall  f1-score   support

              healthy_leaf       0.00      0.00      0.00         5
                blast_mild       0.00      0.00      0.00         5
            blast_moderate       0.00      0.00      0.00         5
              blast_severe       0.25      0.20      0.22         5
          leaf_blight_mild       0.00      0.00      0.00         5
      leaf_blight_moderate       0.00      0.00      0.00         5
        leaf_blight_severe       0.17      0.20      0.18         5
                 rust_mild       0.13      1.00      0.23         5
             rust_moderate       0.00      0.00      0.00         5
               rust_severe       0.28      1.00      0.43         5
                 smut_mild       0.00      0.00      0.00         5
             smut_moderate       0.00      0.00      0.00         5
               smut_severe       0.00      0.00      0.00         5
stress_nutrient_deficiency       0.83      1.00      0.91         5
        stress_pest_damage       0.00      0.00      0.00         5
            stress_sunburn       1.00      1.00      1.00         5

                  accuracy                           0.28        80
                 macro avg       0.17      0.28      0.19        80
              weighted avg       0.17      0.28      0.19        80

```

## severity_head
```
              precision    recall  f1-score   support

        none       1.00      0.50      0.67        20
        mild       0.45      0.90      0.60        20
    moderate       1.00      0.05      0.10        20
      severe       0.66      0.95      0.78        20

    accuracy                           0.60        80
   macro avg       0.78      0.60      0.53        80
weighted avg       0.78      0.60      0.53        80

```

## stress_head
```
              precision    recall  f1-score   support

     disease       0.86      1.00      0.92        60
      stress       1.00      0.67      0.80        15
     healthy       0.00      0.00      0.00         5

    accuracy                           0.88        80
   macro avg       0.62      0.56      0.57        80
weighted avg       0.83      0.88      0.84        80

```
