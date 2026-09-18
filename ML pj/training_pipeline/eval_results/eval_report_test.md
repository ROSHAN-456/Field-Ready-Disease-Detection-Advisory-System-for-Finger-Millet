# Evaluation Report — test set
**Model:** `./checkpoints/best_model.keras`
**Samples:** 1167

## disease_head
```
                            precision    recall  f1-score   support

              healthy_leaf       0.72      0.89      0.80        73
                blast_mild       0.56      0.48      0.52        73
            blast_moderate       0.90      0.84      0.87        73
              blast_severe       0.86      0.82      0.84        73
          leaf_blight_mild       0.60      0.47      0.52        73
      leaf_blight_moderate       0.84      0.78      0.81        73
        leaf_blight_severe       0.81      0.84      0.82        73
                 rust_mild       0.72      0.79      0.76        73
             rust_moderate       0.86      0.90      0.88        73
               rust_severe       0.93      0.97      0.95        73
                 smut_mild       0.77      0.77      0.77        73
             smut_moderate       0.91      0.94      0.93        72
               smut_severe       0.99      0.95      0.97        73
stress_nutrient_deficiency       1.00      0.96      0.98        73
        stress_pest_damage       0.68      0.77      0.72        73
            stress_sunburn       0.99      1.00      0.99        73

                  accuracy                           0.82      1167
                 macro avg       0.82      0.82      0.82      1167
              weighted avg       0.82      0.82      0.82      1167

```

## severity_head
```
              precision    recall  f1-score   support

        none       0.93      0.86      0.89       292
        mild       0.86      0.93      0.89       292
    moderate       0.98      0.97      0.97       291
      severe       0.99      0.98      0.98       292

    accuracy                           0.94      1167
   macro avg       0.94      0.94      0.94      1167
weighted avg       0.94      0.94      0.94      1167

```

## stress_head
```
              precision    recall  f1-score   support

     disease       0.95      0.98      0.97       875
      stress       0.95      0.85      0.90       219
     healthy       0.89      0.86      0.88        73

    accuracy                           0.95      1167
   macro avg       0.93      0.90      0.91      1167
weighted avg       0.95      0.95      0.95      1167

```
