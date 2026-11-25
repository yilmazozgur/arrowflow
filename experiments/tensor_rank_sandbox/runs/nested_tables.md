# Nested CV: mean outer-fold accuracy (%), 15 folds (fit seeds averaged)

| dataset | hybrid_target_mlp[rank] | hybrid_target_mlp[knn] | hybrid_frozen_mlp[rank] | hybrid_frozen_mlp[knn] | hybrid_target_linear[rank] | hybrid_target_linear[knn] | arrowflow_full_knn | mlp | svc_rbf | random_forest | gradient_boosting | numeric_knn | dummy |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| iris | 95.0 | 95.4 | 93.0 | 93.0 | 88.9 | 89.5 | 95.0 | 95.9 | 96.0 | 94.7 | 94.4 | 95.3 | 33.3 |
| wine | 98.1 | 98.1 | 94.8 | 93.8 | 96.7 | 97.9 | 96.8 | 98.1 | 98.5 | 98.2 | 97.4 | 96.6 | 39.9 |
| breast_cancer | 97.2 | 97.2 | 94.7 | 95.6 | 96.8 | 96.8 | 97.3 | 97.1 | 97.7 | 95.7 | 96.1 | 96.5 | 62.7 |
| wine_quality | 56.6 | 69.9 | 54.2 | 70.9 | 56.4 | 68.7 | 72.3 | 67.7 | 66.9 | 72.8 | 69.3 | 71.2 | 46.5 |
| vehicle | 75.0 | 75.6 | 51.3 | 68.2 | 76.0 | 78.3 | 80.1 | 82.8 | 83.5 | 74.3 | 75.7 | 70.7 | 25.8 |
| segment | 93.7 | 96.6 | 82.1 | 95.8 | 90.4 | 96.9 | 97.4 | 97.2 | 96.7 | 98.1 | 97.9 | 96.6 | 14.3 |
| digits | 97.0 | 97.6 | 79.1 | 92.9 | 93.5 | 95.5 | 97.2 | 97.7 | 98.2 | 97.6 | 96.7 | 97.9 | 10.1 |
| balance_scale | 96.3 | 97.5 | 83.7 | 87.9 | 89.7 | 92.1 | 94.0 | 97.8 | 97.9 | 86.2 | 91.6 | 89.8 | 45.8 |
| banknote_authentication | 100.0 | 100.0 | 96.0 | 99.8 | 97.0 | 99.8 | 99.8 | 100.0 | 100.0 | 99.1 | 99.7 | 99.7 | 55.5 |
| diabetes | 70.7 | 73.7 | 69.5 | 74.6 | 73.6 | 75.2 | 76.0 | 76.5 | 77.3 | 76.1 | 76.0 | 75.5 | 65.1 |
| ionosphere | 92.0 | 92.1 | 82.0 | 85.4 | 86.8 | 88.8 | 88.8 | 92.0 | 94.1 | 93.0 | 93.4 | 89.8 | 64.1 |
| qsar_biodeg | 86.3 | 86.6 | 78.1 | 85.1 | 83.6 | 86.5 | 87.4 | 88.0 | 87.8 | 87.0 | 86.8 | 86.3 | 66.3 |
| climate_model_simulation_crashes | 94.1 | 94.2 | 82.3 | 91.3 | 93.3 | 92.7 | 91.7 | 94.4 | 94.6 | 93.6 | 94.1 | 93.2 | 91.5 |
| hcv_egyptian_patients | 25.6 | 25.1 | 25.1 | 25.6 | 24.7 | 25.2 | 24.4 | 25.0 | 24.5 | 24.1 | 23.1 | 25.9 | 26.1 |
| mfeat_zernike | 83.0 | 82.8 | 65.7 | 73.7 | 79.6 | 79.0 | 80.7 | 81.1 | 82.9 | 77.7 | 77.6 | 80.3 | 10.0 |
| steel_plates_fault | 65.3 | 70.6 | 55.9 | 72.7 | 61.8 | 69.9 | 75.2 | 75.8 | 76.1 | 78.8 | 79.0 | 74.5 | 34.7 |
| vertebra_column | 81.4 | 82.2 | 78.2 | 79.1 | 83.0 | 83.0 | 83.1 | 84.6 | 86.0 | 84.2 | 82.2 | 80.2 | 48.4 |
| **mean of 17** | 82.8 | 84.4 | 74.5 | 81.5 | 80.7 | 83.3 | 84.5 | 85.4 | 85.8 | 84.2 | 84.2 | 83.5 | 43.5 |

## P1: hybrid_target_mlp[rank] - hybrid_frozen_mlp[rank] (points; Holm over 17)

Higher mean on 17/17; significantly higher on 11, significantly lower on 0.

| dataset | difference | p | Holm p | folds better |
|---|---|---|---|---|
| iris | +2.07 | 0.2639 | 1.0000 | 10/15 |
| wine | +3.31 | 0.0094 | 0.0564 | 15/15 |
| breast_cancer | +2.48 | 0.0036 | 0.0250 | 14/15 |
| wine_quality | +2.43 | 0.2730 | 1.0000 | 11/15 |
| vehicle | +23.72 | 0.0000 | 0.0000 | 15/15 |
| segment | +11.65 | 0.0000 | 0.0000 | 15/15 |
| digits | +17.89 | 0.0000 | 0.0000 | 15/15 |
| balance_scale | +12.59 | 0.0000 | 0.0000 | 15/15 |
| banknote_authentication | +3.93 | 0.0000 | 0.0000 | 15/15 |
| diabetes | +1.17 | 0.6648 | 1.0000 | 9/15 |
| ionosphere | +10.00 | 0.0000 | 0.0001 | 15/15 |
| qsar_biodeg | +8.12 | 0.0000 | 0.0000 | 15/15 |
| climate_model_simulation_crashes | +11.77 | 0.0000 | 0.0000 | 15/15 |
| hcv_egyptian_patients | +0.48 | 0.6819 | 1.0000 | 8/15 |
| mfeat_zernike | +17.27 | 0.0000 | 0.0000 | 15/15 |
| steel_plates_fault | +9.38 | 0.0000 | 0.0000 | 15/15 |
| vertebra_column | +3.23 | 0.0344 | 0.1722 | 13/15 |

## P2: hybrid_target_mlp[rank] - mlp (points; Holm over 17)

Higher mean on 3/17; significantly higher on 0, significantly lower on 4.

| dataset | difference | p | Holm p | folds better |
|---|---|---|---|---|
| iris | -0.81 | 0.4586 | 1.0000 | 2/15 |
| wine | -0.07 | 0.9144 | 1.0000 | 3/15 |
| breast_cancer | +0.06 | 0.8957 | 1.0000 | 5/15 |
| wine_quality | -11.05 | 0.0000 | 0.0004 | 0/15 |
| vehicle | -7.79 | 0.0001 | 0.0017 | 0/15 |
| segment | -3.49 | 0.0000 | 0.0000 | 0/15 |
| digits | -0.71 | 0.0960 | 1.0000 | 1/15 |
| balance_scale | -1.55 | 0.1305 | 1.0000 | 4/15 |
| banknote_authentication | -0.04 | 0.6534 | 1.0000 | 0/15 |
| diabetes | -5.79 | 0.0069 | 0.0901 | 0/15 |
| ionosphere | -0.00 | 0.9972 | 1.0000 | 8/15 |
| qsar_biodeg | -1.70 | 0.0962 | 1.0000 | 3/15 |
| climate_model_simulation_crashes | -0.29 | 0.6364 | 1.0000 | 8/15 |
| hcv_egyptian_patients | +0.57 | 0.6000 | 1.0000 | 7/15 |
| mfeat_zernike | +1.84 | 0.0928 | 1.0000 | 13/15 |
| steel_plates_fault | -10.49 | 0.0000 | 0.0000 | 0/15 |
| vertebra_column | -3.12 | 0.1093 | 1.0000 | 2/15 |

## P3: hybrid_target_mlp[knn] - arrowflow_full_knn (points; Holm over 17)

Higher mean on 9/17; significantly higher on 1, significantly lower on 2.

| dataset | difference | p | Holm p | folds better |
|---|---|---|---|---|
| iris | +0.44 | 0.6711 | 1.0000 | 7/15 |
| wine | +1.31 | 0.2561 | 1.0000 | 10/15 |
| breast_cancer | -0.12 | 0.8247 | 1.0000 | 6/15 |
| wine_quality | -2.42 | 0.0573 | 0.6301 | 1/15 |
| vehicle | -4.48 | 0.0027 | 0.0405 | 0/15 |
| segment | -0.83 | 0.0264 | 0.3165 | 2/15 |
| digits | +0.35 | 0.2436 | 1.0000 | 9/15 |
| balance_scale | +3.52 | 0.0000 | 0.0005 | 15/15 |
| banknote_authentication | +0.15 | 0.4151 | 1.0000 | 7/15 |
| diabetes | -2.29 | 0.1238 | 1.0000 | 2/15 |
| ionosphere | +3.29 | 0.0853 | 0.8532 | 12/15 |
| qsar_biodeg | -0.80 | 0.3654 | 1.0000 | 3/15 |
| climate_model_simulation_crashes | +2.55 | 0.0100 | 0.1400 | 14/15 |
| hcv_egyptian_patients | +0.75 | 0.5676 | 1.0000 | 9/15 |
| mfeat_zernike | +2.12 | 0.0164 | 0.2131 | 14/15 |
| steel_plates_fault | -4.58 | 0.0000 | 0.0001 | 0/15 |
| vertebra_column | -0.90 | 0.5986 | 1.0000 | 5/15 |
