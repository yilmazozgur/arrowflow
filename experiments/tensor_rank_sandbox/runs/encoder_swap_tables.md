# Encoder swap: mean outer-fold accuracy (%), 15 folds (fit seeds averaged)

| dataset | fixed | learned | random | mlp | hybrid_knn |
|---|---|---|---|---|---|
| iris | 95.0 | 95.9 | 93.5 | 95.9 | 95.4 |
| wine | 96.8 | 98.1 | 96.0 | 98.1 | 98.1 |
| breast_cancer | 97.3 | 97.3 | 95.6 | 97.1 | 97.2 |
| wine_quality | 72.3 | 72.3 | 72.1 | 67.7 | 69.9 |
| vehicle | 80.1 | 76.4 | 70.0 | 82.8 | 75.6 |
| segment | 97.4 | 96.8 | 96.6 | 97.2 | 96.6 |
| digits | 97.2 | 97.7 | 95.5 | 97.7 | 97.6 |
| balance_scale | 94.0 | 98.4 | 89.4 | 97.8 | 97.5 |
| banknote_authentication | 99.8 | 100.0 | 99.8 | 100.0 | 100.0 |
| diabetes | 76.0 | 74.3 | 73.3 | 76.5 | 73.7 |
| ionosphere | 88.8 | 92.8 | 86.5 | 92.0 | 92.1 |
| qsar_biodeg | 87.4 | 87.8 | 87.0 | 88.0 | 86.6 |
| climate_model_simulation_crashes | 91.7 | 94.5 | 91.5 | 94.4 | 94.2 |
| hcv_egyptian_patients | 24.4 | 24.6 | 25.3 | 25.0 | 25.1 |
| mfeat_zernike | 80.7 | 82.5 | 78.3 | 81.1 | 82.8 |
| steel_plates_fault | 75.2 | 73.5 | 74.8 | 75.8 | 70.6 |
| vertebra_column | 83.1 | 83.8 | 78.2 | 84.6 | 82.2 |
| **mean of 17** | 84.5 | 85.1 | 82.5 | 85.4 | 84.4 |

## PRIMARY: learned - fixed (points; Holm over 17)

Mean difference +0.56; higher on 11/17; significantly higher on 1, significantly lower on 0. Verdict: **improves ArrowFlow**.

| dataset | difference | p | Holm p | folds better |
|---|---|---|---|---|
| iris | +0.89 | 0.4661 | 1.0000 | 7/15 |
| wine | +1.31 | 0.3022 | 1.0000 | 9/15 |
| breast_cancer | -0.06 | 0.9197 | 1.0000 | 7/15 |
| wine_quality | -0.01 | 0.9875 | 1.0000 | 9/15 |
| vehicle | -3.71 | 0.0323 | 0.4523 | 2/15 |
| segment | -0.58 | 0.0547 | 0.7115 | 2/15 |
| digits | +0.47 | 0.1405 | 1.0000 | 12/15 |
| balance_scale | +4.46 | 0.0000 | 0.0001 | 15/15 |
| banknote_authentication | +0.19 | 0.4844 | 1.0000 | 7/15 |
| diabetes | -1.69 | 0.1796 | 1.0000 | 2/15 |
| ionosphere | +3.99 | 0.0602 | 0.7221 | 12/15 |
| qsar_biodeg | +0.33 | 0.7318 | 1.0000 | 8/15 |
| climate_model_simulation_crashes | +2.86 | 0.0110 | 0.1653 | 14/15 |
| hcv_egyptian_patients | +0.20 | 0.8812 | 1.0000 | 9/15 |
| mfeat_zernike | +1.75 | 0.0055 | 0.0873 | 14/15 |
| steel_plates_fault | -1.65 | 0.0803 | 0.8832 | 1/15 |
| vertebra_column | +0.72 | 0.5298 | 1.0000 | 6/15 |

## learned - random: learned - random (points; Holm over 17)

Mean difference +2.55; higher on 15/17; significantly higher on 4, significantly lower on 0.

| dataset | difference | p | Holm p | folds better |
|---|---|---|---|---|
| iris | +2.37 | 0.2158 | 1.0000 | 11/15 |
| wine | +2.11 | 0.1685 | 1.0000 | 12/15 |
| breast_cancer | +1.66 | 0.0368 | 0.3684 | 14/15 |
| wine_quality | +0.24 | 0.7580 | 1.0000 | 7/15 |
| vehicle | +6.41 | 0.0005 | 0.0066 | 15/15 |
| segment | +0.24 | 0.5018 | 1.0000 | 9/15 |
| digits | +2.20 | 0.0067 | 0.0831 | 15/15 |
| balance_scale | +9.03 | 0.0000 | 0.0000 | 15/15 |
| banknote_authentication | +0.22 | 0.2425 | 1.0000 | 10/15 |
| diabetes | +0.97 | 0.2643 | 1.0000 | 10/15 |
| ionosphere | +6.24 | 0.0002 | 0.0024 | 15/15 |
| qsar_biodeg | +0.76 | 0.4510 | 1.0000 | 10/15 |
| climate_model_simulation_crashes | +3.05 | 0.0064 | 0.0831 | 14/15 |
| hcv_egyptian_patients | -0.66 | 0.6247 | 1.0000 | 5/15 |
| mfeat_zernike | +4.15 | 0.0000 | 0.0000 | 15/15 |
| steel_plates_fault | -1.27 | 0.1123 | 1.0000 | 3/15 |
| vertebra_column | +5.63 | 0.0132 | 0.1450 | 15/15 |

## random - fixed: random - fixed (points; Holm over 17)

Mean difference -1.99; higher on 1/17; significantly higher on 0, significantly lower on 3.

| dataset | difference | p | Holm p | folds better |
|---|---|---|---|---|
| iris | -1.48 | 0.3305 | 1.0000 | 4/15 |
| wine | -0.81 | 0.5592 | 1.0000 | 5/15 |
| breast_cancer | -1.72 | 0.0165 | 0.1977 | 0/15 |
| wine_quality | -0.26 | 0.7853 | 1.0000 | 6/15 |
| vehicle | -10.13 | 0.0000 | 0.0005 | 0/15 |
| segment | -0.82 | 0.0054 | 0.0755 | 0/15 |
| digits | -1.73 | 0.0130 | 0.1686 | 0/15 |
| balance_scale | -4.57 | 0.0000 | 0.0004 | 0/15 |
| banknote_authentication | -0.03 | 0.8062 | 1.0000 | 2/15 |
| diabetes | -2.66 | 0.1298 | 1.0000 | 2/15 |
| ionosphere | -2.25 | 0.0927 | 0.9273 | 3/15 |
| qsar_biodeg | -0.43 | 0.4729 | 1.0000 | 5/15 |
| climate_model_simulation_crashes | -0.19 | 0.3578 | 1.0000 | 0/15 |
| hcv_egyptian_patients | +0.86 | 0.4020 | 1.0000 | 10/15 |
| mfeat_zernike | -2.40 | 0.0011 | 0.0166 | 0/15 |
| steel_plates_fault | -0.38 | 0.6559 | 1.0000 | 6/15 |
| vertebra_column | -4.91 | 0.0261 | 0.2867 | 0/15 |

## learned - mlp: learned - mlp (points; Holm over 17)

Mean difference -0.30; higher on 6/17; significantly higher on 1, significantly lower on 2.

| dataset | difference | p | Holm p | folds better |
|---|---|---|---|---|
| iris | -0.00 | 1.0000 | 1.0000 | 2/15 |
| wine | -0.00 | 0.9981 | 1.0000 | 5/15 |
| breast_cancer | +0.12 | 0.6790 | 1.0000 | 5/15 |
| wine_quality | +4.64 | 0.0003 | 0.0049 | 15/15 |
| vehicle | -6.42 | 0.0007 | 0.0118 | 0/15 |
| segment | -0.39 | 0.3477 | 1.0000 | 5/15 |
| digits | -0.04 | 0.9021 | 1.0000 | 7/15 |
| balance_scale | +0.59 | 0.3782 | 1.0000 | 10/15 |
| banknote_authentication | +0.00 | 1.0000 | 1.0000 | 0/15 |
| diabetes | -2.21 | 0.0307 | 0.4301 | 1/15 |
| ionosphere | +0.79 | 0.3811 | 1.0000 | 10/15 |
| qsar_biodeg | -0.19 | 0.8346 | 1.0000 | 5/15 |
| climate_model_simulation_crashes | +0.14 | 0.7772 | 1.0000 | 8/15 |
| hcv_egyptian_patients | -0.43 | 0.7619 | 1.0000 | 7/15 |
| mfeat_zernike | +1.34 | 0.1354 | 1.0000 | 12/15 |
| steel_plates_fault | -2.30 | 0.0024 | 0.0356 | 0/15 |
| vertebra_column | -0.72 | 0.6454 | 1.0000 | 5/15 |

## learned - hybrid_knn: learned - hybrid_knn (points; Holm over 17)

Mean difference +0.67; higher on 15/17; significantly higher on 0, significantly lower on 0.

| dataset | difference | p | Holm p | folds better |
|---|---|---|---|---|
| iris | +0.44 | 0.3510 | 1.0000 | 4/15 |
| wine | +0.00 | 0.9972 | 1.0000 | 3/15 |
| breast_cancer | +0.06 | 0.8856 | 1.0000 | 7/15 |
| wine_quality | +2.41 | 0.0377 | 0.5655 | 13/15 |
| vehicle | +0.76 | 0.5586 | 1.0000 | 9/15 |
| segment | +0.25 | 0.4128 | 1.0000 | 10/15 |
| digits | +0.12 | 0.5253 | 1.0000 | 10/15 |
| balance_scale | +0.94 | 0.0308 | 0.4928 | 14/15 |
| banknote_authentication | +0.04 | 0.6534 | 1.0000 | 1/15 |
| diabetes | +0.59 | 0.4678 | 1.0000 | 10/15 |
| ionosphere | +0.70 | 0.3283 | 1.0000 | 11/15 |
| qsar_biodeg | +1.13 | 0.1449 | 1.0000 | 12/15 |
| climate_model_simulation_crashes | +0.31 | 0.5084 | 1.0000 | 8/15 |
| hcv_egyptian_patients | -0.55 | 0.6964 | 1.0000 | 5/15 |
| mfeat_zernike | -0.37 | 0.6567 | 1.0000 | 7/15 |
| steel_plates_fault | +2.93 | 0.0050 | 0.0856 | 15/15 |
| vertebra_column | +1.61 | 0.2432 | 1.0000 | 11/15 |
