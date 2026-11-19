# Model comparison

Data SHA-256: `22e12756ee58fc33015f0439de96acca9244c426374ee3086f87066d7c83eed7`  
Test set: 61 rows. Slices are small, treat slice F1 as indicative.

| Model | CV F1 | Test F1 | Recall | ROC-AUC | F1 sex_female (n=20) | F1 sex_male (n=41) | F1 age_lt_55 (n=28) | F1 age_ge_55 (n=33) | p95 ms | Size KB | Within 50 ms | Selected |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| logistic_regression | 0.801 | 0.867 | 0.929 | 0.958 | 0.923 | 0.851 | 0.933 | 0.844 | 4.4 | 5 | yes | **yes** |
| random_forest | 0.779 | 0.900 | 0.964 | 0.943 | 1.000 | 0.870 | 1.000 | 0.870 | 13.3 | 1021 | yes |  |
| gradient_boosting | 0.776 | 0.881 | 0.929 | 0.945 | 0.923 | 0.870 | 0.933 | 0.864 | 4.1 | 143 | yes |  |
