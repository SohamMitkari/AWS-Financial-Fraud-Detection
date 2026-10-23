# Benchmark Results: Multi-Seed Rigorous Evaluation

Evaluated across **5 independent random seeds** (1, 2, 3, 4, 5) on 50,000 generated transactions per seed (70/30 chronological split, 80/20 natural-rate validation slice).

## Aggregate Model Comparison (Mean ± Std Across Seeds)

| Metric | Logistic Regression (Adam) | LightGBM (Gradient-Boosted Trees) | Oracle Baseline (Generative Ceiling) |
| :--- | :--- | :--- | :--- |
| **PR-AUC (Headline)** | **0.1608 ± 0.0366** | **0.2238 ± 0.0244** | **0.2220 ± 0.0203** |
| **95% Bootstrap CI** | [0.1295, 0.2072] | [0.1792, 0.2833] | [0.1783, 0.2728] |
| **ROC-AUC** | 0.9418 ± 0.0124 | 0.9505 ± 0.0122 | 0.9577 ± 0.0114 |
| **Precision** | 16.17% ± 4.79% | 25.63% ± 5.02% | 25.63% ± 7.15% |
| **Recall (Detection Rate)** | 44.46% ± 7.47% | 37.29% ± 3.25% | 51.63% ± 17.48% |
| **F1 Score** | 23.03% ± 4.26% | 30.11% ± 4.25% | 32.07% ± 3.91% |
| **Median Training Latency** | **1777.36 ms** | **155.00 ms** | N/A (Ground Truth) |

## Key Findings

1. **LightGBM vs Logistic Regression Superiority**: LightGBM achieves a headline PR-AUC of **0.2238** compared to **0.1608** for logistic regression, representing a **+39.2% relative lift**.
2. **Reaching the Oracle Noise Ceiling**: The theoretical generative ceiling (evaluating transactions directly through the generator's exact hand-coded risk logit) is **0.2220 PR-AUC**. LightGBM approaches or matches this ceiling (**0.2238**), recovering virtually all learnable signal amidst the generator's stochastic logit noise.
3. **Training Latency**: LightGBM trains in **155.00 ms** median time compared to **1777.36 ms** for iterative Adam logistic regression (**11.5x faster**), well within AWS Lambda execution constraints.

## Per-Seed Breakdown

| Seed | Model | PR-AUC | 95% Bootstrap CI | ROC-AUC | Precision | Recall | F1 | Train Time |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | Logistic Regression | 0.1493 | [0.1232, 0.1881] | 0.9587 | 17.44% | 52.76% | 26.22% | 1777.36 ms |
| 1 | LightGBM | 0.2340 | [0.1906, 0.2964] | 0.9669 | 27.78% | 36.81% | 31.66% | 126.73 ms |
| 1 | Oracle Baseline | 0.2396 | [0.1939, 0.2980] | 0.9699 | 30.20% | 37.42% | 33.42% | N/A |
| 2 | Logistic Regression | 0.1545 | [0.1259, 0.1987] | 0.9297 | 15.96% | 39.47% | 22.73% | 1792.70 ms |
| 2 | LightGBM | 0.2036 | [0.1668, 0.2544] | 0.9396 | 25.85% | 32.11% | 28.64% | 197.02 ms |
| 2 | Oracle Baseline | 0.2233 | [0.1845, 0.2717] | 0.9591 | 20.32% | 73.68% | 31.85% | N/A |
| 3 | Logistic Regression | 0.1874 | [0.1517, 0.2373] | 0.9320 | 23.22% | 34.25% | 27.68% | 1583.67 ms |
| 3 | LightGBM | 0.2452 | [0.1970, 0.3041] | 0.9411 | 27.34% | 38.67% | 32.04% | 102.05 ms |
| 3 | Oracle Baseline | 0.2253 | [0.1793, 0.2729] | 0.9390 | 31.34% | 37.57% | 34.17% | N/A |
| 4 | Logistic Regression | 0.2035 | [0.1583, 0.2715] | 0.9505 | 14.10% | 47.17% | 21.71% | 1810.75 ms |
| 4 | LightGBM | 0.2442 | [0.1960, 0.3084] | 0.9598 | 30.09% | 40.88% | 34.67% | 155.00 ms |
| 4 | Oracle Baseline | 0.2342 | [0.1859, 0.2859] | 0.9622 | 30.59% | 42.14% | 35.45% | N/A |
| 5 | Logistic Regression | 0.1092 | [0.0884, 0.1402] | 0.9381 | 10.15% | 48.67% | 16.80% | 1627.47 ms |
| 5 | LightGBM | 0.1921 | [0.1458, 0.2533] | 0.9452 | 17.07% | 38.00% | 23.55% | 170.15 ms |
| 5 | Oracle Baseline | 0.1877 | [0.1481, 0.2357] | 0.9581 | 15.71% | 67.33% | 25.47% | N/A |
