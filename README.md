# Serverless Real-Time Financial Fraud Detection Platform

An event-driven serverless machine learning system for financial transaction surveillance, behavioral feature engineering, and automated risk triage. The platform combines AWS Lambda compute with LightGBM gradient-boosted decision trees, Amazon API Gateway HTTP API v2, Amazon S3 artifact storage, and an interactive static single-page client with in-browser tree ensemble scoring.

---

## Architectural Topology

The system uses a decoupled cloud architecture designed for zero idle infrastructure cost, automated out-of-time model training, and continuous real-time client surveillance:

```text
+---------------------------------------------------------------------------------------+
|                                 Client Layer (Amazon S3)                              |
|                                                                                       |
|   Static Single Page Application (HTML5 / Vanilla JavaScript / Pure CSS / Canvas)     |
|   - Hosted via Amazon S3 + CloudFront (Zero external framework runtime)     |
|   - Interactive 7-Stage Pipeline Orchestration (Stages 1 through 7 + Live Dashboard)  |
|   - Real-time client scoring via exported LightGBM decision trees (pure JS walk)      |
|   - Automatic browser fallback to logistic regression if cloud backend is unreachable |
+-------------------------------------------+-------------------------------------------+
                                            |
                      HTTPS JSON REST Calls (CORS Preflight + x-api-key)
                                            |
+-------------------------------------------v-------------------------------------------+
|                                 Ingress (Amazon API Gateway)                          |
|                                                                                       |
|   Amazon API Gateway HTTP API v2                                                      |
|   - Regional Low-Latency Endpoint: ap-southeast-2 (Sydney)                            |
|   - Native CORS Preflight Handling (OPTIONS, GET, POST)                               |
|   - Routes: GET /health, POST /train, POST /predict, GET /model                       |
|   - Shared Secret Authentication on POST /train (x-api-key header)            |
|   - 30-Second HTTP Integration Timeout Envelope                                       |
+-------------------------------------------+-------------------------------------------+
                                            |
                                 Lambda Proxy Integration
                                            |
+-------------------------------------------v-------------------------------------------+
|                                Compute Engine (AWS Lambda)                            |
|                                                                                       |
|   AWS Lambda Execution Environment                                                    |
|   - Runtime: Python 3.12 (LightGBM + NumPy + SciPy Lambda Layer; x86_64)              |
|   - 2048 MB Memory Profile, 60s Execution Timeout (30s API Gateway cap)               |
|   - Trees need no feature scaling (monotonic splitting handles raw scales natively)   |
|   - Class-weighted LightGBM gradient boosting (scale_pos_weight = N_neg / N_pos)      |
|   - Early stopping (30 rounds) on held-out 20% validation slice + 12s wall-clock guard |
|   - Dynamic operational threshold derivation from chronological train validation slice|
|   - Exact rank-based ROC-AUC (Mann-Whitney U) & Average Precision PR-AUC evaluation   |
|   - Compact tree model export (nodes, thresholds, leaves) for pure-JS browser scoring |
+-------------------------------------------+-------------------------------------------+
                                            |
                         boto3 PutObject / GetObject (JSON + Native Models)
                                            |
+-------------------------------------------v-------------------------------------------+
|                                 Storage Layer (Amazon S3)                             |
|                                                                                       |
|   Amazon S3 Artifact Storage (`fraud-detection-data-anish`)                           |
|   - /models/latest.json                 (Active compact browser model: trees & th)    |
|   - /models/latest.txt                  (Native LightGBM text model string)           |
|   - /models/model-<ISO-TIMESTAMP>.json  (Versioned JSON model checkpoints)            |
|   - /models/model-<ISO-TIMESTAMP>.txt   (Versioned native LightGBM checkpoints)       |
|   - /artifacts/metrics.json             (Out-of-time test partition evaluation)       |
|   - /pipeline_status.json               (Operational health, timings, audit telemetry)|
+---------------------------------------------------------------------------------------+
```

---

## Technical Specifications & Environment

* **Cloud Region:** Asia Pacific Sydney (`ap-southeast-2`)
* **Compute Engine:** AWS Lambda (`python3.12`, 2048 MB memory profile, 60s timeout, capped by 30s API Gateway HTTP integration timeout)
* **Ingress API:** Amazon API Gateway HTTP API v2 with regional proxy integration
* **Frontend Delivery:** Amazon S3 + CloudFront
* **Model Artifact Storage:** Amazon Simple Storage Service (S3) bucket (`fraud-detection-data-anish`)
* **Numeric Dependencies:** LightGBM 4.7+, NumPy 1.26+, SciPy (packaged via AWS Lambda Layer; see [`deploy/LAYER.md`](deploy/LAYER.md))

---

## Leak-Free Feature Engineering (19 Causal Features)

A fatal flaw in naive fraud architectures is target and lookahead data leakage: history aggregates and label references bleed future information into past rows. In this architecture, **every feature is strictly causal**: for any transaction at time $t$, statistics are computed **only** from transactions strictly preceding it ($t' < t$). Future transactions and ground truth labels are strictly isolated.

| Index | Feature Name | Computation Logic | Causal Isolation Mechanism |
| :--- | :--- | :--- | :--- |
| **0** | `amount_log1p` | $\ln(1 + \text{amount})$ | Current transaction amount only |
| **1** | `balance_orig_diff` | $\text{oldbalanceOrg} - \text{newbalanceOrig}$ | Current origin ledger delta |
| **2** | `balance_dest_diff` | $\text{newbalanceDest} - \text{oldbalanceDest}$ | Current destination ledger delta |
| **3** | `orig_balance_error` | $1$ if $\|\text{expected\_newO} - \text{newbalanceOrig}\| > 0.05$ else $0$, where $\text{expected\_newO} = (\text{oldbalanceOrg} + \text{amount})$ if $\text{CASH\_IN}$ else $\max(0, \text{oldbalanceOrg} - \text{amount})$ | Direction-aware ledger consistency audit |
| **4** | `dest_balance_error` | $1$ if $\|\text{expected\_newD} - \text{newbalanceDest}\| > 0.05$ else $0$, where $\text{expected\_newD} = \max(0, \text{oldbalanceDest} - \text{amount})$ if $\text{CASH\_IN}$ else $(\text{oldbalanceDest} + \text{amount})$ | Destination ledger continuity verification |
| **5** | `orig_drained` | $1$ if $\text{oldbalanceOrg} > 0 \land \text{newbalanceOrig} == 0$ else $0$ | Account liquidation flag |
| **6** | `hour_of_day` | $\text{step} \pmod{24}$ | Diurnal temporal risk cycle |
| **7** | `type_TRANSFER` | One-hot binary indicator | Channel risk indicator |
| **8** | `type_CASH_OUT` | One-hot binary indicator | High-velocity withdrawal channel |
| **9** | `type_PAYMENT` | One-hot binary indicator | Low-risk retail channel |
| **10** | `type_DEBIT` | One-hot binary indicator | Direct debit retail channel |
| **11** | `type_CASH_IN` | One-hot binary indicator | Inbound deposit channel |
| **12** | `orig_prior_count` | Number of prior transactions by this `nameOrig` | Running history store ($t' < t$) |
| **13** | `orig_prior_mean_amount` | Cumulative mean of prior amounts from `nameOrig` | Running history store ($t' < t$) |
| **14** | `amount_vs_orig_mean` | $\text{amount} / \text{orig\_prior\_mean\_amount}$ | Relative deviation ratio from account baseline |
| **15** | `dest_prior_count` | Number of prior transactions received by `nameDest` | Destination history store ($t' < t$) |
| **16** | `dest_prior_max_amount` | Maximum single transaction previously received by `nameDest` | Destination history store ($t' < t$) |
| **17** | `is_new_dest` | $1$ if $\text{dest\_prior\_count} == 0$ else $0$ | First-time payee indicator |
| **18** | `orig_txn_last_24_steps` | Count of transactions from `nameOrig` in last 24 steps | Temporal sliding window ($t - 24 < t' < t$) |

> [!IMPORTANT]
> Raw fields `isFraud`, `nameOrig`, `nameDest`, and `step` are **never** passed into the feature matrix. Feature history stores are updated strictly **after** feature extraction for the current row.

---

## Imbalance Handling, Partitioning, and Dynamic Thresholding

### 1. Chronological Out-of-Time Splitting
Financial fraud models cannot use random cross-validation because shuffling leaks future transaction patterns into past evaluations. The dataset is stably sorted by `step` and partitioned chronologically:
* **Default Split:** 70% Training Partition (historical steps), 30% Evaluation Partition (out-of-time future steps).
* **Validation Slice:** The final 20% of the training partition is held out for early stopping and threshold calibration. The first 80% is fitted.

### 2. Class Imbalance Mitigation
Fraud accounts for approximately $1.0\% - 1.3\%$ of transactions in realistic distributions. To prevent model collapse:
* **Gradient-Boosted Class Weighting:** `scale_pos_weight = N_neg / N_pos` computed strictly on the fitted 80% slice. This scales positive-class gradients during tree building, penalizing missed frauds proportionately to prevalence. Note that while this preserves ranking capability (ROC-AUC / PR-AUC), it produces uncalibrated probability scores that require dynamic operational threshold calibration.
* **Payload Transmission Guard:** When payload sizes approach serverless constraints (Amazon API Gateway HTTP API v2 has a **10 MB** payload limit, and AWS Lambda synchronous request/response invocations have a **6 MB** payload limit), majority-class undersampling may be applied **strictly to the training fit slice**. The **validation slice and test partition are never resampled** to guarantee honest out-of-sample evaluation.

### 3. Dynamic Threshold Derivation (Single Source of Truth)
Fixed default thresholds ($p \ge 0.5$) fail in banking operations and are invalid for uncalibrated `scale_pos_weight` scores. Thresholds are derived mathematically by the Lambda trainer from the **validation slice (the held-out 20% of the training partition)** via a 0.01 grid search ($T \in [0.05, 0.95]$, 91 cuts):
1. **Decision Threshold ($T_{\text{dec}}$):** The threshold maximizing the $F_1$ score on the validation slice:
   $$T_{\text{dec}} = \arg\max_{T \in [0.05, 0.95]} F_1(T)$$
2. **Review Threshold ($T_{\text{rev}}$):** The highest cut capturing $\ge 90\%$ recall on the validation slice for secondary human triage (defaults to $0.05$ if $90\%$ recall is unreachable).
3. **Block Threshold ($T_{\text{block}}$):** The lowest cut $\ge T_{\text{dec}}$ achieving precision $\ge 85\%$ with at least 3 true positives ($\text{TP} \ge 3$) for automated transaction cancellation. If no cut meets this condition, the trainer selects the highest-precision available cut with $\text{TP} \ge 3$ and reports `block_target_met: false`.
4. **Ordering Invariant & Diagnostics:** The trainer strictly guarantees $T_{\text{rev}} \le T_{\text{dec}} \le T_{\text{block}}$. When the validation slice contains fewer than 30 positive fraud cases ($N_{\text{pos}} < 30$), the API issues a diagnostic warning noting high sampling variance.

---

## Machine Learning Formulation: LightGBM Gradient-Boosted Trees

### Why Gradient-Boosted Trees Fit Tabular Financial Fraud Data
1. **Invariance to Feature Scales:** Tree split algorithms find cut points ($x_j \le \theta$) using rank orderings. Standard scaling, min-max normalisation, and outlier clipping are mathematically unnecessary.
2. **Non-Linear Interactions:** Trees capture multi-way non-linear interactions natively (e.g., *account drained* **AND** *hour between 1 AM and 5 AM* **AND** *new destination account*) without explicit polynomial feature generation.
3. **Robustness to Extreme Skew:** Transaction amounts follow heavy-tailed Pareto distributions where linear models struggle with extreme leverage points; tree split search is invariant to monotonic transformations.

### Mathematical Formulation
LightGBM builds an additive ensemble of $K$ decision trees:
$$\hat{y}_i = \sum_{k=1}^K f_k(\mathbf{x}_i), \quad f_k \in \mathcal{F}$$
At iteration $t$, the objective function minimizes the second-order Taylor expansion of weighted binary logloss:
$$\mathcal{L}^{(t)} \approx \sum_{i=1}^N \left[ g_i f_t(\mathbf{x}_i) + \frac{1}{2} h_i f_t^2(\mathbf{x}_i) \right] + \Omega(f_t)$$
where $g_i = \frac{\partial \ell}{\partial \hat{y}^{(t-1)}}$ and $h_i = \frac{\partial^2 \ell}{\partial (\hat{y}^{(t-1)})^2}$.

### Key Hyperparameters
* `objective`: `"binary"` (logloss with sigmoid link function)
* `learning_rate`: `0.05` (shrinkage rate per boosting step)
* `num_leaves`: `15` (shallow trees limiting tree complexity and preventing overfitting)
* `max_depth`: `6` (structural regularization preventing deep path memorization)
* `min_data_in_leaf`: `50` (prevents isolated leaves on sparse account IDs)
* `feature_fraction`: `0.8` (subsamples 80% of features per tree)
* `bagging_fraction`: `0.8`, `bagging_freq`: `1` (row subsampling per iteration)
* `lambda_l2`: `1.0` (L2 leaf weight regularization penalty)
* `scale_pos_weight`: $\frac{N_{\text{neg}}}{N_{\text{pos}}}$ (loss multiplier for fraud class)
* `num_threads`: `2` (optimised for 2 vCPUs on 2048 MB Lambda)

### Operational Safeguards
1. **Early Stopping:** Evaluates validation slice loss every round; stops boosting if `average_precision` does not improve for `30` consecutive iterations.
2. **Wall-Clock Guard Callback:** A custom LightGBM callback raises `EarlyStopException` after 12 seconds of boosting, guaranteeing that training never approaches the 30-second API Gateway integration timeout.

### Feature Importance
Computed using total information gain (`importance_type="gain"`), normalised to non-negative percentages:
$$\text{gain\_pct}_j = \frac{\text{Gain}_j}{\sum_{k} \text{Gain}_k} \times 100\%$$
Returned in descending order for the top 10 most predictive features.

### Tree Ensemble Export for In-Browser Scoring
`booster.dump_model()` is reduced to a compact JSON tree structure:
* Split node: `split_feature` (index), `threshold` (number), `decision_type` (`"<="`), `default_left` (bool), `missing_type`, `left_child`, `right_child`.
* Leaf node: `leaf_value` (floating-point logit contribution).
* **Client Scoring:** Pure JavaScript walks each tree iteratively (`x[split_feature] <= threshold ? left : right`), sums leaf values, and applies the sigmoid function:
  $$p = \sigma\left(\sum_{k=1}^K \text{leaf}_k(\mathbf{x})\right) = \frac{1}{1 + e^{-\sum \text{leaf}_k}}$$
* Base prediction (`boost_from_average`) is automatically folded into the leaves by LightGBM, ensuring complete mathematical parity ($< 10^{-16}$) with native Python predictions.

---

## Empirical Benchmark: Multi-Seed Rigorous Evaluation

Evaluated across **5 independent random seeds** (1, 2, 3, 4, 5) on 50,000 generated transactions per seed (70/30 chronological split, 80/20 natural-rate validation slice; source: `results/benchmark.md` and `results/benchmark.json`):

| Metric | Logistic Regression (Adam) | LightGBM (Gradient-Boosted Trees) | Oracle Baseline (Generative Ceiling) |
| :--- | :--- | :--- | :--- |
| **PR-AUC (Headline)** | **0.1608 ± 0.0366** | **0.2238 ± 0.0244** | **0.2220 ± 0.0203** |
| **95% Bootstrap CI** | [0.1295, 0.2072] | [0.1792, 0.2833] | [0.1783, 0.2728] |
| **ROC-AUC** | 0.9418 ± 0.0124 | 0.9505 ± 0.0122 | 0.9577 ± 0.0114 |
| **Precision** | 16.17% ± 4.79% | 25.63% ± 5.02% | 25.63% ± 7.15% |
| **Recall (Detection Rate)** | 44.46% ± 7.47% | 37.29% ± 3.25% | 51.63% ± 17.48% |
| **F1 Score** | 23.03% ± 4.26% | 30.11% ± 4.25% | 32.07% ± 3.91% |
| **Median Training Latency** | **1777.36 ms** | **155.00 ms** (**11.5x faster**) | N/A (Ground Truth) |

### Key Benchmark Takeaways
1. **LightGBM Superiority**: LightGBM achieves a headline PR-AUC of **0.2238** compared to **0.1608** for logistic regression, representing a **+39.2% relative lift**.
2. **Close to the Oracle Ceiling**: The oracle scores transactions with the generator's own hand-coded risk formula, giving **0.2220 PR-AUC**. LightGBM's **0.2238** is statistically indistinguishable from it (the bootstrap intervals overlap heavily), which indicates the model recovers most of the signal the generator contains. Because the labels come from our own formula, this shows the pipeline works, not that the model would reach the same performance on real fraud.
3. **Training Latency**: LightGBM trains in **155.00 ms** median time compared to **1777.36 ms** for iterative Adam logistic regression (**11.5x faster**), well within AWS Lambda execution constraints.

### Test Partition Comparison (`sample_payload.json`, seed 20261006)
Evaluated on the exact same chronological test partition ($14,976$ records, $174$ confirmed frauds, $1.16\%$ prevalence; source: `CHANGELOG.md`):

| Metric | Logistic Regression (L2 Adam) | LightGBM (Gradient-Boosted Trees) | Operational Significance |
| :--- | :--- | :--- | :--- |
| **Model Architecture** | Linear Hyperplane + StandardScaler | **Tree Ensemble (No Scaler Needed)** | Trees capture non-linear fraud ring interactions |
| **Scaler Dependency** | Required (`fit_standard_scaler`) | **NONE (Scale-Invariant)** | Eliminates train-test scaler mismatch risk |
| **Ensemble Complexity** | 19 weights + 1 bias ($32.95 \text{ KB}$) | **105 Trees ($278.94 \text{ KB}$)** | Compact JSON payload, instant browser parsing |
| **Decision Threshold ($T_{\text{dec}}$)** | $0.87$ (F1 cut) | **$0.91$ (F1 cut)** | Calibrated directly to validation distribution |
| **Precision** | $18.23\%$ | **$25.15\%$** | $+38.0\%$ relative precision improvement |
| **Recall (Detection Rate)** | $43.68\%$ ($76/174$ detected) | **$48.28\%$ ($84/174$ detected)** | $+8$ additional frauds blocked ($76 \to 84$) |
| **F1 Score** | $25.72\%$ | **$33.07\%$** | $+28.6\%$ relative F1 improvement |
| **Accuracy** | $97.07\%$ | **$97.73\%$** | Legitimate transactions cleared reliably |
| **ROC-AUC** | $0.9420$ | **$0.9430$** | High discriminatory power without overfitting |
| **PR-AUC (Headline Metric)** | $0.1641$ | **$0.2577$** | **$+57.0\%$ relative PR-AUC lift ($22\times$ random)** |
| **Training Latency** | $2048.28 \text{ ms}$ | **$236.68 \text{ ms}$** | Fast training in Lambda |
| **Test Confusion Matrix** | $76\text{ TP} / 341\text{ FP} / 98\text{ FN} / 14461\text{ TN}$ | **$84\text{ TP} / 250\text{ FP} / 90\text{ FN} / 14552\text{ TN}$** | $+8$ frauds blocked, $-91$ false alarms |



---

## Real Datasets & Cross-Domain Evaluation

The platform includes a **Real dataset mode** alongside the synthetic transaction simulator. This mode accepts datasets conforming to ONE standard CSV format across diverse domains (credit card transactions, cryptocurrency graphs, ransomware extortion, and banking credit default).

### Raw Dataset Sources & Directory Placement

To evaluate real datasets, download the raw data files from their canonical sources and place them into their respective subfolders under `dataset/`:

| Dataset | Canonical Source | Raw File Name(s) | Destination Path |
| :--- | :--- | :--- | :--- |
| **Credit Card Fraud** | [Kaggle: Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) | `creditcard.csv` | `dataset/creditcard/creditcard.csv` |
| **Elliptic Bitcoin** | [Kaggle: Elliptic Data Set](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set) | `elliptic_txs_features.csv`, `elliptic_txs_classes.csv` | `dataset/elliptic/elliptic_txs_features.csv`<br>`dataset/elliptic/elliptic_txs_classes.csv` |
| **Ethereum Fraud** | [Kaggle: Ethereum Fraud Detection Dataset](https://www.kaggle.com/datasets/vagifa/ethereum-frauddetection-dataset) | `transaction_dataset.csv` | `dataset/ethereum/transaction_dataset.csv` |
| **BitcoinHeist Ransomware** | [UCI ML Repository: BitcoinHeist](https://archive.ics.uci.edu/dataset/526/bitcoinheistransomwaredataset) | `BitcoinHeistData.csv` | `dataset/bitcoinheist/BitcoinHeistData.csv` |
| **Taiwan Credit Default** | [UCI ML Repository: Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients) | `UCI_Credit_Card.csv` | `dataset/taiwan/UCI_Credit_Card.csv` |

### Data Preparation Commands

Each raw dataset is standardized into the platform's single uniform CSV representation via its dedicated preparation script:

```bash
python scripts/prep_creditcard.py
python scripts/prep_elliptic.py
python scripts/prep_ethereum.py
python scripts/prep_bitcoinheist.py
python scripts/prep_taiwan.py
```

Each script produces `<name>_std.csv` in `dataset/<name>/` and writes a stratified 300-row fixture with $\ge 30$ positives to `tests/fixtures/real/<name>_sample300.csv`.

### Standard CSV Format Specification

The Real dataset mode intentionally enforces ONE standard schema:
1. **Header Row:** Must be present as the first line.
2. **`label` Column:** Integer binary target ($0$ = non-fraud / licit / non-default, $1$ = fraud / illicit / default).
3. **Optional `order` Column:** Monotonically increasing numeric timestamp, step, or chronological index for time-aware splitting.
4. **Feature Columns:** All remaining columns must be numeric features. Non-numeric columns are detected and excluded at Stage 1.
5. **Missing Values:** Blank cells (`,,`) represent missing values and are parsed as IEEE 754 `NaN`. No ad-hoc string sentinel values are used.

### Pipeline Design for Real Datasets

* **Row Cap:** Enforced at **300,000 rows**. Incremental streaming CSV parsing (`stream.getReader()` + `TextDecoder`) halts at exactly 300,000 rows via `reader.cancel()` with an explicit on-screen notice.
* **Split Rules:** 70% train partition / 30% held-out test partition. The 70% train partition is subdivided into 80% fit slice / 20% validation slice. Datasets with an `order` column (`creditcard`, `elliptic`, `bitcoinheist`) use a strictly **chronological split** preserving time causality. Datasets without an order column (`ethereum`, `taiwan`) use a deterministic **seeded random split** (`mulberry32(42)`).
* **Class-Balanced Fit Slice:** Due to severe class imbalance, all positive instances in the fit partition are retained, and negative instances are undersampled 1:1 using seeded shuffle (`mulberry32(42)`). Because the fit slice is class-balanced (50/50 prior), raw tree logit outputs do not reflect population base rates and predicted probabilities are **not calibrated** posterior probabilities.
* **Threshold Derivation from Natural Validation Slice:** To counteract fit-slice uncalibration, triage thresholds ($T_{\text{dec}}, T_{\text{rev}}, T_{\text{block}}$) are calibrated empirically on the natural-prevalence validation slice using `derive_thresholds`.
* **Validation Subsampling Under 4.5 MB Budget:** HTTP API / Lambda synchronous payload limits require serialized POST `/train` payloads to remain strictly under **4.5 MB**. If the natural validation slice pushes the JSON payload over 4.5 MB, it is uniformly subsampled (seeded `mulberry32`, preserving natural class prevalence) to fit the budget. If the fit slice alone exceeds 4.5 MB (observed with wide datasets such as Elliptic with 165 features and ~2,500 fit frauds = 8.64 MB), the system safely aborts transmission with a clear diagnostic error rather than exceeding API limits.
* **Test Slice Never Sent:** The 30% test slice is strictly isolated in browser memory and is **never sent over the wire** to AWS Lambda.
* **PR-AUC as Headline Metric:** Under extreme class imbalance, ROC-AUC gives an overly optimistic picture due to the overwhelming volume of true negatives. Precision-Recall AUC (PR-AUC, step-wise average precision) is the primary headline metric, evaluated alongside ROC-AUC, Precision, Recall, F1, and the test confusion matrix at $T_{\text{dec}}$.

### Measured Cross-Domain Benchmark Results

The following table contains measured evaluation numbers generated directly by `scripts/run_real_eval.js` and recorded in `results/real_eval.json`:

| Dataset | Split Type | Total Rows | Positives | Prevalence | Train Sent (Fit + Val) | Payload | PR-AUC (Headline) | ROC-AUC | Precision | Recall | F1 Score | Test Confusion Matrix (@ $T_{\text{dec}}$) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Credit Card Fraud** (`creditcard`) | Chronological | 284,807 | 492 | 0.17% | 16,415 (712 fit + 15,703 val*) | 4.24 MB | **0.7522** | 0.9782 | 87.10% | 75.00% | 80.60% | TP=81, FP=12, FN=27, TN=85,323 (@ 0.80) |
| **Elliptic Bitcoin** (`elliptic`) | Chronological | 46,564 | 4,545 | 9.76% | Exceeds Budget | 8.64 MB | N/A | N/A | N/A | N/A | N/A | Exceeds 4.5 MB transmission budget (Fit slice alone 8.64 MB for 165 features; safely halted) |
| **Ethereum Fraud** (`ethereum`) | Seeded (42) | 9,841 | 2,179 | 22.14% | 3,747 (2,370 fit + 1,377 val) | 0.54 MB | **0.9953** | 0.9984 | 97.61% | 96.88% | 97.24% | TP=653, FP=16, FN=21, TN=2,263 (@ 0.66) |
| **BitcoinHeist Ransomware** (`bitcoinheist`) | Chronological | 291,670 | 4,131 | 1.42% | 44,765 (3,932 fit + 40,833 val) | 1.36 MB | **0.0115** | 0.5194 | 1.36% | 78.38% | 2.67% | TP=747, FP=54,219, FN=206, TN=32,329 (@ 0.50) |
| **Taiwan Credit Default** (`taiwan`) | Seeded (42) | 30,000 | 6,636 | 22.12% | 11,622 (7,422 fit + 4,200 val) | 1.01 MB | **0.5642** | 0.7811 | 49.00% | 58.52% | 53.34% | TP=1,174, FP=1,222, FN=832, TN=5,772 (@ 0.55) |

*\*Note: For Credit Card Fraud, the validation slice was automatically subsampled from 39,872 rows to 15,703 rows to stay under the 4.5 MB transmission budget.*

---

## API Reference (Amazon API Gateway HTTP API v2)

Regional Base URL: `https://<api-id>.execute-api.ap-southeast-2.amazonaws.com`

### 1. `GET /health`
Returns service availability, deployed version, layer diagnostic status, and active model telemetry.

```json
{
  "status": "healthy",
  "version": "2.0.0",
  "service": "aws-financial-fraud-detection",
  "model_available": true,
  "last_trained_at": "2026-10-22T03:39:14Z",
  "lightgbm_available": true,
  "lightgbm_version": "4.7.0"
}
```

### 2. `POST /train`
Accepts causal feature matrices, fits LightGBM on 80% train slice with early stopping on 20% validation slice, derives operational thresholds, evaluates on test partition, persists JSON and native text models to S3, and returns compact browser trees.

#### Request Payload
```json
{
  "feature_names": ["amount_log1p", "balance_orig_diff", "..."],
  "train": {
    "X": [[9.2103, 1000.0, ...], ...],
    "y": [0, 1, 0, ...]
  },
  "test": {
    "X": [[11.5129, 50000.0, ...], ...],
    "y": [1, 0, 0, ...]
  },
  "options": {
    "num_boost_round": 300,
    "learning_rate": 0.05,
    "num_leaves": 15,
    "max_depth": 6,
    "min_data_in_leaf": 50,
    "seed": 42
  },
  "meta": {
    "split": "70/30",
    "n_raw": 50000,
    "n_clean": 49918
  }
}
```

#### Response Payload (`200 OK`)
```json
{
  "status": "SUCCESS",
  "model_version": "model-2026-10-22T03-39-14Z",
  "model": {
    "trees": [
      {
        "split_feature": 5,
        "threshold": 0.5,
        "decision_type": "<=",
        "default_left": true,
        "missing_type": "None",
        "left_child": { "leaf_value": -1.24 },
        "right_child": { "leaf_value": 2.15 }
      }
    ],
    "feature_names": ["amount_log1p", "balance_orig_diff", "..."],
    "thresholds": {
      "decision": 0.91,
      "review": 0.25,
      "block": 0.95,
      "block_target_met": false
    },
    "model_version": "model-2026-10-22T03-39-14Z",
    "objective": "binary"
  },
  "metrics": {
    "precision": 0.2515,
    "recall": 0.4828,
    "f1": 0.3307,
    "accuracy": 0.9773,
    "roc_auc": 0.9430,
    "pr_auc": 0.2577
  },
  "confusion": { "tp": 84, "fp": 250, "fn": 90, "tn": 14552 },
  "baseline": { "accuracy": 0.9884 },
  "importance": [
    { "feature": "orig_drained", "gain_pct": 77.40 },
    { "feature": "amount_vs_orig_mean", "gain_pct": 5.28 },
    { "feature": "hour_of_day", "gain_pct": 4.77 }
  ],
  "eval_history": [
    { "iteration": 1, "binary_logloss": 0.6512, "average_precision": 0.1845 }
  ],
  "trees_count": 105,
  "best_iteration": 105,
  "params": {
    "objective": "binary",
    "learning_rate": 0.05,
    "num_leaves": 15,
    "max_depth": 6,
    "scale_pos_weight": 71.04
  },
  "timings": { "train_ms": 234.76, "total_ms": 285.12 },
  "persisted": true,
  "s3_errors": []
}
```

### 3. `POST /predict`
Batched scoring for up to 1,000 transaction feature vectors using the native LightGBM booster loaded from `models/latest.txt` on S3 or Lambda memory cache.

#### Request Payload
```json
{
  "features": [
    [10.8197, 50000.0, 50000.0, 0, 0, 1, 14, 1, 0, 0, 0, 0, 3, 12000.0, 4.16, 1, 50000.0, 0, 2]
  ]
}
```

#### Response Payload (`200 OK`)
```json
{
  "status": "SUCCESS",
  "model_version": "model-2026-10-22T03-39-14Z",
  "scores": [0.9624],
  "actions": ["Block"],
  "thresholds": { "decision": 0.91, "review": 0.25, "block": 0.95 }
}
```

### 4. `GET /model`
Fetches the active compact browser model artifact bundle (`models/latest.json`) containing decision trees, feature names, thresholds, and metrics.

---

## Deployment & Lambda Layer Packaging

For full dependency and OpenMP runtime analysis, refer to [`deploy/LAYER.md`](deploy/LAYER.md).

### 1. Build the Lambda Layer
Obtain `libgomp.so.1` and build the layer package for Python 3.12 / x86_64:
```bash
./deploy/get_libgomp.sh
./deploy/build_layer.sh
```

### 2. Deploy Backend via AWS SAM
AWS SAM automatically packages and uploads the layer directly from `ContentUri: ../build/layer/lightgbm-layer.zip` during deployment:
```bash
sam build -t deploy/template.yaml
sam deploy --guided \
  --stack-name fraud-detection-platform \
  --region ap-southeast-2
```

---

## Running Automated Verification & Unit Tests

### 1. Python Unit Tests (Pytest)
Validates hand-computed metrics, LightGBM convergence, tree export parity ($< 10^{-9}$), wall-clock guard early stop, threshold isolation to validation slice, mocked S3 roundtrip, and layer failure handling:
```bash
pytest backend/tests/test_model.py -v
```

### 2. End-to-End Browser Tree Scorer Parity Test (Node.js)
Extracts the real `scoreRow` function from `frontend/index.html`, runs it against 1,000 test predictions from Python LightGBM, and verifies exact mathematical agreement:
```bash
node backend/tests/verify_parity.js
```

### 3. Real Generator Class Prevalence Audit (Node.js)
Extracts `generate()` directly from `frontend/index.html` and asserts that fraud rate on 50,000 records is within the realistic $[0.8\%, 2.0\%]$ interval:
```bash
node backend/tests/test_generator.js
```

### 4. Causal Feature Independence Test (Node.js)
Proves zero lookahead leakage by mutating future rows and verifying past feature invariance:
```bash
node backend/tests/test_causal.js
```

### 5. Benchmark Evaluation Script (Python)
Evaluates Logistic Regression vs LightGBM on the exact same dataset partition:
```bash
python backend/tests/compare_before_after.py
```

### 6. Full Headless Browser End-to-End Verification (Microsoft Edge)
Launches headless Edge, mocks the cloud backend, executes all 7 stages in Manual mode, tests the parity verification button, executes Stage 7 across 50,000 rows, tests the live Dashboard stream, tests Auto mode across all 7 stages, captures screenshots, and asserts 0 console errors:
```bash
node backend/tests/verify_browser.js
```

---

## System Limitations (Academic & Operational Honesty)

1. **Synthetic Labels From Hand-Coded Formula (Generative Ceiling):**
   * *Mechanism:* Ground-truth fraud labels in the synthetic transaction feed are generated from a hand-coded logistic risk formula (transaction type, account draining, high amount vs account mean, nocturnal hours, and new payees) perturbed by logistic noise.
   * *Limitation:* The Oracle Baseline (`results/benchmark.md`) evaluates transactions directly through the generator's exact hand-coded logit function, establishing a theoretical noise ceiling of **0.2220 ± 0.0203 PR-AUC** (95% Bootstrap CI: [0.1783, 0.2728]). LightGBM achieves **0.2238 ± 0.0244 PR-AUC**, indicating that it recovers most of the signal embedded in the synthetic data generator. Performance cannot exceed this ceiling without changing the underlying generative process.
2. **Post-Transaction Balance Features & PaySim Ablation:**
   * *Mechanism:* Features 1 through 5 (`balance_orig_diff`, `balance_dest_diff`, `orig_balance_error`, `dest_balance_error`, `orig_drained`) rely on post-transaction balance fields (`newbalanceOrig`, `newbalanceDest`).
   * *Limitation:* In real-time authorization systems, post-transaction ledger states may not yet be settled at the exact millisecond of risk scoring. On the Kaggle PaySim dataset (`results/paysim.json`), full 19-feature LightGBM achieved **0.0053 PR-AUC** and **0.7418 ROC-AUC** across 100,000 transactions (72 trees). `eval_paysim.py` also supports a `--drop-post-balance` ablation (removing features 1 to 5), which was not run for this submission and is listed as future work. Note that this 100,000-row sample contains very few frauds (9 in the test slice), so these PaySim numbers are indicative only.
3. **Uncalibrated Probability Scores from `scale_pos_weight`:**
   * *Mechanism:* To handle extreme imbalance ($1.0\% - 1.3\%$ fraud prevalence), LightGBM is trained with `scale_pos_weight = N_neg / N_pos` (e.g. $71.04$).
   * *Limitation:* While this inverse-prevalence weighting scales the positive loss gradients and preserves ranking performance (ROC-AUC and PR-AUC), it systematically shifts raw model output scores away from true posterior probabilities ($P(Y=1|X)$). The resulting sigmoid scores are uncalibrated risk rankings rather than Bayesian probabilities. Consequently, default thresholds ($p \ge 0.5$) are meaningless, necessitating dynamic empirical threshold derivation on the validation slice.
4. **Threshold Sampling Variance with Few Validation Positives:**
   * *Mechanism:* Triage thresholds ($T_{\text{dec}}, T_{\text{rev}}, T_{\text{block}}$) are computed by `derive_thresholds` on the held-out 20% validation slice.
   * *Limitation:* When fraud prevalence is low, the validation slice contains very few confirmed fraud instances (e.g. only 7 positive fraud cases in the 14,000-row PaySim validation set; ~25–35 in synthetic validation slices). With $N_{\text{pos}} < 30$, precision and recall at individual threshold cut points exhibit high sampling variance (a single false positive or false negative causes large discrete swings). The system issues explicit diagnostic warnings when $N_{\text{pos}} < 30$ and indicates when the block precision target cannot be met (`block_target_met: false`).
5. **Validation Slice Reuse (Dual Purpose):**
   * *Mechanism:* The final 20% of the training partition is used both for early stopping of LightGBM boosting rounds and for calibrating triage thresholds ($T_{\text{dec}}, T_{\text{rev}}, T_{\text{block}}$).
   * *Limitation:* Reusing the same validation slice introduces a slight optimistic bias into the validation threshold selection. In enterprise financial production, a three-way split (e.g. 60% train / 20% early-stopping validation / 20% threshold-calibration validation / holdout test) is preferred. Here, dual reuse preserves sample efficiency under serverless payload limits (10 MB HTTP API, 6 MB Lambda) while keeping the test set 100% isolated.
6. **Fallback Algorithm Duality:** If the cloud Lambda is unreachable or the LightGBM layer is missing, the frontend falls back to an in-browser logistic regression trainer. The UI clearly labels this with a warning banner (`"Fallback: logistic regression (cloud model unavailable)"`).
7. **Fixed Single-CSV Schema vs. Arbitrary Schema Support (Future Work):**
   * *Mechanism:* The Real dataset mode intentionally enforces ONE standard CSV schema (header row, binary 0/1 `label`, optional ascending `order`, and numeric feature columns with blank cells as missing `NaN`).
   * *Limitation:* The platform deliberately avoids complex automatic schema inference (such as automatic string categorical encoding, multi-format timestamp parsing, unstructured text embeddings, or multi-table join resolution). Any dataset with text or non-numeric metadata must be preprocessed prior to upload into this standard representation (as implemented in `scripts/prep_<name>.py`). Providing full arbitrary-schema auto-ingestion, automatic categorical one-hot encoding, and feature extraction for unstandardized CSV files is left as future work.

---