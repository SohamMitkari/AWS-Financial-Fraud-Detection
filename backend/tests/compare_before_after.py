"""
Benchmark Comparison: Logistic Regression vs LightGBM Gradient-Boosted Trees
Evaluated on the exact same chronological dataset and test partition.
No AWS credentials required; S3 is mocked.
"""

import sys
import os
import json
import math
import time
import random
from unittest.mock import MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import lambda_function as lf

# Load exported sample payload
payload_path = os.path.join(os.path.dirname(__file__), "sample_payload.json")
with open(payload_path, "r") as f:
    payload = json.load(f)

X_train = payload["train"]["X"]
y_train = payload["train"]["y"]
X_test = payload["test"]["X"]
y_test = payload["test"]["y"]
feature_names = payload["feature_names"]
n_test = len(y_test)
pos_test = sum(y_test)
neg_test = n_test - pos_test

# ---------------------------------------------------------------------------
# 1. PURE-PYTHON LOGISTIC REGRESSION (Previous Baseline Model)
# ---------------------------------------------------------------------------
def fit_standard_scaler(X):
    n = len(X)
    d = len(X[0])
    means = [sum(X[i][j] for i in range(n)) / n for j in range(d)]
    variances = [sum((X[i][j] - means[j]) ** 2 for i in range(n)) / n for j in range(d)]
    stds = [math.sqrt(v) if v > 1e-8 else 1.0 for v in variances]
    return {"mean": means, "std": stds}

def transform_scaler(X, scaler):
    means, stds = scaler["mean"], scaler["std"]
    d = len(means)
    return [[(row[j] - means[j]) / stds[j] for j in range(d)] for row in X]

def train_logistic_regression(X, y, epochs=15, lr=0.015, l2=1e-4, seed=42, batch_size=256):
    rng = random.Random(seed)
    n = len(X)
    d = len(X[0])
    pos_c = sum(y)
    neg_c = n - pos_c
    w_pos = n / (2.0 * max(1, pos_c))
    w_neg = n / (2.0 * max(1, neg_c))
    w = [0.0] * d
    b = 0.0
    m_w, v_w = [0.0] * d, [0.0] * d
    m_b, v_b = 0.0, 0.0
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    indices = list(range(n))
    t_step = 0
    for epoch in range(1, epochs + 1):
        rng.shuffle(indices)
        for start in range(0, n, batch_size):
            end = min(n, start + batch_size)
            grad_w = [0.0] * d
            grad_b = 0.0
            batch_weight = 0.0
            for idx in range(start, end):
                i = indices[idx]
                sw = w_pos if y[i] == 1 else w_neg
                batch_weight += sw
                z = sum(w[j] * X[i][j] for j in range(d)) + b
                p = lf.sigmoid(z)
                err = sw * (p - y[i])
                grad_b += err
                for j in range(d):
                    grad_w[j] += err * X[i][j]
            t_step += 1
            inv_bw = 1.0 / max(1e-8, batch_weight)
            gb = grad_b * inv_bw
            m_b = beta1 * m_b + (1 - beta1) * gb
            v_b = beta2 * v_b + (1 - beta2) * (gb * gb)
            m_b_hat = m_b / (1 - beta1 ** t_step)
            v_b_hat = v_b / (1 - beta2 ** t_step)
            b -= lr * m_b_hat / (math.sqrt(v_b_hat) + eps)
            for j in range(d):
                gw = grad_w[j] * inv_bw + l2 * w[j]
                m_w[j] = beta1 * m_w[j] + (1 - beta1) * gw
                v_w[j] = beta2 * v_w[j] + (1 - beta2) * (gw * gw)
                m_hat = m_w[j] / (1 - beta1 ** t_step)
                v_hat = v_w[j] / (1 - beta2 ** t_step)
                w[j] -= lr * m_hat / (math.sqrt(v_hat) + eps)
    return w, b

# Run Logistic Regression pipeline
t_lr_start = time.perf_counter()
scaler = fit_standard_scaler(X_train)
X_tr_s = transform_scaler(X_train, scaler)
X_te_s = transform_scaler(X_test, scaler)

val_cut = int(len(X_tr_s) * 0.8)
sub_X = X_tr_s[:val_cut]
sub_y = y_train[:val_cut]
val_X = X_tr_s[val_cut:]
val_y = y_train[val_cut:]

w, b = train_logistic_regression(sub_X, sub_y, epochs=15, lr=0.015, l2=1e-4, seed=42)
t_lr_end = time.perf_counter()
lr_train_ms = round((t_lr_end - t_lr_start) * 1000, 2)

d_feat = len(w)
val_scores_lr = [lf.sigmoid(sum(w[j] * val_X[i][j] for j in range(d_feat)) + b) for i in range(len(val_X))]
th_lr = lf.derive_thresholds(val_y, val_scores_lr)

test_scores_lr = [lf.sigmoid(sum(w[j] * X_te_s[i][j] for j in range(d_feat)) + b) for i in range(n_test)]
lr_eval = lf.evaluate_predictions(y_test, test_scores_lr, th_lr["decision"])
lr_roc = lf.compute_roc_auc(y_test, test_scores_lr)
lr_pr = lf.compute_pr_auc(y_test, test_scores_lr)

lr_model_bundle = {
    "weights": w, "bias": b, "scaler": scaler, "thresholds": th_lr
}
lr_size_kb = round(len(json.dumps(lr_model_bundle)) / 1024, 2)


# ---------------------------------------------------------------------------
# 2. LIGHTGBM GRADIENT-BOOSTED TREES (New Production Model)
# ---------------------------------------------------------------------------
# Mock S3 and API Key for lambda_handler
os.environ["TRAIN_API_KEY"] = "benchmark-secret-key-16-chars"
mock_s3 = MagicMock()
mock_s3.put_object.return_value = {"ResponseMetadata": {"HTTPStatusCode": 200}}
lf.s3_client = mock_s3

train_event = {
    "requestContext": {"http": {"method": "POST"}},
    "rawPath": "/train",
    "headers": {"x-api-key": "benchmark-secret-key-16-chars"},
    "body": json.dumps({
        "feature_names": feature_names,
        "train": {"X": X_train, "y": y_train},
        "test": {"X": X_test, "y": y_test},
        "options": {
            "num_boost_round": 300,
            "learning_rate": 0.05,
            "num_leaves": 15,
            "max_depth": 6,
            "min_data_in_leaf": 50,
            "seed": 42
        }
    })
}

resp = lf.lambda_handler(train_event)
assert resp["statusCode"] == 200, f"Lambda train failed: {resp['body']}"
lgb_response = json.loads(resp["body"])

lgb_train_ms = lgb_response["timings"]["train_ms"]
lgb_model = lgb_response["model"]
compact_trees = lgb_model["trees"]
th_lgb = lgb_model["thresholds"]
lgb_size_kb = round(len(json.dumps(lgb_model)) / 1024, 2)

# Score test set using pure tree walker (mirroring browser JS scoreRow)
def score_tree_walker(node, x):
    while "leaf_value" not in node:
        val = x[node["split_feature"]]
        if val is None or math.isnan(val):
            node = node["left_child"] if node.get("default_left", True) else node["right_child"]
        elif val <= node["threshold"]:
            node = node["left_child"]
        else:
            node = node["right_child"]
    return node["leaf_value"]

def score_row_walker(trees, x):
    raw = sum(score_tree_walker(t, x) for t in trees)
    return lf.sigmoid(raw)

test_scores_lgb = [score_row_walker(compact_trees, row) for row in X_test]
lgb_eval = lf.evaluate_predictions(y_test, test_scores_lgb, th_lgb["decision"])
lgb_roc = lf.compute_roc_auc(y_test, test_scores_lgb)
lgb_pr = lf.compute_pr_auc(y_test, test_scores_lgb)

base_acc = (n_test - pos_test) / n_test
lr_cm = f"{lr_eval['confusion']['tp']} TP / {lr_eval['confusion']['fp']} FP / {lr_eval['confusion']['fn']} FN / {lr_eval['confusion']['tn']} TN"
lgb_cm = f"{lgb_eval['confusion']['tp']} TP / {lgb_eval['confusion']['fp']} FP / {lgb_eval['confusion']['fn']} FN / {lgb_eval['confusion']['tn']} TN"

# ---------------------------------------------------------------------------
# 3. PRINT COMPARISON TABLE
# ---------------------------------------------------------------------------
print("=" * 86)
print(f"BENCHMARK COMPARISON ON TEST PARTITION ({n_test:,} records, {pos_test} frauds, prevalence: {pos_test/n_test*100:.2f}%)")
print("=" * 86)
print(f"{'Metric':<32} | {'Logistic Regression':<23} | {'LightGBM (Trees)':<23}")
print("-" * 86)
print(f"{'Model Architecture':<32} | {'L2 Logistic (Adam)':<23} | {'Gradient-Boosted Trees':<23}")
print(f"{'Data Scaler Required':<32} | {'YES (StandardScaler)':<23} | {'NONE (Invariant)':<23}")
tree_str = f"{lgb_response['trees_count']} trees (best {lgb_response['best_iteration']})"
print(f"{'Number of Trees':<32} | {'N/A (Linear weights)':<23} | {tree_str:<23}")
print(f"{'Decision Threshold (F1 cut)':<32} | {th_lr['decision']:<23.2f} | {th_lgb['decision']:<23.2f}")
print(f"{'Precision':<32} | {lr_eval['precision']*100:<22.2f}% | {lgb_eval['precision']*100:<22.2f}%")
print(f"{'Recall (Detection Rate)':<32} | {lr_eval['recall']*100:<22.2f}% | {lgb_eval['recall']*100:<22.2f}%")
print(f"{'F1 Score':<32} | {lr_eval['f1']*100:<22.2f}% | {lgb_eval['f1']*100:<22.2f}%")
print(f"{'Accuracy':<32} | {lr_eval['accuracy']*100:<22.2f}% | {lgb_eval['accuracy']*100:<22.2f}%")
print(f"{'ROC-AUC':<32} | {lr_roc:<23.4f} | {lgb_roc:<23.4f}")
print(f"{'PR-AUC (Headline)':<32} | {lr_pr:<23.4f} | {lgb_pr:<23.4f}")
print(f"{'Baseline Accuracy':<32} | {base_acc*100:<22.2f}% | {base_acc*100:<22.2f}%")
print(f"{'Training Latency':<32} | {f'{lr_train_ms} ms':<23} | {f'{lgb_train_ms} ms':<23}")
print(f"{'Model Artifact Size':<32} | {f'{lr_size_kb} KB':<23} | {f'{lgb_size_kb} KB':<23}")
print(f"{'Confusion Matrix':<32} | {lr_cm:<23} | {lgb_cm:<23}")
print("=" * 86)
