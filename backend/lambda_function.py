"""
Serverless Financial Fraud Detection Platform - AWS Lambda Backend
Runtime: Python 3.12 (LightGBM + NumPy layer + boto3 from runtime)
Gradient-Boosted Decision Trees for Tabular Financial Fraud Detection
"""

import json
import os
import math
import time
import datetime
import logging
import hmac

# Configure structured logging
logger = logging.getLogger("fraud_detection_lambda")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        json.dumps({
            "timestamp": "%(asctime)s",
            "level": "%(levelname)s",
            "message": "%(message)s"
        })
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# Optional boto3 client
try:
    import boto3
    s3_client = boto3.client("s3")
except Exception as s3_init_err:
    s3_client = None
    logger.warning(f"boto3 initialization warning: {s3_init_err}")

# Environment variables
DATA_BUCKET_NAME = os.environ.get("DATA_BUCKET_NAME", "fraud-detection-data-anish-syd")
ALLOWED_ORIGIN = os.environ.get("ALLOWED_ORIGIN", "*")

# In-memory model cache across warm Lambda invocations
_CACHED_MODEL = None         # Compact browser-format JSON model bundle
_CACHED_BOOSTER = None       # Native LightGBM Booster instance
_CACHED_VERSION = None       # Model version identifier


def check_lightgbm():
    """Checks whether LightGBM and NumPy are available in the Lambda environment."""
    try:
        import numpy as np  # noqa: F401
        import lightgbm as lgb
        return True, lgb.__version__, None
    except Exception as e:
        return False, None, str(e)


def get_cors_headers():
    return {
        "Content-Type": "application/json",
        "Access-Control-Allow-Origin": ALLOWED_ORIGIN,
        "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type, Authorization, X-Requested-With, Accept, x-api-key",
        "Access-Control-Max-Age": "86400"
    }


def json_response(status_code, body_dict):
    return {
        "statusCode": status_code,
        "headers": get_cors_headers(),
        "body": json.dumps(body_dict)
    }


def sigmoid(z):
    """Numerically stable sigmoid function."""
    if z < -40.0:
        return 0.0
    if z > 40.0:
        return 1.0
    return 1.0 / (1.0 + math.exp(-z))


def compute_roc_auc(y_true, y_score):
    """
    Computes exact Receiver Operating Characteristic Area Under Curve (ROC-AUC)
    using the rank-based Mann-Whitney U statistic with exact tie corrections.
    """
    n = len(y_true)
    if n == 0:
        return 0.5
    pos_count = sum(y_true)
    neg_count = n - pos_count
    if pos_count == 0 or neg_count == 0:
        return 0.5

    paired = sorted(zip(y_score, y_true), key=lambda x: x[0])

    rank_sum_pos = 0.0
    i = 0
    while i < n:
        j = i
        while j < n - 1 and paired[j + 1][0] == paired[j][0]:
            j += 1
        avg_rank = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            if paired[k][1] == 1:
                rank_sum_pos += avg_rank
        i = j + 1

    u = rank_sum_pos - (pos_count * (pos_count + 1)) / 2.0
    auc = u / (pos_count * neg_count)
    return max(0.0, min(1.0, auc))


def compute_pr_auc(y_true, y_score):
    """
    Computes Precision-Recall Area Under Curve (PR-AUC / Average Precision)
    with exact tie handling across unique score thresholds.
    """
    n = len(y_true)
    if n == 0:
        return 0.0
    pos_count = sum(y_true)
    if pos_count == 0:
        return 0.0

    paired = sorted(zip(y_score, y_true), key=lambda x: x[0], reverse=True)

    tp = 0
    fp = 0
    prev_recall = 0.0
    ap = 0.0

    i = 0
    while i < n:
        j = i
        while j < n - 1 and paired[j + 1][0] == paired[j][0]:
            j += 1
        for k in range(i, j + 1):
            if paired[k][1] == 1:
                tp += 1
            else:
                fp += 1
        precision = tp / (tp + fp)
        recall = tp / pos_count
        ap += (recall - prev_recall) * precision
        prev_recall = recall
        i = j + 1

    return max(0.0, min(1.0, ap))


def evaluate_predictions(y_true, y_score, threshold):
    """
    Computes precision, recall, accuracy, F1, and confusion matrix at a decision threshold.
    """
    tp = fp = fn = tn = 0
    n = len(y_true)
    for i in range(n):
        pred = 1 if y_score[i] >= threshold else 0
        actual = y_true[i]
        if pred == 1 and actual == 1:
            tp += 1
        elif pred == 1 and actual == 0:
            fp += 1
        elif pred == 0 and actual == 1:
            fn += 1
        else:
            tn += 1

    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    accuracy = (tp + tn) / max(1, n)
    f1 = 2.0 * precision * recall / max(1e-6, precision + recall)

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "accuracy": round(accuracy, 4),
        "f1": round(f1, 4),
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn}
    }


def derive_thresholds(y_val, y_score_val):
    """
    Derives decision, review, and block thresholds from the validation slice scores:
    - Search thresholds 0.05 to 0.95 in steps of 0.01 (using numpy).
    - decision: F1-maximising cut.
    - review: highest cut with recall >= 0.90.
    - block: lowest cut >= decision with precision >= 0.85 and tp >= 3.
      If no cut meets the block target, uses the highest-precision cut with tp >= 3
      and reports block_target_met=False.
    - Guaranteed ordering invariant: review <= decision <= block.
    - Diagnostics: val_n, val_positives, val_prevalence, and precision/recall at each threshold.
    - Warnings: messages if val_positives < 30 or block_target_met is False.
    """
    import numpy as np

    y_true_arr = np.asarray(y_val, dtype=np.int32)
    y_score_arr = np.asarray(y_score_val, dtype=np.float32)

    val_n = int(len(y_true_arr))
    val_positives = int(np.sum(y_true_arr == 1))
    val_prevalence = float(val_positives / max(1, val_n))

    grid = np.round(np.linspace(0.05, 0.95, 91), 2)
    grid_metrics = []

    for th in grid:
        th_val = round(float(th), 2)
        preds = (y_score_arr >= th_val)
        tp = int(np.sum((preds == 1) & (y_true_arr == 1)))
        fp = int(np.sum((preds == 1) & (y_true_arr == 0)))
        fn = int(np.sum((preds == 0) & (y_true_arr == 1)))
        tn = int(np.sum((preds == 0) & (y_true_arr == 0)))
        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        grid_metrics.append({
            "threshold": th_val,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn
        })

    # 1. Decision threshold: F1-maximising cut
    best_f1 = -1.0
    best_dec_th = 0.50
    for m in grid_metrics:
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            best_dec_th = m["threshold"]

    # 2. Review threshold: highest cut with recall >= 0.90
    recall_cuts = [m["threshold"] for m in grid_metrics if m["recall"] >= 0.90]
    if recall_cuts:
        review_th = max(recall_cuts)
    else:
        review_th = grid_metrics[0]["threshold"]

    # 3. Block threshold: lowest cut >= decision with precision >= 0.85 and tp >= 3
    target_cuts = [
        m for m in grid_metrics
        if m["threshold"] >= best_dec_th and m["precision"] >= 0.85 and m["tp"] >= 3
    ]
    if target_cuts:
        block_th = min(m["threshold"] for m in target_cuts)
        block_target_met = True
    else:
        block_target_met = False
        fallback_cuts = [m for m in grid_metrics if m["tp"] >= 3]
        if fallback_cuts:
            cuts_ge_dec = [m for m in fallback_cuts if m["threshold"] >= best_dec_th]
            cand_pool = cuts_ge_dec if cuts_ge_dec else fallback_cuts
            best_cand = max(cand_pool, key=lambda m: (m["precision"], m["tp"], -m["threshold"]))
            block_th = best_cand["threshold"]
        else:
            block_th = best_dec_th

    # Maintain review <= decision <= block invariant
    review_th = round(min(review_th, best_dec_th), 2)
    best_dec_th = round(float(best_dec_th), 2)
    block_th = round(max(block_th, best_dec_th), 2)

    # Warnings list
    warnings = []
    if val_positives < 30:
        warnings.append(
            f"Validation set has only {val_positives} positive fraud cases (< 30). "
            "Threshold derivation may have high sampling variance."
        )
    if not block_target_met:
        warnings.append(
            f"Block threshold target (precision >= 0.85 with tp >= 3) was not met. "
            f"Selected highest-precision available cut {block_th:.2f} (block_target_met=False)."
        )

    diagnostics = {
        "val_n": val_n,
        "val_positives": val_positives,
        "val_prevalence": round(val_prevalence, 4),
        "thresholds": grid_metrics,
        "precision_recall": [
            {"threshold": m["threshold"], "precision": m["precision"], "recall": m["recall"]}
            for m in grid_metrics
        ]
    }

    return {
        "decision": best_dec_th,
        "review": review_th,
        "block": block_th,
        "block_target_met": block_target_met,
        "threshold_diagnostics": diagnostics,
        "diagnostics": diagnostics,
        "warnings": warnings
    }


def compact_tree_node(node):
    """
    Recursively reduces a LightGBM tree structure to only the necessary fields
    for in-browser scoring: split_feature, threshold, decision_type, default_left,
    missing_type, left_child, right_child, and leaf_value.
    Strips away training statistics (counts, gains, weights).
    """
    if "leaf_value" in node:
        return {
            "leaf_value": float(node["leaf_value"])
        }
    return {
        "split_feature": int(node["split_feature"]),
        "threshold": float(node["threshold"]),
        "decision_type": str(node.get("decision_type", "<=")),
        "default_left": bool(node.get("default_left", True)),
        "missing_type": str(node.get("missing_type", "None")),
        "left_child": compact_tree_node(node["left_child"]),
        "right_child": compact_tree_node(node["right_child"])
    }


def make_time_guard_callback(time_limit_sec=12.0):
    """
    LightGBM callback that enforces a strict wall-clock timeout.
    Safely halts boosting after time_limit_sec without failing the invocation.
    """
    import lightgbm as lgb
    start_time = time.perf_counter()

    def _guard(env):
        if time.perf_counter() - start_time >= time_limit_sec:
            logger.warning(
                f"Training wall-clock guard triggered at iteration {env.iteration} "
                f"({time.perf_counter() - start_time:.2f}s elapsed >= {time_limit_sec}s limit)."
            )
            raise lgb.EarlyStopException(env.iteration, env.evaluation_result_list)

    _guard.order = 100
    return _guard


def upload_to_s3(key, data, content_type="application/json"):
    """Safely attempts to persist payload to S3 without raising unhandled exceptions."""
    if not s3_client:
        return False, "boto3 s3 client not initialized"
    try:
        body = json.dumps(data, indent=2) if content_type == "application/json" else str(data)
        s3_client.put_object(
            Bucket=DATA_BUCKET_NAME,
            Key=key,
            Body=body.encode("utf-8"),
            ContentType=content_type
        )
        return True, None
    except Exception as e:
        logger.warning(f"S3 PutObject failure for {key} in {DATA_BUCKET_NAME}: {e}")
        return False, str(e)


def fit_model(X_train, y_train, options=None, feature_names=None):
    """
    Fits LightGBM gradient-boosted trees on chronological 80/20 train/validation split.
    Returns:
        booster: Trained LightGBM Booster
        X_val: np.ndarray of validation features (20% held-out slice)
        y_val: np.ndarray of validation labels
        scale_pos_weight: float ratio of negative to positive examples in fitted 80%
        actual_params: dict of resolved hyperparameters
        evals_result: dict of recorded evaluations
        train_duration_ms: float training latency in milliseconds
    """
    import numpy as np
    import lightgbm as lgb

    if options is None:
        options = {}
    if feature_names is None and isinstance(options, dict):
        feature_names = options.get("feature_names")

    # Chronological split of train partition:
    # First 80% is fitted; last 20% is held out as validation slice for early stopping and thresholds.
    # If fit_rows is specified in options, use it as the split boundary.
    n_train_total = len(X_train)
    if isinstance(options, dict) and "fit_rows" in options and options["fit_rows"] is not None:
        val_split_idx = int(options["fit_rows"])
    elif isinstance(options, dict) and "val_split_idx" in options and options["val_split_idx"] is not None:
        val_split_idx = int(options["val_split_idx"])
    else:
        val_split_idx = int(n_train_total * 0.80)
    val_split_idx = max(1, min(n_train_total - 1, val_split_idx))

    X_subtrain = np.array(X_train[:val_split_idx], dtype=np.float32)
    y_subtrain = np.array(y_train[:val_split_idx], dtype=np.int32)
    X_val = np.array(X_train[val_split_idx:], dtype=np.float32)
    y_val = np.array(y_train[val_split_idx:], dtype=np.int32)

    # Class imbalance weighting computed strictly on the fitted 80%
    n_pos_sub = int(np.sum(y_subtrain))
    n_neg_sub = len(y_subtrain) - n_pos_sub
    scale_pos_weight = float(n_neg_sub / max(1, n_pos_sub))

    # User-configurable parameters with sane clamping and validation
    learning_rate = max(0.001, min(1.0, float(options.get("learning_rate", 0.05))))
    num_leaves = max(2, min(255, int(options.get("num_leaves", 15))))
    max_depth = max(-1, min(32, int(options.get("max_depth", 6))))
    min_data_in_leaf = max(1, min(1000, int(options.get("min_data_in_leaf", 50))))
    feature_fraction = max(0.1, min(1.0, float(options.get("feature_fraction", 0.8))))
    bagging_fraction = max(0.1, min(1.0, float(options.get("bagging_fraction", 0.8))))
    bagging_freq = max(0, min(10, int(options.get("bagging_freq", 1))))
    lambda_l2 = max(0.0, min(100.0, float(options.get("lambda_l2", options.get("l2", 1.0)))))
    seed = int(options.get("seed", 42))
    num_boost_round = max(10, min(1000, int(options.get("num_boost_round", options.get("epochs", 300)))))
    time_limit_sec = max(0.0001, min(25.0, float(options.get("time_limit_sec", 12.0))))

    actual_params = {
        "objective": "binary",
        "learning_rate": learning_rate,
        "num_leaves": num_leaves,
        "max_depth": max_depth,
        "min_data_in_leaf": min_data_in_leaf,
        "feature_fraction": feature_fraction,
        "bagging_fraction": bagging_fraction,
        "bagging_freq": bagging_freq,
        "lambda_l2": lambda_l2,
        "scale_pos_weight": round(scale_pos_weight, 4),
        "num_threads": 2,
        "seed": seed,
        "deterministic": True,
        "force_row_wise": True,
        "verbosity": -1,
        "metric": ["average_precision", "binary_logloss"]
    }

    # Construct LightGBM datasets
    ds_train_kwargs = {"feature_name": feature_names} if feature_names else {}
    ds_train = lgb.Dataset(X_subtrain, label=y_subtrain, free_raw_data=False, **ds_train_kwargs)
    ds_val = lgb.Dataset(X_val, label=y_val, reference=ds_train, free_raw_data=False, **ds_train_kwargs)

    evals_result = {}
    time_guard = make_time_guard_callback(time_limit_sec)

    t_train_start = time.perf_counter()
    booster = lgb.train(
        actual_params,
        ds_train,
        num_boost_round=num_boost_round,
        valid_sets=[ds_val],
        valid_names=["val"],
        callbacks=[
            # first_metric_only: stop on average_precision only. With scale_pos_weight the unweighted
            # validation logloss bottoms out after a handful of trees and would end training too early.
            lgb.early_stopping(stopping_rounds=30, first_metric_only=True, verbose=False),
            lgb.record_evaluation(evals_result),
            time_guard
        ]
    )
    t_train_end = time.perf_counter()
    train_duration_ms = round((t_train_end - t_train_start) * 1000, 2)

    return booster, X_val, y_val, scale_pos_weight, actual_params, evals_result, train_duration_ms


def handle_train(body_json, request_id):
    """Handles POST /train route using LightGBM gradient-boosted trees."""
    t_start = time.perf_counter()

    # 1. Check LightGBM layer availability
    lgb_avail, lgb_ver, lgb_err = check_lightgbm()
    if not lgb_avail:
        logger.error(f"POST /train failed: LightGBM layer is not attached. Error: {lgb_err}")
        return json_response(500, {
            "error": "LightGBM layer is not attached to this Lambda",
            "detail": lgb_err
        })

    import numpy as np
    import lightgbm as lgb

    # 2. Input Validation
    feature_names = body_json.get("feature_names")
    train_data = body_json.get("train", {})
    test_data = body_json.get("test")
    options = body_json.get("options", {})
    meta = body_json.get("meta", {})

    if not isinstance(feature_names, list) or not feature_names:
        return json_response(400, {"error": "Invalid payload: feature_names must be a non-empty list of strings."})

    X_train = train_data.get("X")
    y_train = train_data.get("y")

    if not (isinstance(X_train, list) and isinstance(y_train, list) and len(X_train) == len(y_train)):
        return json_response(400, {"error": "Invalid payload: train.X and train.y must be equal-length lists."})
    if len(X_train) < 10:
        return json_response(400, {"error": "Insufficient dataset size: train partition requires at least 10 rows."})
    if len(X_train) > 60000:
        return json_response(400, {"error": "Dataset partition exceeds allowed maximum row limits."})

    has_test = False
    X_test = None
    y_test = None
    if isinstance(test_data, dict) and "X" in test_data and "y" in test_data:
        X_test = test_data.get("X")
        y_test = test_data.get("y")
        if not (isinstance(X_test, list) and isinstance(y_test, list) and len(X_test) == len(y_test)):
            return json_response(400, {"error": "Invalid payload: test.X and test.y must be equal-length lists."})
        if len(X_test) < 10:
            return json_response(400, {"error": "Insufficient dataset size: test partition requires at least 10 rows."})
        if len(X_test) > 30000:
            return json_response(400, {"error": "Dataset partition exceeds allowed maximum row limits."})
        has_test = True

    d = len(feature_names)
    for r_idx, row in enumerate(X_train[:100]):
        if len(row) != d:
            return json_response(400, {"error": f"Feature count mismatch at train row {r_idx}: expected {d}, got {len(row)}."})

    # Validate numeric values (allow null / None / NaN, disallow Inf or non-numeric types)
    for row in X_train:
        for val in row:
            if val is not None and (not isinstance(val, (int, float)) or math.isinf(val)):
                return json_response(400, {"error": "Invalid values detected in feature matrix (must be numeric or null)."})

    if has_test:
        for row in X_test:
            for val in row:
                if val is not None and (not isinstance(val, (int, float)) or math.isinf(val)):
                    return json_response(400, {"error": "Invalid values detected in test feature matrix (must be numeric or null)."})

    # 3. Model Fitting
    train_options = dict(options)
    if "feature_names" not in train_options and feature_names:
        train_options["feature_names"] = feature_names

    booster, X_val, y_val, scale_pos_weight, actual_params, evals_result, train_duration_ms = fit_model(
        X_train, y_train, train_options
    )

    # 4. Derive validation thresholds strictly on the validation slice
    val_scores = booster.predict(X_val).tolist()
    threshold_res = derive_thresholds(y_val.tolist(), val_scores)
    thresholds = {
        "decision": threshold_res["decision"],
        "review": threshold_res["review"],
        "block": threshold_res["block"],
        "block_target_met": threshold_res["block_target_met"]
    }
    threshold_diagnostics = threshold_res["threshold_diagnostics"]
    warnings = threshold_res["warnings"]

    # 5. Evaluate on TEST partition (if provided)
    if has_test:
        X_test_arr = np.array(X_test, dtype=np.float32)
        y_test_list = [int(v) for v in y_test]
        test_scores = booster.predict(X_test_arr).tolist()
        eval_results = evaluate_predictions(y_test_list, test_scores, thresholds["decision"])
        roc_auc = compute_roc_auc(y_test_list, test_scores)
        pr_auc = compute_pr_auc(y_test_list, test_scores)

        # Always-Legitimate baseline comparison
        n_test = len(y_test_list)
        n_test_neg = n_test - sum(y_test_list)
        baseline_metrics = {
            "name": "Always Legitimate Baseline",
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "accuracy": round(n_test_neg / max(1, n_test), 4)
        }
        test_metrics = {
            "precision": eval_results["precision"],
            "recall": eval_results["recall"],
            "f1": eval_results["f1"],
            "accuracy": eval_results["accuracy"],
            "roc_auc": round(roc_auc, 4),
            "pr_auc": round(pr_auc, 4)
        }
        confusion_matrix = eval_results["confusion"]
    else:
        test_metrics = None
        confusion_matrix = None
        baseline_metrics = None
        roc_auc = None
        pr_auc = None

    # Top 10 feature importances (normalized LightGBM gain, non-negative percentages)
    gains = booster.feature_importance(importance_type="gain")
    total_gain = float(np.sum(gains))
    importance_list = []
    for j in range(d):
        gain_pct = (float(gains[j]) / total_gain * 100.0) if total_gain > 0 else 0.0
        importance_list.append({
            "feature": feature_names[j],
            "gain_pct": round(gain_pct, 2)
        })
    importance_list.sort(key=lambda x: x["gain_pct"], reverse=True)
    top_importance = importance_list[:10]

    # Compact exported trees for browser scoring
    raw_dump = booster.dump_model()
    compact_trees = [compact_tree_node(t["tree_structure"]) for t in raw_dump.get("tree_info", [])]
    n_trees_exported = len(compact_trees)  # trees up to best_iteration (what the browser and S3 model use)

    # Format evaluation history for telemetry / client graphs (up to 50 points)
    eval_history = []
    val_metrics = evals_result.get("val", {})
    val_loss = val_metrics.get("binary_logloss", [])
    val_ap = val_metrics.get("average_precision", [])
    n_pts = len(val_loss)
    step_size = max(1, math.ceil(n_pts / 50))
    for idx in range(0, n_pts, step_size):
        eval_history.append({
            "iteration": idx + 1,
            "binary_logloss": round(float(val_loss[idx]), 4),
            "average_precision": round(float(val_ap[idx]), 4) if idx < len(val_ap) else None
        })

    iso_timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    model_version = f"model-{iso_timestamp.replace(':', '-')}"
    total_duration_ms = round((time.perf_counter() - t_start) * 1000, 2)

    # Browser-format model artifact bundle (no scaler)
    trained_model_bundle = {
        "model_version": model_version,
        "trained_at": iso_timestamp,
        "feature_names": feature_names,
        "trees": compact_trees,
        "thresholds": thresholds,
        "threshold_diagnostics": threshold_diagnostics,
        "warnings": warnings,
        "objective": "binary",
        "best_iteration": int(booster.best_iteration),
        "num_trees": n_trees_exported,
        "params": actual_params,
        "metrics": test_metrics,
        "confusion": confusion_matrix,
        "baseline": baseline_metrics,
        "importance": top_importance,
        "meta": meta
    }

    # Update Lambda in-memory cache
    global _CACHED_MODEL, _CACHED_BOOSTER, _CACHED_VERSION
    _CACHED_MODEL = trained_model_bundle
    _CACHED_BOOSTER = booster
    _CACHED_VERSION = model_version

    # Persist to S3: models/latest.json, models/latest.txt, plus timestamped versions & telemetry
    s3_persisted = True
    s3_errors = []

    native_model_str = booster.model_to_string()

    ok1, err1 = upload_to_s3("models/latest.json", trained_model_bundle, "application/json")
    ok2, err2 = upload_to_s3(f"models/{model_version}.json", trained_model_bundle, "application/json")
    ok3, err3 = upload_to_s3("models/latest.txt", native_model_str, "text/plain")
    ok4, err4 = upload_to_s3(f"models/{model_version}.txt", native_model_str, "text/plain")
    ok5, err5 = upload_to_s3("artifacts/metrics.json", trained_model_bundle["metrics"], "application/json") if has_test else (True, None)

    pipeline_status = {
        "status": "SUCCESS",
        "last_trained_at": iso_timestamp,
        "model_version": model_version,
        "metrics": trained_model_bundle["metrics"],
        "trees_count": n_trees_exported,
        "best_iteration": int(booster.best_iteration),
        "timings": {"train_ms": train_duration_ms, "total_ms": total_duration_ms},
        "warnings": warnings
    }
    ok6, err6 = upload_to_s3("pipeline_status.json", pipeline_status, "application/json")

    if not (ok1 and ok2 and ok3 and ok4 and ok5 and ok6):
        s3_persisted = False
        s3_errors = [e for e in [err1, err2, err3, err4, err5, err6] if e]

    response_payload = {
        "status": "SUCCESS",
        "model_version": model_version,
        "model": {
            "trees": compact_trees,
            "feature_names": feature_names,
            "thresholds": thresholds,
            "model_version": model_version,
            "objective": "binary"
        },
        "thresholds": thresholds,
        "threshold_diagnostics": threshold_diagnostics,
        "warnings": warnings,
        "metrics": trained_model_bundle["metrics"],
        "confusion": confusion_matrix,
        "baseline": baseline_metrics,
        "importance": top_importance,
        "eval_history": eval_history,
        "trees_count": n_trees_exported,
        "best_iteration": int(booster.best_iteration),
        "params": actual_params,
        "timings": {
            "train_ms": train_duration_ms,
            "total_ms": total_duration_ms
        },
        "persisted": s3_persisted,
        "s3_errors": s3_errors if not s3_persisted else []
    }

    test_cnt = len(X_test) if has_test else 0
    roc_str = f"{roc_auc:.4f}" if has_test else "N/A"
    pr_str = f"{pr_auc:.4f}" if has_test else "N/A"
    logger.info(
        f"train_complete request_id={request_id} train_rows={len(X_train)} "
        f"test_rows={test_cnt} trees={n_trees_exported} best_iter={booster.best_iteration} "
        f"train_ms={train_duration_ms} roc_auc={roc_str} pr_auc={pr_str}"
    )

    return json_response(200, response_payload)


def handle_predict(body_json, request_id):
    """Handles POST /predict route with real LightGBM native inference."""
    global _CACHED_BOOSTER, _CACHED_MODEL, _CACHED_VERSION

    # 1. Check LightGBM layer availability
    lgb_avail, lgb_ver, lgb_err = check_lightgbm()
    if not lgb_avail:
        logger.error(f"POST /predict failed: LightGBM layer is not attached. Error: {lgb_err}")
        return json_response(500, {
            "error": "LightGBM layer is not attached to this Lambda",
            "detail": lgb_err
        })

    import numpy as np
    import lightgbm as lgb

    # Fallback to S3 if Lambda container was cold
    if _CACHED_BOOSTER is None and s3_client is not None:
        try:
            resp_txt = s3_client.get_object(Bucket=DATA_BUCKET_NAME, Key="models/latest.txt")
            txt_model = resp_txt["Body"].read().decode("utf-8")
            _CACHED_BOOSTER = lgb.Booster(model_str=txt_model)

            resp_json = s3_client.get_object(Bucket=DATA_BUCKET_NAME, Key="models/latest.json")
            _CACHED_MODEL = json.loads(resp_json["Body"].read().decode("utf-8"))
            _CACHED_VERSION = _CACHED_MODEL.get("model_version", "v1")
            logger.info("Loaded models/latest.txt and models/latest.json from S3 into cold container.")
        except Exception as e:
            logger.warning(f"Could not load native model from S3: {e}")

    if _CACHED_BOOSTER is None:
        return json_response(400, {
            "error": "No trained model available. Please execute model training via POST /train first."
        })

    features = body_json.get("features")
    if not isinstance(features, list) or not features:
        return json_response(400, {"error": "Invalid payload: features must be a non-empty 2D array of numeric values."})
    if len(features) > 1000:
        return json_response(400, {"error": "Batch size limit exceeded: POST /predict accepts at most 1,000 rows."})

    expected_dim = _CACHED_BOOSTER.num_feature()
    for r_idx, row in enumerate(features[:100]):
        if len(row) != expected_dim:
            return json_response(400, {
                "error": f"Feature dimension mismatch at row {r_idx}: expected {expected_dim}, got {len(row)}."
            })

    try:
        X_arr = np.array(features, dtype=np.float32)
        raw_scores = _CACHED_BOOSTER.predict(X_arr)
    except Exception as pred_err:
        logger.error(f"Prediction computation failed: {pred_err}")
        return json_response(500, {"error": f"Inference execution failed: {str(pred_err)}"})

    thresholds = (_CACHED_MODEL.get("thresholds") if _CACHED_MODEL else None) or {
        "decision": 0.50,
        "review": 0.40,
        "block": 0.75
    }

    scores = []
    actions = []
    for s in raw_scores:
        s_val = float(s)
        scores.append(s_val)
        if s_val >= thresholds["block"]:
            actions.append("Block")
        elif s_val >= thresholds["review"]:
            actions.append("Review")
        else:
            actions.append("Approve")

    return json_response(200, {
        "status": "SUCCESS",
        "model_version": _CACHED_VERSION or (_CACHED_MODEL.get("model_version") if _CACHED_MODEL else "v1"),
        "scores": scores,
        "actions": actions,
        "thresholds": thresholds
    })


def handle_get_model(request_id):
    """Handles GET /model route returning the active compact browser-format JSON bundle."""
    global _CACHED_MODEL
    model_bundle = _CACHED_MODEL

    if model_bundle is None and s3_client is not None:
        try:
            resp = s3_client.get_object(Bucket=DATA_BUCKET_NAME, Key="models/latest.json")
            model_bundle = json.loads(resp["Body"].read().decode("utf-8"))
            _CACHED_MODEL = model_bundle
        except Exception as e:
            logger.warning(f"Could not load models/latest.json from S3: {e}")

    if model_bundle is None:
        return json_response(404, {"error": "No model found on S3 or in Lambda memory."})

    return json_response(200, model_bundle)


def handle_health(request_id):
    """Handles GET /health route reporting service health and LightGBM layer diagnostics."""
    global _CACHED_MODEL
    lgb_avail, lgb_ver, lgb_err = check_lightgbm()

    body = {
        "status": "healthy",
        "version": "2.0.0",
        "service": "aws-financial-fraud-detection",
        "model_available": _CACHED_MODEL is not None,
        "last_trained_at": _CACHED_MODEL.get("trained_at") if _CACHED_MODEL else None,
        "lightgbm_available": lgb_avail,
        "lightgbm_version": lgb_ver
    }
    if not lgb_avail and lgb_err:
        body["lightgbm_error"] = lgb_err

    return json_response(200, body)


def verify_train_auth(event, request_id: str, raw_path: str):
    """
    Verifies that the request provides an x-api-key header matching TRAIN_API_KEY.
    If TRAIN_API_KEY is unset in the environment, fails closed with an error.
    Returns None if authorized, or a json_response dictionary if unauthorized or misconfigured.
    """
    train_api_key = os.environ.get("TRAIN_API_KEY")
    if not train_api_key:
        logger.error(f"TRAIN_API_KEY environment variable is unset; failing closed request_id={request_id} path={raw_path}")
        return json_response(500, {
            "error": "Server configuration error: TRAIN_API_KEY environment variable is not configured."
        })

    headers = (event or {}).get("headers") or {}
    provided_key = None
    for k, v in headers.items():
        if k.lower() == "x-api-key":
            provided_key = v
            break

    if not provided_key or not hmac.compare_digest(str(provided_key), str(train_api_key)):
        logger.warning(f"Unauthorized API key request_id={request_id} path={raw_path}")
        return json_response(401, {
            "error": "Unauthorized: invalid or missing x-api-key header."
        })

    return None


def lambda_handler(event, context=None):
    """Main AWS Lambda handler supporting HTTP API v2."""
    if event is None:
        event = {}
    request_id = getattr(context, "aws_request_id", f"local-{int(time.time() * 1000)}")

    # 1. Parse HTTP Method and Path
    http_method = (
        event.get("requestContext", {}).get("http", {}).get("method")
        or event.get("httpMethod")
        or "GET"
    ).upper()

    raw_path = (
        event.get("rawPath")
        or event.get("requestContext", {}).get("http", {}).get("path")
        or event.get("path")
        or "/"
    ).rstrip("/")
    if not raw_path:
        raw_path = "/"

    # 2. CORS Preflight
    if http_method == "OPTIONS":
        return {
            "statusCode": 200,
            "headers": get_cors_headers(),
            "body": ""
        }

    # 3. Protected endpoint authentication (POST /train and /pipeline)
    is_protected_train_route = (http_method == "POST" and raw_path == "/train") or (raw_path == "/pipeline")
    if is_protected_train_route:
        auth_error = verify_train_auth(event, request_id, raw_path)
        if auth_error is not None:
            return auth_error

    # 4. Route Dispatch
    try:
        # Route: GET /health
        if http_method == "GET" and raw_path == "/health":
            return handle_health(request_id)

        # Route: GET /model
        if http_method == "GET" and raw_path == "/model":
            return handle_get_model(request_id)

        # Route: POST /train
        if http_method == "POST" and (raw_path == "/train" or raw_path == "/pipeline"):
            body_str = event.get("body", "{}")
            if event.get("isBase64Encoded", False):
                import base64
                body_str = base64.b64decode(body_str).decode("utf-8")
            if not body_str:
                return json_response(400, {"error": "Missing request body."})
            try:
                body_json = json.loads(body_str)
            except Exception as json_err:
                return json_response(400, {"error": f"Malformed JSON in request body: {str(json_err)}"})
            return handle_train(body_json, request_id)

        # Route: POST /predict
        if http_method == "POST" and raw_path == "/predict":
            body_str = event.get("body", "{}")
            if event.get("isBase64Encoded", False):
                import base64
                body_str = base64.b64decode(body_str).decode("utf-8")
            if not body_str:
                return json_response(400, {"error": "Missing request body."})
            try:
                body_json = json.loads(body_str)
            except Exception as json_err:
                return json_response(400, {"error": f"Malformed JSON in request body: {str(json_err)}"})
            return handle_predict(body_json, request_id)

        # Backward compatibility for GET /pipeline
        if http_method == "GET" and raw_path == "/pipeline":
            return handle_health(request_id)

        # Unknown route
        return json_response(404, {
            "error": f"Not Found: No route matching {http_method} {raw_path}",
            "available_routes": ["GET /health", "POST /train", "POST /predict", "GET /model"]
        })

    except Exception as unhandled_err:
        logger.error(f"Unhandled error in lambda_handler request_id={request_id}: {unhandled_err}", exc_info=True)
        return json_response(500, {
            "error": "Internal server error occurred processing request.",
            "request_id": request_id
        })