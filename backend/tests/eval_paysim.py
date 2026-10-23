"""
PaySim Evaluation Script.
Evaluates LightGBM model on the Kaggle PaySim financial fraud dataset.

Usage:
    python backend/tests/eval_paysim.py --csv dataset\\paysim.csv [--max-rows N] [--drop-post-balance] [--output results/paysim.json]
"""

import os
import sys
import csv
import time
import json
import argparse
from typing import Dict, Any, List

# Ensure backend directory is in sys.path
backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

import features as py_feat
import lambda_function as lf
import numpy as np


def print_missing_instructions(csv_path: str) -> None:
    """Print instructions on how to obtain and place the Kaggle PaySim dataset."""
    msg = f"""
{'=' * 80}
PaySim Dataset Not Found
{'=' * 80}
CSV file was not found at: {csv_path}

To run this evaluation against the Kaggle PaySim synthetic financial fraud dataset:
1. Download 'paysim1' from Kaggle:
   https://www.kaggle.com/datasets/ealaxi/paysim1
2. Extract the downloaded CSV (e.g. PS_20174392719_1491204439457_log.csv).
3. Place or link the CSV file at:
   {csv_path}
4. Re-run this evaluation command:
   python backend/tests/eval_paysim.py --csv {csv_path} [--max-rows 100000] [--drop-post-balance]
{'=' * 80}
"""
    print(msg)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate LightGBM model on Kaggle PaySim synthetic fraud dataset."
    )
    parser.add_argument(
        "--csv",
        required=True,
        help="Path to PaySim CSV file (e.g. dataset\\paysim.csv)",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Maximum number of rows to process from CSV (default: all rows)",
    )
    parser.add_argument(
        "--drop-post-balance",
        action="store_true",
        help="Remove features that use post-transaction balances (indexes 1 to 5)",
    )
    parser.add_argument(
        "--output",
        default="results/paysim.json",
        help="Path to output JSON results file (default: results/paysim.json)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    csv_path = os.path.abspath(args.csv)

    if not os.path.exists(csv_path):
        print_missing_instructions(args.csv)
        sys.exit(0)

    print(f"Loading PaySim dataset from: {csv_path}")
    if args.max_rows:
        print(f"Limiting ingestion to first {args.max_rows:,} rows.")
    if args.drop_post_balance:
        print("Feature mode: --drop-post-balance ENABLED (indices 1 to 5 removed).")
    else:
        print("Feature mode: All 19 causal features enabled.")

    POST_BALANCE_INDICES = {1, 2, 3, 4, 5}
    if args.drop_post_balance:
        keep_indices = [i for i in range(len(py_feat.FEATURE_NAMES)) if i not in POST_BALANCE_INDICES]
        feature_names = [py_feat.FEATURE_NAMES[i] for i in keep_indices]
    else:
        keep_indices = list(range(len(py_feat.FEATURE_NAMES)))
        feature_names = list(py_feat.FEATURE_NAMES)

    # 1. Streaming Feature Extraction
    stream = py_feat.CausalFeatureStream()
    X = []
    y = []

    t_feat_start = time.perf_counter()
    rows_read = 0

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            vec = stream.process_row(row)
            if args.drop_post_balance:
                vec = [vec[i] for i in keep_indices]
            X.append(vec)
            try:
                y.append(int(row.get("isFraud", 0)))
            except (ValueError, TypeError):
                y.append(0)

            rows_read += 1
            if rows_read % 100000 == 0:
                print(f"  Processed {rows_read:,} rows...")
            if args.max_rows and rows_read >= args.max_rows:
                break

    t_feat_end = time.perf_counter()
    feat_duration_sec = round(t_feat_end - t_feat_start, 2)
    print(f"Feature extraction completed: {rows_read:,} rows in {feat_duration_sec}s ({(rows_read / max(0.01, feat_duration_sec)):.0f} rows/s).")

    # 2. Chronological 70/30 Split by step
    n_total = len(X)
    train_count = int(n_total * 0.70)
    X_train_full = X[:train_count]
    y_train_full = y[:train_count]
    X_test = X[train_count:]
    y_test = y[train_count:]

    # 3. Train part 80/20 chronological split into fit and validation (natural prevalence)
    fit_count = int(train_count * 0.80)
    X_fit = X_train_full[:fit_count]
    y_fit = y_train_full[:fit_count]
    X_val = X_train_full[fit_count:]
    y_val = y_train_full[fit_count:]

    n_fit_pos = sum(y_fit)
    n_val_pos = sum(y_val)
    n_test_pos = sum(y_test)
    n_test = len(y_test)

    print("\n" + "=" * 80)
    print(f"DATASET PARTITION SUMMARY ({n_total:,} total records)")
    print("=" * 80)
    print(f"  Fit Slice (80% of Train) : {len(X_fit):,} records | {n_fit_pos} frauds ({n_fit_pos/max(1, len(X_fit))*100:.3f}% prevalence)")
    print(f"  Validation Slice (20%)   : {len(X_val):,} records | {n_val_pos} frauds ({n_val_pos/max(1, len(X_val))*100:.3f}% prevalence)")
    print(f"  Test Partition (30%)     : {len(X_test):,} records | {n_test_pos} frauds ({n_test_pos/max(1, len(X_test))*100:.3f}% prevalence)")
    print("=" * 80)

    # 4. Model Training via lf.fit_model
    # X_train passed to fit_model consists of X_fit + X_val; fit_model splits at 80% boundary
    X_train_input = X_fit + X_val
    y_train_input = y_fit + y_val

    train_options = {
        "num_boost_round": 300,
        "learning_rate": 0.05,
        "num_leaves": 15,
        "max_depth": 6,
        "min_data_in_leaf": 50,
        "seed": 42,
        "feature_names": feature_names,
    }

    print("\nTraining LightGBM model via fit_model...")
    booster, X_val_ret, y_val_ret, scale_pos_weight, actual_params, evals_result, train_duration_ms = lf.fit_model(
        X_train_input, y_train_input, options=train_options, feature_names=feature_names
    )
    print(f"Training completed in {train_duration_ms:.2f} ms ({booster.num_trees()} trees, scale_pos_weight: {scale_pos_weight:.2f}).")

    # 5. Derive Thresholds on Validation Slice
    val_scores = booster.predict(X_val_ret).tolist()
    threshold_res = lf.derive_thresholds(y_val_ret.tolist(), val_scores)
    thresholds = {
        "review": threshold_res["decision"] if threshold_res["review"] > threshold_res["decision"] else threshold_res["review"],
        "decision": threshold_res["decision"],
        "block": threshold_res["block"],
        "block_target_met": threshold_res["block_target_met"],
    }
    # Enforce review <= decision <= block invariant
    thresholds["review"] = min(thresholds["review"], thresholds["decision"])
    thresholds["block"] = max(thresholds["block"], thresholds["decision"])

    print("\n" + "=" * 80)
    print("DERIVED VALIDATION THRESHOLDS (strictly on validation slice)")
    print("=" * 80)
    print(f"  Review Threshold   : {thresholds['review']:.2f}")
    print(f"  Decision Threshold : {thresholds['decision']:.2f} (F1-optimal cut)")
    print(f"  Block Threshold    : {thresholds['block']:.2f} (Target met: {thresholds['block_target_met']})")
    if threshold_res.get("warnings"):
        print("  Warnings:")
        for w in threshold_res["warnings"]:
            print(f"    - {w}")
    print("=" * 80)

    # 6. Evaluation on Test Partition
    print("\nEvaluating model on test partition...")
    X_test_arr = np.array(X_test, dtype=np.float32)
    test_scores = booster.predict(X_test_arr).tolist()

    pr_auc = lf.compute_pr_auc(y_test, test_scores)
    roc_auc = lf.compute_roc_auc(y_test, test_scores)
    base_acc = (n_test - n_test_pos) / max(1, n_test)

    eval_by_threshold = {}
    for name in ["review", "decision", "block"]:
        th = thresholds[name]
        res = lf.evaluate_predictions(y_test, test_scores, th)
        eval_by_threshold[name] = {
            "threshold": th,
            "precision": res["precision"],
            "recall": res["recall"],
            "f1": res["f1"],
            "accuracy": res["accuracy"],
            "confusion": res["confusion"],
        }

    # Print summary table
    print("\n" + "=" * 80)
    print(f"TEST PARTITION EVALUATION RESULTS ({n_test:,} records, {n_test_pos} frauds, prevalence: {n_test_pos/max(1, n_test)*100:.3f}%)")
    print("=" * 80)
    print(f"{'Metric':<30} | {'Value':<45}")
    print("-" * 80)
    print(f"{'PR-AUC (Headline)':<30} | {pr_auc:<45.4f}")
    print(f"{'ROC-AUC':<30} | {roc_auc:<45.4f}")
    print(f"{'Baseline Accuracy':<30} | {base_acc*100:<44.2f}%")
    print("-" * 80)
    print(f"{'Threshold Cut':<15} | {'Cut':<6} | {'Precision':<10} | {'Recall':<10} | {'F1':<8} | {'Confusion (TP/FP/FN/TN)':<24}")
    print("-" * 80)
    for name in ["review", "decision", "block"]:
        e = eval_by_threshold[name]
        cm = f"{e['confusion']['tp']}/{e['confusion']['fp']}/{e['confusion']['fn']}/{e['confusion']['tn']}"
        print(
            f"{name.capitalize():<15} | {e['threshold']:<6.2f} | "
            f"{e['precision']*100:>8.2f}% | {e['recall']*100:>8.2f}% | "
            f"{e['f1']*100:>6.2f}% | {cm:<24}"
        )
    print("=" * 80)

    # 7. Write Results JSON
    output_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    results_data = {
        "dataset": {
            "csv_path": args.csv,
            "total_rows_read": rows_read,
            "drop_post_balance": args.drop_post_balance,
            "features_count": len(feature_names),
            "train_rows": len(X_train_full),
            "fit_rows": len(X_fit),
            "fit_frauds": n_fit_pos,
            "val_rows": len(X_val),
            "val_frauds": n_val_pos,
            "test_rows": len(X_test),
            "test_frauds": n_test_pos,
            "test_prevalence_pct": round(n_test_pos / max(1, n_test) * 100, 4),
        },
        "features": feature_names,
        "model": {
            "trees_count": booster.num_trees(),
            "best_iteration": getattr(booster, "best_iteration", booster.num_trees()),
            "train_duration_ms": train_duration_ms,
            "scale_pos_weight": scale_pos_weight,
            "params": actual_params,
        },
        "thresholds": {
            "review": thresholds["review"],
            "decision": thresholds["decision"],
            "block": thresholds["block"],
            "block_target_met": thresholds["block_target_met"],
            "warnings": threshold_res.get("warnings", []),
        },
        "test_metrics": {
            "pr_auc": round(pr_auc, 4),
            "roc_auc": round(roc_auc, 4),
            "baseline_accuracy": round(base_acc, 4),
            "at_thresholds": eval_by_threshold,
            "validation_diagnostics": threshold_res.get("threshold_diagnostics", {}),
        },
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=2)

    print(f"\nWrote results to {output_path}")


if __name__ == "__main__":
    main()
