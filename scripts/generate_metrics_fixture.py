"""
Generates tests/fixtures/metrics_expected.json using scikit-learn's
average_precision_score and roc_auc_score on seeded scores and labels with ties.
"""

import json
import os
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

def create_dataset(seed, size, p_pos, round_digits=2):
    rng = np.random.RandomState(seed)
    y_true = rng.choice([0, 1], size=size, p=[1.0 - p_pos, p_pos]).tolist()
    # Rounding generates explicit ties across threshold boundaries
    y_score = [round(float(s), round_digits) for s in rng.uniform(0.0, 1.0, size=size)]
    return y_true, y_score

def main():
    fixtures = {}

    # Case 1: Standard seeded dataset with ties
    y1, s1 = create_dataset(seed=42, size=1000, p_pos=0.15, round_digits=2)
    fixtures["standard_seeded"] = {
        "description": "1000 samples, 15% prevalence, 2-decimal rounded scores with multiple ties",
        "y_true": y1,
        "y_score": s1,
        "pr_auc": float(average_precision_score(y1, s1)),
        "roc_auc": float(roc_auc_score(y1, s1))
    }

    # Case 2: Handcrafted ties across thresholds
    y2 = [0, 0, 1, 0, 1, 1, 0, 1, 1, 0]
    s2 = [0.1, 0.2, 0.2, 0.5, 0.5, 0.5, 0.8, 0.8, 0.9, 0.9]
    fixtures["handcrafted_ties"] = {
        "description": "Small handcrafted vector with known ties and alternating labels",
        "y_true": y2,
        "y_score": s2,
        "pr_auc": float(average_precision_score(y2, s2)),
        "roc_auc": float(roc_auc_score(y2, s2))
    }

    # Case 3: High imbalance fraud-like distribution (prevalence 1.5%) with ties
    y3, s3 = create_dataset(seed=123, size=2000, p_pos=0.015, round_digits=2)
    fixtures["imbalance_ties"] = {
        "description": "2000 samples, 1.5% fraud prevalence, rounded scores with ties",
        "y_true": y3,
        "y_score": s3,
        "pr_auc": float(average_precision_score(y3, s3)),
        "roc_auc": float(roc_auc_score(y3, s3))
    }

    # Case 4: Coarser ties (1-decimal rounded scores)
    y4, s4 = create_dataset(seed=999, size=500, p_pos=0.20, round_digits=1)
    fixtures["coarse_ties"] = {
        "description": "500 samples, 1-decimal rounded scores (heavy ties in 10 buckets)",
        "y_true": y4,
        "y_score": s4,
        "pr_auc": float(average_precision_score(y4, s4)),
        "roc_auc": float(roc_auc_score(y4, s4))
    }

    out_dir = os.path.join(os.path.dirname(__file__), "..", "tests", "fixtures")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "metrics_expected.json")

    with open(out_file, "w") as f:
        json.dump(fixtures, f, indent=2)

    print(f"Generated {out_file} with {len(fixtures)} test cases using scikit-learn.")
    for name, data in fixtures.items():
        print(f"  {name}: PR-AUC = {data['pr_auc']:.8f}, ROC-AUC = {data['roc_auc']:.8f}")

if __name__ == "__main__":
    main()
