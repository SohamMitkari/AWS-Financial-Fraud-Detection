"""
Automated Parity Data Generator for Real Mode (Part C)
Trains LightGBM via lambda_function.lambda_handler on creditcard_sample300.csv (Part A sample),
exports compact trees, and generates predictions across rows with NaNs in features that had no NaNs in training.
"""

import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend")))
os.environ["TRAIN_API_KEY"] = "parity-test-key-32chars"
import lambda_function as lf

def main():
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    csv_path = os.path.join(root_dir, "tests", "fixtures", "real", "creditcard_sample300.csv")
    out_path = os.path.join(root_dir, "tests", "fixtures", "real_parity_dump.json")

    with open(csv_path, "r", encoding="utf-8") as f:
        lines = [line.strip().split(",") for line in f if line.strip()]

    headers = lines[0]
    label_idx = headers.index("label") if "label" in headers else headers.index("Class")
    feat_indices = [i for i in range(len(headers)) if i != label_idx]
    feature_names = [headers[i] for i in feat_indices]

    X = []
    y = []
    for row in lines[1:]:
        y.append(int(float(row[label_idx])))
        X.append([float(row[i]) for i in feat_indices])

    # POST /train via local lambda_handler with NO test field
    event = {
        "requestContext": {"http": {"method": "POST"}},
        "rawPath": "/train",
        "headers": {"x-api-key": "parity-test-key-32chars"},
        "body": json.dumps({
            "feature_names": feature_names,
            "train": {"X": X, "y": y},
            "options": {
                "fit_rows": 240,
                "num_boost_round": 30,
                "learning_rate": 0.05,
                "num_leaves": 12,
                "min_data_in_leaf": 5,
                "seed": 42
            }
        })
    }

    resp = lf.lambda_handler(event)
    assert resp["statusCode"] == 200, f"Lambda train failed: {resp}"
    body = json.loads(resp["body"])
    model_bundle = body["model"]
    trees = model_bundle["trees"]

    # Python booster from memory
    booster = lf._CACHED_BOOSTER
    assert booster is not None, "Cached booster was not populated"

    # Construct test rows:
    # 1. Clean test rows
    # 2. Rows with null / NaN in various features (features that had NO NaNs during training)
    # 3. Rows with multiple NaNs
    test_rows = []
    for i in range(min(50, len(X))):
        test_rows.append(list(X[i]))

    num_feats = len(feature_names)
    for i in range(50):
        row_nan = list(X[i % len(X)])
        # Inject null in 1 or 2 columns that had no NaN in training
        col1 = i % num_feats
        col2 = (i * 3 + 1) % num_feats
        row_nan[col1] = None
        row_nan[col2] = None
        test_rows.append(row_nan)

    # Calculate native Python booster predictions
    # In Python, booster.predict converts None to np.nan
    X_test_arr = np.array([[np.nan if v is None else v for v in r] for r in test_rows], dtype=np.float32)
    py_preds = booster.predict(X_test_arr).tolist()

    out_data = {
        "model": model_bundle,
        "feature_names": feature_names,
        "test_rows": test_rows,
        "python_preds": py_preds
    }

    with open(out_path, "w", encoding="utf-8") as f_out:
        json.dump(out_data, f_out, indent=2)

    print(f"Exported real parity fixture to {out_path} ({len(trees)} trees, {len(test_rows)} test rows).")

if __name__ == "__main__":
    main()
