import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import lambda_function as lf

test_dir = os.path.dirname(os.path.abspath(__file__))
payload_path = os.path.join(test_dir, "sample_payload.json")
out_path = os.path.join(test_dir, "temp_parity_data.json")

with open(payload_path, "r") as f:
    payload = json.load(f)

X_tr = payload["train"]["X"]
y_tr = payload["train"]["y"]
X_te = payload["test"]["X"]
y_te = payload["test"]["y"]

options = {
    "num_boost_round": 100,
    "feature_names": payload.get("feature_names")
}

booster, X_val, y_val, scale_pos_weight, actual_params, evals_result, train_duration = lf.fit_model(
    X_tr, y_tr, options
)

raw_dump = booster.dump_model()
compact_trees = [lf.compact_tree_node(t["tree_structure"]) for t in raw_dump["tree_info"]]

test_1k = X_te[:1000]
test_1k_arr = np.array(test_1k, dtype=np.float32)
python_preds = booster.predict(test_1k_arr).tolist()

out_data = {
    "model": {
        "trees": compact_trees,
        "feature_names": payload["feature_names"],
        "objective": "binary"
    },
    "test_rows": test_1k,
    "python_preds": python_preds
}

with open(out_path, "w") as f_out:
    json.dump(out_data, f_out)

print(f"Exported {len(compact_trees)} trees and {len(python_preds)} predictions to {out_path}.")
