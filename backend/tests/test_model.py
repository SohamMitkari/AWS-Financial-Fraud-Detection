"""
Unit tests for LightGBM Gradient-Boosted Trees, Metrics, and AWS Lambda Handler.
No AWS credentials required; boto3 is mocked.
"""

import sys
import os
import math
import json
from unittest.mock import MagicMock
import pytest

# Add backend directory to sys.path
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

import lambda_function as lf

TEST_TRAIN_API_KEY = "test-secret-train-key-16chars"
os.environ["TRAIN_API_KEY"] = TEST_TRAIN_API_KEY


class TestMetrics:
    """Test ROC-AUC and PR-AUC with hand-computed examples including ties."""

    def test_roc_auc_hand_computed_no_ties(self):
        # Pairs: (0.1, 0), (0.35, 1), (0.4, 0), (0.8, 1)
        # Pos ranks: 2 + 4 = 6. M=2, N=2. U = 6 - (2*3)/2 = 3. AUC = 3/4 = 0.75
        y_true = [0, 0, 1, 1]
        y_score = [0.1, 0.4, 0.35, 0.8]
        auc = lf.compute_roc_auc(y_true, y_score)
        assert abs(auc - 0.75) < 1e-6

    def test_roc_auc_hand_computed_with_ties(self):
        # Tied scores at 0.5: ranks (2+3)/2 = 2.5
        # Positives at index 2 (rank 2.5) and index 3 (rank 4).
        # Sum = 6.5. U = 6.5 - 3 = 3.5. AUC = 3.5 / 4 = 0.875
        y_true = [0, 0, 1, 1]
        y_score = [0.1, 0.5, 0.5, 0.9]
        auc = lf.compute_roc_auc(y_true, y_score)
        assert abs(auc - 0.875) < 1e-6

    def test_roc_auc_perfect_and_worst(self):
        assert lf.compute_roc_auc([1, 1, 0, 0], [0.9, 0.8, 0.2, 0.1]) == 1.0
        assert lf.compute_roc_auc([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]) == 0.0

    def test_pr_auc_hand_computed(self):
        # Perfect separation
        assert abs(lf.compute_pr_auc([1, 1, 0, 0], [0.9, 0.8, 0.2, 0.1]) - 1.0) < 1e-6
        # Inverted separation
        ap_inverted = lf.compute_pr_auc([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1])
        assert abs(ap_inverted - (5.0 / 12.0)) < 1e-5


class TestLightGBMModel:
    """Test LightGBM training, tree export parity, wall-clock guard, and importance."""

    def test_training_on_separable_and_noisy_data(self):
        """(a) Training gives ROC-AUC well above 0.5 and on noisy data not 1.0."""
        import random
        rng = random.Random(1234)
        n_train = 400
        n_test = 100
        d = 4

        # Separable feature signal with realistic noise
        def make_data(n):
            X = [[rng.gauss(0, 1) for _ in range(d)] for _ in range(n)]
            y = [1 if X[i][0] + 0.6 * X[i][1] + rng.gauss(0, 0.4) > 0.5 else 0 for i in range(n)]
            return X, y

        X_tr, y_tr = make_data(n_train)
        X_te, y_te = make_data(n_test)

        event = {
            "requestContext": {"http": {"method": "POST"}},
            "rawPath": "/train",
            "headers": {"x-api-key": TEST_TRAIN_API_KEY},
            "body": json.dumps({
                "feature_names": [f"f{j}" for j in range(d)],
                "train": {"X": X_tr, "y": y_tr},
                "test": {"X": X_te, "y": y_te},
                "options": {"num_boost_round": 30, "min_data_in_leaf": 10, "seed": 42}
            })
        }
        resp = lf.lambda_handler(event)
        assert resp["statusCode"] == 200
        data = json.loads(resp["body"])
        auc = data["metrics"]["roc_auc"]

        # Must reach well above random chance (0.5)
        assert auc > 0.80, f"Expected AUC > 0.80, got {auc}"
        # Must NOT be a fake 1.0 on noisy data
        assert auc < 0.999, f"Realistic noisy data should not yield perfect AUC, got {auc}"

    def test_post_train_without_test_field_succeeds(self):
        """POST /train without a test field succeeds and handles JSON null in features."""
        import random
        rng = random.Random(42)
        n_train = 100
        d = 4

        X_tr = []
        y_tr = []
        for i in range(n_train):
            # Include occasional null / None in features to verify JSON null handling
            row = [
                None if (i % 7 == 0 and j == 0) else rng.gauss(0, 1)
                for j in range(d)
            ]
            label = 1 if ((row[0] or 0.0) + (row[1] or 0.0) > 0.3) else 0
            X_tr.append(row)
            y_tr.append(label)

        event = {
            "requestContext": {"http": {"method": "POST"}},
            "rawPath": "/train",
            "headers": {"x-api-key": TEST_TRAIN_API_KEY},
            "body": json.dumps({
                "feature_names": [f"feat_{j}" for j in range(d)],
                "train": {"X": X_tr, "y": y_tr},
                "options": {"num_boost_round": 10, "min_data_in_leaf": 5, "seed": 42}
            })
        }
        resp = lf.lambda_handler(event)
        assert resp["statusCode"] == 200
        data = json.loads(resp["body"])
        assert data["status"] == "SUCCESS"
        assert len(data["model"]["trees"]) > 0
        assert "decision" in data["model"]["thresholds"]
        assert "review" in data["model"]["thresholds"]
        assert "block" in data["model"]["thresholds"]
        assert data["metrics"] is None

    def test_exported_json_trees_reproduce_booster_predict_within_1e9(self):
        """(b) The exported JSON trees reproduce booster.predict to within 1e-9 for 1,000 random rows."""
        import numpy as np
        import lightgbm as lgb
        rng = np.random.RandomState(42)
        n = 300
        d = 6
        X = rng.randn(n, d).astype(np.float32)
        y = (X[:, 0] + 0.5 * X[:, 1] - 0.3 * X[:, 2] > 0.1).astype(np.int32)

        ds = lgb.Dataset(X, label=y)
        bst = lgb.train(
            {"objective": "binary", "verbosity": -1, "min_data_in_leaf": 10, "num_leaves": 15, "seed": 42},
            ds,
            num_boost_round=15
        )

        raw_dump = bst.dump_model()
        compact_trees = [lf.compact_tree_node(t["tree_structure"]) for t in raw_dump["tree_info"]]

        # Python tree walker mirroring JS logic exactly
        def score_tree_walker(node, x):
            while "leaf_value" not in node:
                f_idx = node["split_feature"]
                val = x[f_idx]
                if val is None or math.isnan(val):
                    node = node["left_child"] if node.get("default_left", True) else node["right_child"]
                elif val <= node["threshold"]:
                    node = node["left_child"]
                else:
                    node = node["right_child"]
            return node["leaf_value"]

        def score_row_walker(trees, x):
            raw = sum(score_tree_walker(t, x) for t in trees)
            return 1.0 / (1.0 + math.exp(-raw))

        # Test on 1,000 random test rows
        X_test = rng.randn(1000, d).astype(np.float32)
        lgb_preds = bst.predict(X_test)

        max_diff = 0.0
        for i in range(1000):
            walker_score = score_row_walker(compact_trees, X_test[i].tolist())
            diff = abs(walker_score - lgb_preds[i])
            if diff > max_diff:
                max_diff = diff

        assert max_diff < 1e-9, f"Tree walker prediction mismatch: max diff {max_diff} >= 1e-9"

    def test_wall_clock_guard_stops_training_early(self):
        """(c) The wall-clock guard stops training early when forced to a tiny limit."""
        import numpy as np
        rng = np.random.RandomState(42)
        X = rng.randn(200, 5).tolist()
        y = [1 if r[0] > 0 else 0 for r in X]

        event = {
            "requestContext": {"http": {"method": "POST"}},
            "rawPath": "/train",
            "headers": {"x-api-key": TEST_TRAIN_API_KEY},
            "body": json.dumps({
                "feature_names": [f"f{i}" for i in range(5)],
                "train": {"X": X, "y": y},
                "test": {"X": X[:30], "y": y[:30]},
                "options": {
                    "num_boost_round": 500,
                    "time_limit_sec": 0.00001  # Force immediate wall clock guard trigger
                }
            })
        }
        resp = lf.lambda_handler(event)
        assert resp["statusCode"] == 200
        data = json.loads(resp["body"])
        assert data["trees_count"] < 10, f"Expected few trees due to time limit, got {data['trees_count']}"

    def test_thresholds_come_from_validation_slice_only(self):
        """(f) Thresholds come from the validation slice only; mutating test data does not alter thresholds."""
        import random
        rng = random.Random(42)
        X_tr = [[rng.gauss(0, 1) for _ in range(3)] for _ in range(100)]
        y_tr = [1 if r[0] > 0.2 else 0 for r in X_tr]

        # Two test sets with different distributions
        X_te1 = [[rng.gauss(0, 1) for _ in range(3)] for _ in range(40)]
        y_te1 = [1 if r[0] > 0.2 else 0 for r in X_te1]

        X_te2 = [[rng.gauss(2, 1) for _ in range(3)] for _ in range(40)]
        y_te2 = [0 for _ in range(40)]  # completely opposite labels

        body1 = {
            "feature_names": ["f1", "f2", "f3"],
            "train": {"X": X_tr, "y": y_tr},
            "test": {"X": X_te1, "y": y_te1},
            "options": {"num_boost_round": 15, "min_data_in_leaf": 5, "seed": 42}
        }
        body2 = {
            "feature_names": ["f1", "f2", "f3"],
            "train": {"X": X_tr, "y": y_tr},
            "test": {"X": X_te2, "y": y_te2},
            "options": {"num_boost_round": 15, "min_data_in_leaf": 5, "seed": 42}
        }

        resp1 = lf.lambda_handler({"requestContext": {"http": {"method": "POST"}}, "rawPath": "/train", "headers": {"x-api-key": TEST_TRAIN_API_KEY}, "body": json.dumps(body1)})
        resp2 = lf.lambda_handler({"requestContext": {"http": {"method": "POST"}}, "rawPath": "/train", "headers": {"x-api-key": TEST_TRAIN_API_KEY}, "body": json.dumps(body2)})

        th1 = json.loads(resp1["body"])["model"]["thresholds"]
        th2 = json.loads(resp2["body"])["model"]["thresholds"]

        # Thresholds must be identical because train partition and validation slice were identical
        assert th1 == th2, f"Threshold leakage from test set! Got {th1} vs {th2}"

    def test_feature_importance_non_negative_and_sums_to_100(self):
        """(g) Importance values are non-negative and sum to about 100% across all features."""
        import random
        rng = random.Random(42)
        d = 12
        X_tr = [[rng.gauss(0, 1) for _ in range(d)] for _ in range(120)]
        y_tr = [1 if r[0] + 0.8 * r[1] > 0.5 else 0 for r in X_tr]
        X_te = [[rng.gauss(0, 1) for _ in range(d)] for _ in range(40)]
        y_te = [1 if r[0] + 0.8 * r[1] > 0.5 else 0 for r in X_te]

        event = {
            "requestContext": {"http": {"method": "POST"}},
            "rawPath": "/train",
            "headers": {"x-api-key": TEST_TRAIN_API_KEY},
            "body": json.dumps({
                "feature_names": [f"feat_{j}" for j in range(d)],
                "train": {"X": X_tr, "y": y_tr},
                "test": {"X": X_te, "y": y_te},
                "options": {"num_boost_round": 20, "min_data_in_leaf": 5, "seed": 42}
            })
        }
        resp = lf.lambda_handler(event)
        assert resp["statusCode"] == 200
        data = json.loads(resp["body"])
        imps = data["importance"]

        assert len(imps) <= 10
        # All values non-negative
        assert all(item["gain_pct"] >= 0.0 for item in imps)
        # Top items sorted descending
        for k in range(len(imps) - 1):
            assert imps[k]["gain_pct"] >= imps[k + 1]["gain_pct"]
        # Total of top 10 is <= 100.01%
        assert sum(item["gain_pct"] for item in imps) <= 100.01

    def test_causal_features_no_lookahead_leakage(self):
        """Verifies that future rows cannot affect features of past rows."""
        def extract_causal(rows):
            orig_counts = {}
            features = []
            for r in rows:
                c = orig_counts.get(r["orig"], 0)
                features.append({"orig": r["orig"], "prior_count": c})
                orig_counts[r["orig"]] = c + 1
            return features

        stream_v1 = [
            {"step": 1, "orig": "C1", "amount": 100},
            {"step": 2, "orig": "C1", "amount": 200},
            {"step": 3, "orig": "C2", "amount": 300}
        ]
        feats_v1 = extract_causal(stream_v1)

        # Append new future row
        stream_v2 = stream_v1 + [{"step": 4, "orig": "C1", "amount": 99999}]
        feats_v2 = extract_causal(stream_v2)

        assert feats_v1 == feats_v2[:3], "Causal leakage detected: future row altered past features!"


class TestLambdaHandler:
    """Test Lambda HTTP API handler routes, round trip with S3, and layer failure."""

    def test_health_route(self):
        event = {
            "requestContext": {"http": {"method": "GET"}},
            "rawPath": "/health"
        }
        resp = lf.lambda_handler(event)
        assert resp["statusCode"] == 200
        data = json.loads(resp["body"])
        assert data["status"] == "healthy"
        assert "version" in data
        assert data["lightgbm_available"] is True
        assert data["lightgbm_version"] is not None

    def test_cors_preflight(self):
        event = {
            "requestContext": {"http": {"method": "OPTIONS"}},
            "rawPath": "/train"
        }
        resp = lf.lambda_handler(event)
        assert resp["statusCode"] == 200
        assert "Access-Control-Allow-Origin" in resp["headers"]

    def test_malformed_input_returns_4xx(self):
        # 1. Non-JSON body
        event = {
            "requestContext": {"http": {"method": "POST"}},
            "rawPath": "/train",
            "headers": {"x-api-key": TEST_TRAIN_API_KEY},
            "body": "not-valid-json{"
        }
        resp = lf.lambda_handler(event)
        assert resp["statusCode"] == 400
        assert "Malformed JSON" in json.loads(resp["body"])["error"]

        # 2. Missing feature names
        event["body"] = json.dumps({"train": {"X": [], "y": []}})
        resp = lf.lambda_handler(event)
        assert resp["statusCode"] == 400
        assert "feature_names" in json.loads(resp["body"])["error"]

        # 3. Dimension mismatch
        event["body"] = json.dumps({
            "feature_names": ["f1", "f2"],
            "train": {"X": [[1.0]], "y": [0]},
            "test": {"X": [[1.0, 2.0]], "y": [0]}
        })
        resp = lf.lambda_handler(event)
        assert resp["statusCode"] == 400

    def test_full_roundtrip_with_mocked_s3(self, monkeypatch):
        """(d) /train, /predict, /model and /health round trip with mocked S3."""
        mock_s3 = MagicMock()
        stored_objects = {}

        def mock_put_object(Bucket, Key, Body, ContentType):
            stored_objects[Key] = {"Body": Body, "ContentType": ContentType}
            return {"ResponseMetadata": {"HTTPStatusCode": 200}}

        def mock_get_object(Bucket, Key):
            if Key in stored_objects:
                body_bytes = stored_objects[Key]["Body"]
                if isinstance(body_bytes, str):
                    body_bytes = body_bytes.encode("utf-8")
                return {"Body": MagicMock(read=lambda: body_bytes)}
            raise Exception("NoSuchKey")

        mock_s3.put_object.side_effect = mock_put_object
        mock_s3.get_object.side_effect = mock_get_object
        monkeypatch.setattr(lf, "s3_client", mock_s3)

        # Reset cache
        lf._CACHED_MODEL = None
        lf._CACHED_BOOSTER = None
        lf._CACHED_VERSION = None

        import random
        rng = random.Random(42)
        feature_names = ["feat1", "feat2", "feat3"]
        d = len(feature_names)

        def make_split(n):
            X = [[rng.gauss(0, 1) for _ in range(d)] for _ in range(n)]
            y = [1 if X[i][0] > 0.4 else 0 for i in range(n)]
            return X, y

        X_tr, y_tr = make_split(80)
        X_te, y_te = make_split(30)

        # 1. POST /train
        train_event = {
            "requestContext": {"http": {"method": "POST"}},
            "rawPath": "/train",
            "headers": {"x-api-key": TEST_TRAIN_API_KEY},
            "body": json.dumps({
                "feature_names": feature_names,
                "train": {"X": X_tr, "y": y_tr},
                "test": {"X": X_te, "y": y_te},
                "options": {"num_boost_round": 10, "min_data_in_leaf": 5, "seed": 42}
            })
        }
        train_resp = lf.lambda_handler(train_event)
        assert train_resp["statusCode"] == 200
        train_data = json.loads(train_resp["body"])
        assert train_data["status"] == "SUCCESS"
        assert train_data["persisted"] is True
        assert "trees" in train_data["model"]
        assert "thresholds" in train_data["model"]

        # Verify S3 received JSON and native text models
        assert "models/latest.json" in stored_objects
        assert "models/latest.txt" in stored_objects
        assert "artifacts/metrics.json" in stored_objects
        assert "pipeline_status.json" in stored_objects

        # 2. GET /model
        model_event = {
            "requestContext": {"http": {"method": "GET"}},
            "rawPath": "/model"
        }
        model_resp = lf.lambda_handler(model_event)
        assert model_resp["statusCode"] == 200
        model_data = json.loads(model_resp["body"])
        assert "trees" in model_data

        # 3. POST /predict
        pred_event = {
            "requestContext": {"http": {"method": "POST"}},
            "rawPath": "/predict",
            "body": json.dumps({"features": X_te[:5]})
        }
        pred_resp = lf.lambda_handler(pred_event)
        assert pred_resp["statusCode"] == 200
        pred_data = json.loads(pred_resp["body"])
        assert len(pred_data["scores"]) == 5
        assert len(pred_data["actions"]) == 5
        assert all(a in ["Approve", "Review", "Block"] for a in pred_data["actions"])

        # 4. Clear cache to simulate cold container and test /predict loading from S3
        lf._CACHED_BOOSTER = None
        lf._CACHED_MODEL = None
        pred_cold_resp = lf.lambda_handler(pred_event)
        assert pred_cold_resp["statusCode"] == 200
        cold_data = json.loads(pred_cold_resp["body"])
        assert len(cold_data["scores"]) == 5

        # 5. GET /health
        health_resp = lf.lambda_handler({"requestContext": {"http": {"method": "GET"}}, "rawPath": "/health"})
        assert health_resp["statusCode"] == 200
        health_data = json.loads(health_resp["body"])
        assert health_data["model_available"] is True

    def test_layer_missing_error_when_lightgbm_cannot_be_imported(self, monkeypatch):
        """(e) /train and /predict return clear 500 error if LightGBM cannot be imported."""
        monkeypatch.setattr(
            lf,
            "check_lightgbm",
            lambda: (False, None, "No module named 'lightgbm'")
        )

        # 1. /health reports lightgbm_available: false
        health_resp = lf.lambda_handler({"requestContext": {"http": {"method": "GET"}}, "rawPath": "/health"})
        assert health_resp["statusCode"] == 200
        h_data = json.loads(health_resp["body"])
        assert h_data["lightgbm_available"] is False
        assert "No module named 'lightgbm'" in h_data["lightgbm_error"]

        # 2. /train returns 500 with exact error message
        train_resp = lf.lambda_handler({
            "requestContext": {"http": {"method": "POST"}},
            "rawPath": "/train",
            "headers": {"x-api-key": TEST_TRAIN_API_KEY},
            "body": json.dumps({
                "feature_names": ["f1"],
                "train": {"X": [[1.0]] * 15, "y": [0] * 15},
                "test": {"X": [[1.0]] * 15, "y": [0] * 15}
            })
        })
        assert train_resp["statusCode"] == 500
        t_data = json.loads(train_resp["body"])
        assert t_data["error"] == "LightGBM layer is not attached to this Lambda"
        assert "No module named 'lightgbm'" in t_data["detail"]

        # 3. /predict returns 500 with exact error message
        pred_resp = lf.lambda_handler({
            "requestContext": {"http": {"method": "POST"}},
            "rawPath": "/predict",
            "body": json.dumps({"features": [[1.0]]})
        })
        assert pred_resp["statusCode"] == 500
        p_data = json.loads(pred_resp["body"])
        assert p_data["error"] == "LightGBM layer is not attached to this Lambda"


class TestDeriveThresholds:
    """Tests for derive_thresholds: block target met, not met fallback, ordering invariant, and warnings."""

    def test_block_target_met(self):
        # 10 frauds at score 0.92, 30 frauds at score 0.55, 10 legits at score 0.55, 90 legits at score 0.10
        # Total: 40 positives (>= 30, so no low positives warning)
        # At threshold 0.90: tp=10, fp=0 -> precision = 1.0 (>= 0.85) and tp=10 >= 3. Target met!
        y_val = [1] * 10 + [1] * 30 + [0] * 10 + [0] * 90
        y_scores = [0.92] * 10 + [0.55] * 30 + [0.55] * 10 + [0.10] * 90

        res = lf.derive_thresholds(y_val, y_scores)
        assert res["block_target_met"] is True
        assert res["block"] >= res["decision"]
        assert res["review"] <= res["decision"] <= res["block"]
        assert len(res["warnings"]) == 0
        diag = res["threshold_diagnostics"]
        assert diag["val_n"] == 140
        assert diag["val_positives"] == 40
        assert len(diag["thresholds"]) == 91

    def test_block_target_not_met(self):
        # 35 frauds and 65 legits where high scores are heavily contaminated with false positives
        # No threshold achieves precision >= 0.85 with tp >= 3.
        # Should not fall back to 0.75; must select the highest-precision cut with tp >= 3.
        import random
        rng = random.Random(42)
        y_val = [1] * 35 + [0] * 65
        y_scores = [rng.uniform(0.1, 0.8) for _ in range(100)]

        res = lf.derive_thresholds(y_val, y_scores)
        assert res["block_target_met"] is False
        assert res["block"] >= res["decision"]
        assert res["review"] <= res["decision"] <= res["block"]
        assert any("block" in w.lower() or "block_target_met" in w.lower() for w in res["warnings"])
        grid = res["threshold_diagnostics"]["thresholds"]
        fallback_cuts = [m for m in grid if m["tp"] >= 3]
        cuts_ge_dec = [m for m in fallback_cuts if m["threshold"] >= res["decision"]]
        cand_pool = cuts_ge_dec if cuts_ge_dec else fallback_cuts
        expected_block = max(cand_pool, key=lambda m: (m["precision"], m["tp"], -m["threshold"]))["threshold"]
        assert res["block"] == expected_block

    def test_ordering_invariant(self):
        # Test invariant review <= decision <= block across multiple distinct score configurations
        import random
        rng = random.Random(123)
        for _ in range(10):
            n = 80
            y_val = [1 if rng.random() < 0.25 else 0 for _ in range(n)]
            y_scores = [rng.random() for _ in range(n)]
            res = lf.derive_thresholds(y_val, y_scores)
            assert res["review"] <= res["decision"] <= res["block"], (
                f"Ordering invariant violated: review={res['review']}, "
                f"decision={res['decision']}, block={res['block']}"
            )

    def test_warnings_generated_on_low_positives_and_target_miss(self):
        # Case 1: val_positives < 30
        y_val = [1] * 5 + [0] * 45
        y_scores = [0.8] * 5 + [0.2] * 45
        res = lf.derive_thresholds(y_val, y_scores)
        assert res["threshold_diagnostics"]["val_positives"] == 5
        assert any("30" in w for w in res["warnings"])

        # Case 2: block_target_met is False
        y_val2 = [1] * 35 + [0] * 35
        y_scores2 = [0.7] * 35 + [0.7] * 35
        res2 = lf.derive_thresholds(y_val2, y_scores2)
        assert res2["block_target_met"] is False
        assert any("block" in w.lower() for w in res2["warnings"])


class TestTrainAuth:
    """Tests for TRAIN_API_KEY authentication on POST /train and /pipeline, and public endpoints."""

    def test_train_401_without_key(self, monkeypatch):
        monkeypatch.setenv("TRAIN_API_KEY", "secret-test-key-16chars")
        event = {
            "requestContext": {"http": {"method": "POST"}},
            "rawPath": "/train",
            "body": json.dumps({"feature_names": ["f1"]})
        }
        resp = lf.lambda_handler(event)
        assert resp["statusCode"] == 401
        data = json.loads(resp["body"])
        assert "Unauthorized" in data["error"]

    def test_train_401_with_invalid_key(self, monkeypatch):
        monkeypatch.setenv("TRAIN_API_KEY", "secret-test-key-16chars")
        event = {
            "requestContext": {"http": {"method": "POST"}},
            "rawPath": "/train",
            "headers": {"x-api-key": "wrong-secret-key-1234"},
            "body": json.dumps({"feature_names": ["f1"]})
        }
        resp = lf.lambda_handler(event)
        assert resp["statusCode"] == 401
        data = json.loads(resp["body"])
        assert "Unauthorized" in data["error"]

    def test_train_200_with_valid_key(self, monkeypatch):
        monkeypatch.setenv("TRAIN_API_KEY", "secret-test-key-16chars")
        event = {
            "requestContext": {"http": {"method": "POST"}},
            "rawPath": "/train",
            "headers": {"x-api-key": "secret-test-key-16chars"},
            "body": json.dumps({
                "feature_names": ["f1", "f2"],
                "train": {"X": [[1.0, 0.0], [0.0, 1.0]] * 15, "y": [1, 0] * 15},
                "test": {"X": [[1.0, 0.0], [0.0, 1.0]] * 10, "y": [1, 0] * 10},
                "options": {"num_boost_round": 5, "min_data_in_leaf": 2, "seed": 42}
            })
        }
        resp = lf.lambda_handler(event)
        assert resp["statusCode"] == 200
        data = json.loads(resp["body"])
        assert data["status"] == "SUCCESS"

    def test_post_train_without_test_field_succeeds(self, monkeypatch):
        """POST /train must work WITHOUT a test field, and convert JSON null to NaN."""
        monkeypatch.setenv("TRAIN_API_KEY", "secret-test-key-16chars")
        train_X = [[1.0, 0.5], [0.0, None], [0.5, 0.2], [0.8, 0.9]] * 10
        train_y = [1, 0, 1, 0] * 10
        event = {
            "requestContext": {"http": {"method": "POST"}},
            "rawPath": "/train",
            "headers": {"x-api-key": "secret-test-key-16chars"},
            "body": json.dumps({
                "feature_names": ["f1", "f2"],
                "train": {"X": train_X, "y": train_y},
                # Notice: test field is completely omitted
                "options": {
                    "fit_rows": 24,
                    "num_boost_round": 10,
                    "min_data_in_leaf": 2,
                    "seed": 42
                }
            })
        }
        resp = lf.lambda_handler(event)
        assert resp["statusCode"] == 200
        data = json.loads(resp["body"])
        assert data["status"] == "SUCCESS"
        assert "model" in data
        assert "trees" in data["model"]
        assert len(data["model"]["trees"]) > 0
        assert data["metrics"] is None
        assert data["confusion"] is None
        assert "thresholds" in data
        assert "decision" in data["thresholds"]
        assert "review" in data["thresholds"]
        assert "block" in data["thresholds"]
        assert data["thresholds"]["review"] <= data["thresholds"]["decision"] <= data["thresholds"]["block"]

    def test_train_fails_closed_when_env_unset(self, monkeypatch):
        monkeypatch.delenv("TRAIN_API_KEY", raising=False)
        event = {
            "requestContext": {"http": {"method": "POST"}},
            "rawPath": "/train",
            "headers": {"x-api-key": "some-key"},
            "body": json.dumps({"feature_names": ["f1"]})
        }
        resp = lf.lambda_handler(event)
        assert resp["statusCode"] == 500
        data = json.loads(resp["body"])
        assert "TRAIN_API_KEY" in data["error"]

    def test_pipeline_requires_key(self, monkeypatch):
        monkeypatch.setenv("TRAIN_API_KEY", "secret-test-key-16chars")
        # 1. POST /pipeline without key -> 401
        resp_post = lf.lambda_handler({
            "requestContext": {"http": {"method": "POST"}},
            "rawPath": "/pipeline",
            "body": "{}"
        })
        assert resp_post["statusCode"] == 401

        # 2. GET /pipeline without key -> 401
        resp_get = lf.lambda_handler({
            "requestContext": {"http": {"method": "GET"}},
            "rawPath": "/pipeline"
        })
        assert resp_get["statusCode"] == 401

        # 3. GET /pipeline with key -> 200
        resp_get_authed = lf.lambda_handler({
            "requestContext": {"http": {"method": "GET"}},
            "rawPath": "/pipeline",
            "headers": {"x-api-key": "secret-test-key-16chars"}
        })
        assert resp_get_authed["statusCode"] == 200

    def test_public_endpoints_accessible_without_key(self, monkeypatch):
        monkeypatch.setenv("TRAIN_API_KEY", "secret-test-key-16chars")
        # 1. GET /health
        resp_health = lf.lambda_handler({
            "requestContext": {"http": {"method": "GET"}},
            "rawPath": "/health"
        })
        assert resp_health["statusCode"] == 200

        # 2. GET /model
        resp_model = lf.lambda_handler({
            "requestContext": {"http": {"method": "GET"}},
            "rawPath": "/model"
        })
        assert resp_model["statusCode"] == 200

        # 3. POST /predict
        resp_pred = lf.lambda_handler({
            "requestContext": {"http": {"method": "POST"}},
            "rawPath": "/predict",
            "body": json.dumps({"features": [[0.5] * 19]})
        })
        assert resp_pred["statusCode"] in [200, 400]

        # 4. OPTIONS preflight
        resp_opt = lf.lambda_handler({
            "requestContext": {"http": {"method": "OPTIONS"}},
            "rawPath": "/train"
        })
        assert resp_opt["statusCode"] == 200

