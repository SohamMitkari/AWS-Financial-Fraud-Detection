"""
Cross-Language Parity Test: Python features.py vs JavaScript frontend feature extraction.

Asserts that Python causal feature vectors extracted from generated transaction records
match the JavaScript implementation in frontend/index.html within 1e-6 across all 19 features.
"""

import os
import sys
import json
import subprocess
import pytest

# Ensure backend root is on sys.path
backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

import features as py_feat

TEMP_JSON_PATH = os.path.join(os.path.dirname(__file__), "temp_feature_parity.json")
DUMP_SCRIPT_PATH = os.path.join(os.path.dirname(__file__), "dump_rows_for_parity.js")


@pytest.fixture(scope="module")
def parity_data():
    """Ensure temp JSON file with 2,000 JS-evaluated rows exists."""
    if not os.path.exists(TEMP_JSON_PATH):
        res = subprocess.run(
            ["node", DUMP_SCRIPT_PATH, TEMP_JSON_PATH],
            capture_output=True,
            text=True,
            check=True,
        )
    with open(TEMP_JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def test_feature_names_parity(parity_data):
    """Assert feature names and ordering match JavaScript FEATURE_NAMES."""
    js_feature_names = parity_data["feature_names"]
    assert py_feat.FEATURE_NAMES == js_feature_names, (
        f"Feature names mismatch:\nPython: {py_feat.FEATURE_NAMES}\nJS: {js_feature_names}"
    )
    assert len(py_feat.FEATURE_NAMES) == 19


def test_streaming_and_batch_feature_parity(parity_data):
    """Assert Python streaming causal features match JS vectors within 1e-6."""
    rows = parity_data["rows"]
    js_vectors = parity_data["js_vectors"]

    assert len(rows) == 2000, f"Expected 2,000 rows, got {len(rows)}"
    assert len(js_vectors) == 2000, f"Expected 2,000 JS vectors, got {len(js_vectors)}"

    # Test batch extraction
    py_vectors = py_feat.extract_causal_features(rows)
    assert len(py_vectors) == len(rows)

    max_diff = 0.0
    mismatches = []

    for i in range(len(rows)):
        py_row = py_vectors[i]
        js_row = js_vectors[i]
        assert len(py_row) == 19
        assert len(js_row) == 19

        for j in range(19):
            diff = abs(py_row[j] - js_row[j])
            if diff > max_diff:
                max_diff = diff
            if diff > 1e-6:
                mismatches.append(
                    f"Row {i}, Feature {j} ({py_feat.FEATURE_NAMES[j]}): "
                    f"Python={py_row[j]}, JS={js_row[j]}, diff={diff:.8e}"
                )

    assert not mismatches, (
        f"Found {len(mismatches)} feature parity mismatches > 1e-6 (max diff: {max_diff:.8e}):\n"
        + "\n".join(mismatches[:10])
    )
    assert max_diff <= 1e-6, f"Max difference {max_diff:.8e} exceeded 1e-6"


def test_stream_generator_parity(parity_data):
    """Assert stream_causal_features generator yields identical vectors."""
    rows = parity_data["rows"][:100]
    js_vectors = parity_data["js_vectors"][:100]

    gen = py_feat.stream_causal_features(rows)
    for i, py_vec in enumerate(gen):
        js_vec = js_vectors[i]
        for j in range(19):
            assert abs(py_vec[j] - js_vec[j]) <= 1e-6


def test_expected_balances_parity():
    """Verify expected_balances logic for both CASH_IN and non-CASH_IN."""
    # CASH_IN
    new_o, new_d = py_feat.expected_balances("CASH_IN", 1000.0, 500.0, 200.0)
    assert new_o == 1200.0
    assert new_d == 300.0

    # TRANSFER / PAYMENT
    new_o, new_d = py_feat.expected_balances("TRANSFER", 1000.0, 500.0, 200.0)
    assert new_o == 800.0
    assert new_d == 700.0

    # Overdraft clamp to 0
    new_o, new_d = py_feat.expected_balances("PAYMENT", 100.0, 50.0, 500.0)
    assert new_o == 0.0
    assert new_d == 550.0
