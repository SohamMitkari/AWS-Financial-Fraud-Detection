"""
Strictly Causal Feature Engineering (Python Port).

Exact parity with `buildFeatureVector`, `expectedBalances`, and `extractCausalFeatures`
defined in `frontend/index.html`.
Enforces zero lookahead leakage by maintaining causal customer history updated
strictly AFTER evaluating each transaction.
"""

import math
from typing import Dict, Any, List, Optional, Tuple, Iterator

FEATURE_NAMES: List[str] = [
    "amount_log1p",
    "balance_orig_diff",
    "balance_dest_diff",
    "orig_balance_error",
    "dest_balance_error",
    "orig_drained",
    "hour_of_day",
    "type_TRANSFER",
    "type_CASH_OUT",
    "type_PAYMENT",
    "type_DEBIT",
    "type_CASH_IN",
    "orig_prior_count",
    "orig_prior_mean_amount",
    "amount_vs_orig_mean",
    "dest_prior_count",
    "dest_prior_max_amount",
    "is_new_dest",
    "orig_txn_last_24_steps",
]


def expected_balances(txn_type: str, old_o: Any, old_d: Any, amount: Any) -> Tuple[float, float]:
    """
    Compute expected new originator and destination balances given transaction type,
    initial balances, and transaction amount. Matches Stage 2 and buildFeatureVector in frontend.
    """
    try:
        amt = float(amount or 0.0)
    except (ValueError, TypeError):
        amt = 0.0
    safe_amt = abs(amt)

    try:
        val_o = float(old_o) if old_o is not None and math.isfinite(float(old_o)) else 0.0
    except (ValueError, TypeError):
        val_o = 0.0

    try:
        val_d = float(old_d) if old_d is not None and math.isfinite(float(old_d)) else 0.0
    except (ValueError, TypeError):
        val_d = 0.0

    safe_old_o = max(0.0, val_o)
    safe_old_d = max(0.0, val_d)

    is_cash_in = str(txn_type or "").upper() == "CASH_IN"
    expected_new_o = (safe_old_o + safe_amt) if is_cash_in else max(0.0, safe_old_o - safe_amt)
    expected_new_d = max(0.0, safe_old_d - safe_amt) if is_cash_in else (safe_old_d + safe_amt)

    return expected_new_o, expected_new_d


def build_feature_vector(
    record: Dict[str, Any],
    orig_state: Optional[Dict[str, Any]] = None,
    dest_state: Optional[Dict[str, Any]] = None,
) -> List[float]:
    """
    Extract 19-element causal feature vector from a single record and prior account states.
    Matches formulas, precision (toFixed(4)), and order in frontend/index.html.
    """
    try:
        amount = float(record.get("amount") or 0.0)
    except (ValueError, TypeError):
        amount = 0.0

    try:
        old_o = float(record.get("oldbalanceOrg") or 0.0)
    except (ValueError, TypeError):
        old_o = 0.0

    try:
        new_o = float(record.get("newbalanceOrig") or 0.0)
    except (ValueError, TypeError):
        new_o = 0.0

    try:
        old_d = float(record.get("oldbalanceDest") or 0.0)
    except (ValueError, TypeError):
        old_d = 0.0

    try:
        new_d = float(record.get("newbalanceDest") or 0.0)
    except (ValueError, TypeError):
        new_d = 0.0

    try:
        step = int(record.get("step") or 1)
    except (ValueError, TypeError):
        step = 1

    txn_type = str(record.get("type") or "").upper()

    amount_log1p = math.log1p(max(0.0, amount))
    balance_orig_diff = old_o - new_o
    balance_dest_diff = new_d - old_d

    exp_new_o, exp_new_d = expected_balances(txn_type, old_o, old_d, amount)
    orig_balance_error = 1.0 if abs(exp_new_o - new_o) > 0.05 else 0.0
    dest_balance_error = 1.0 if abs(exp_new_d - new_d) > 0.05 else 0.0

    orig_drained = 1.0 if (old_o > 0 and new_o == 0) else 0.0
    hour_of_day = float(step % 24)

    type_transfer = 1.0 if txn_type == "TRANSFER" else 0.0
    type_cash_out = 1.0 if txn_type == "CASH_OUT" else 0.0
    type_payment = 1.0 if txn_type == "PAYMENT" else 0.0
    type_debit = 1.0 if txn_type == "DEBIT" else 0.0
    type_cash_in = 1.0 if txn_type == "CASH_IN" else 0.0

    o_state = orig_state or {"count": 0, "total": 0.0, "steps": []}
    orig_prior_count = float(o_state.get("count", 0))
    o_total = float(o_state.get("total", 0.0))
    orig_prior_mean_amount = (o_total / orig_prior_count) if orig_prior_count > 0 else 0.0
    amount_vs_orig_mean = (amount / orig_prior_mean_amount) if orig_prior_mean_amount > 0 else 1.0

    d_state = dest_state or {"count": 0, "max": 0.0}
    dest_prior_count = float(d_state.get("count", 0))
    dest_prior_max_amount = float(d_state.get("max", 0.0))
    is_new_dest = 1.0 if dest_prior_count == 0 else 0.0

    o_steps = o_state.get("steps") or []
    orig_txn_last_24_steps = float(sum(1 for s in o_steps if s > step - 24))

    return [
        round(amount_log1p, 4),
        round(balance_orig_diff, 4),
        round(balance_dest_diff, 4),
        orig_balance_error,
        dest_balance_error,
        orig_drained,
        hour_of_day,
        type_transfer,
        type_cash_out,
        type_payment,
        type_debit,
        type_cash_in,
        orig_prior_count,
        round(orig_prior_mean_amount, 4),
        round(amount_vs_orig_mean, 4),
        dest_prior_count,
        round(dest_prior_max_amount, 4),
        is_new_dest,
        orig_txn_last_24_steps,
    ]


class CausalFeatureStream:
    """
    Streaming causal feature engine for online transaction ingestion.
    Maintains originator and destination transaction history without lookahead leakage.
    """

    def __init__(self):
        self.orig_hist: Dict[str, Dict[str, Any]] = {}
        self.dest_hist: Dict[str, Dict[str, Any]] = {}

    def reset(self) -> None:
        """Clear all historical account states."""
        self.orig_hist.clear()
        self.dest_hist.clear()

    def process_row(self, record: Dict[str, Any]) -> List[float]:
        """
        Process a single transaction dict row and return its 19-element feature vector.
        Updates causal history strictly after evaluating features for the row.
        """
        orig = str(record.get("nameOrig") or "")
        dest = str(record.get("nameDest") or "")
        try:
            amount = float(record.get("amount") or 0.0)
        except (ValueError, TypeError):
            amount = 0.0
        try:
            step = int(record.get("step") or 1)
        except (ValueError, TypeError):
            step = 1

        o_state = self.orig_hist.get(orig, {"count": 0, "total": 0.0, "steps": []})
        d_state = self.dest_hist.get(dest, {"count": 0, "max": 0.0})

        vec = build_feature_vector(record, o_state, d_state)

        # Update historical state strictly AFTER evaluating row
        new_steps = [s for s in o_state["steps"] if s > step - 48]
        new_steps.append(step)
        self.orig_hist[orig] = {
            "count": o_state["count"] + 1,
            "total": o_state["total"] + amount,
            "steps": new_steps,
        }
        self.dest_hist[dest] = {
            "count": d_state["count"] + 1,
            "max": max(d_state["max"], amount),
        }

        return vec


def stream_causal_features(
    records: Iterator[Dict[str, Any]],
    stream: Optional[CausalFeatureStream] = None,
) -> Iterator[List[float]]:
    """
    Streaming generator over dict rows yielding 19-element causal feature vectors.
    """
    engine = stream if stream is not None else CausalFeatureStream()
    for record in records:
        yield engine.process_row(record)


def extract_causal_features(
    records: List[Dict[str, Any]],
    stream: Optional[CausalFeatureStream] = None,
) -> List[List[float]]:
    """
    Convenience function extracting causal feature vectors for a list of records.
    """
    return list(stream_causal_features(records, stream=stream))
