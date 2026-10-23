#!/usr/bin/env python3
"""
Dataset preparation script for Taiwan Default of Credit Card Clients Dataset.

Raw file: dataset/taiwan/UCI_Credit_Card.csv
Standard format output: dataset/taiwan/taiwan_std.csv
Sample output: tests/fixtures/real/taiwan_sample300.csv
"""

import os
import sys
from pathlib import Path
import pandas as pd


def get_raw_path(repo_root: Path) -> Path:
    candidates = [
        repo_root / "dataset" / "taiwan" / "UCI_Credit_Card.csv",
        repo_root / "dataset" / "UCI_Credit_Card.csv",
    ]
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]


def prep_taiwan() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    raw_path = get_raw_path(repo_root)

    if not raw_path.exists():
        sys.exit(f"Raw file missing: {raw_path} not found. Please place UCI_Credit_Card.csv in dataset/taiwan/.")

    out_dir = repo_root / "dataset" / "taiwan"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "taiwan_std.csv"

    fixture_dir = repo_root / "tests" / "fixtures" / "real"
    fixture_dir.mkdir(parents=True, exist_ok=True)
    fixture_path = fixture_dir / "taiwan_sample300.csv"

    df = pd.read_csv(raw_path)

    # Strip whitespace from column names
    df.columns = [c.strip() for c in df.columns]

    # drop ID
    if "ID" in df.columns:
        df = df.drop(columns=["ID"])

    # label=default.payment.next.month; no order column
    df = df.rename(columns={"default.payment.next.month": "label"})

    # Ensure label is int 0/1
    df["label"] = df["label"].astype(int)

    # Place feature columns then label
    feature_cols = [c for c in df.columns if c != "label"]
    df = df[feature_cols + ["label"]]

    # Write standard dataset
    df.to_csv(out_path, index=False, na_rep="")

    total_rows = len(df)
    positives = int((df["label"] == 1).sum())
    prevalence = positives / total_rows if total_rows > 0 else 0.0

    print(f"[taiwan] Rows: {total_rows}, Positives: {positives}, Prevalence: {prevalence:.6f} ({prevalence * 100:.4f}%)")

    # 300-row sample: seed 42, stratified so it contains at least 30 positives; no order column
    pos_df = df[df["label"] == 1]
    neg_df = df[df["label"] == 0]

    natural_pos = int(round(300 * prevalence))
    n_pos = max(30, natural_pos)
    n_pos = min(n_pos, len(pos_df))
    n_neg = 300 - n_pos

    sample_pos = pos_df.sample(n=n_pos, random_state=42)
    sample_neg = neg_df.sample(n=n_neg, random_state=42)

    sample = pd.concat([sample_pos, sample_neg])
    # No order column: keep stable row order by original index
    sample = sample.sort_index().reset_index(drop=True)

    sample.to_csv(fixture_path, index=False, na_rep="")
    print(f"[taiwan] Wrote 300-row sample to {fixture_path} (positives: {int((sample['label'] == 1).sum())})")


if __name__ == "__main__":
    prep_taiwan()
