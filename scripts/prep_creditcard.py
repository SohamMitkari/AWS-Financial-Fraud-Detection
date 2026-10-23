#!/usr/bin/env python3
"""
Dataset preparation script for Credit Card Fraud Detection (Kaggle).

Raw file: dataset/creditcard/creditcard.csv
Standard format output: dataset/creditcard/creditcard_std.csv
Sample output: tests/fixtures/real/creditcard_sample300.csv
"""

import os
import sys
from pathlib import Path
import pandas as pd


def get_raw_path(repo_root: Path) -> Path:
    candidates = [
        repo_root / "dataset" / "creditcard" / "creditcard.csv",
        repo_root / "dataset" / "creditcard.csv",
    ]
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]


def prep_creditcard() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    raw_path = get_raw_path(repo_root)

    if not raw_path.exists():
        sys.exit(f"Raw file missing: {raw_path} not found. Please place creditcard.csv in dataset/creditcard/.")

    out_dir = repo_root / "dataset" / "creditcard"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "creditcard_std.csv"

    fixture_dir = repo_root / "tests" / "fixtures" / "real"
    fixture_dir.mkdir(parents=True, exist_ok=True)
    fixture_path = fixture_dir / "creditcard_sample300.csv"

    # Read in chunks where the file is large
    chunks = []
    for chunk in pd.read_csv(raw_path, chunksize=100_000):
        # Strip whitespace from column names
        chunk.columns = [c.strip() for c in chunk.columns]
        # label=Class, order=Time, features V1-V28 and Amount
        chunk = chunk.rename(columns={"Time": "order", "Class": "label"})
        feature_cols = [f"V{i}" for i in range(1, 29)] + ["Amount"]
        chunk = chunk[["order"] + feature_cols + ["label"]]
        chunks.append(chunk)

    df = pd.concat(chunks, ignore_index=True)
    # Ensure sorted by order
    df = df.sort_values(by="order", kind="stable").reset_index(drop=True)

    # Ensure label is integer 0/1
    df["label"] = df["label"].astype(int)

    # Write standard dataset
    df.to_csv(out_path, index=False, na_rep="")

    total_rows = len(df)
    positives = int((df["label"] == 1).sum())
    prevalence = positives / total_rows if total_rows > 0 else 0.0

    print(f"[creditcard] Rows: {total_rows}, Positives: {positives}, Prevalence: {prevalence:.6f} ({prevalence * 100:.4f}%)")

    # 300-row sample: seed 42, stratified so it contains at least 30 positives; sorted by order
    pos_df = df[df["label"] == 1]
    neg_df = df[df["label"] == 0]

    natural_pos = int(round(300 * prevalence))
    n_pos = max(30, natural_pos)
    n_pos = min(n_pos, len(pos_df))
    n_neg = 300 - n_pos

    sample_pos = pos_df.sample(n=n_pos, random_state=42)
    sample_neg = neg_df.sample(n=n_neg, random_state=42)

    sample = pd.concat([sample_pos, sample_neg])
    sample = sample.sort_values(by="order", kind="stable").reset_index(drop=True)

    sample.to_csv(fixture_path, index=False, na_rep="")
    print(f"[creditcard] Wrote 300-row sample to {fixture_path} (positives: {int((sample['label'] == 1).sum())})")


if __name__ == "__main__":
    prep_creditcard()
