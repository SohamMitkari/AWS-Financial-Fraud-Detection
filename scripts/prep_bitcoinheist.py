#!/usr/bin/env python3
"""
Dataset preparation script for BitcoinHeist Ransomware Dataset.

Raw file: dataset/bitcoinheist/BitcoinHeistData.csv
Standard format output: dataset/bitcoinheist/bitcoinheist_std.csv
Sample output: tests/fixtures/real/bitcoinheist_sample300.csv
"""

import os
import sys
from pathlib import Path
import pandas as pd


def get_raw_path(repo_root: Path) -> Path:
    candidates = [
        repo_root / "dataset" / "bitcoinheist" / "BitcoinHeistData.csv",
        repo_root / "dataset" / "BitcoinHeistData.csv",
    ]
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]


def prep_bitcoinheist() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    raw_path = get_raw_path(repo_root)

    if not raw_path.exists():
        sys.exit(f"Raw file missing: {raw_path} not found. Please place BitcoinHeistData.csv in dataset/bitcoinheist/.")

    out_dir = repo_root / "dataset" / "bitcoinheist"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "bitcoinheist_std.csv"

    fixture_dir = repo_root / "tests" / "fixtures" / "real"
    fixture_dir.mkdir(parents=True, exist_ok=True)
    fixture_path = fixture_dir / "bitcoinheist_sample300.csv"

    feature_cols = ["length", "weight", "count", "looped", "neighbors", "income"]
    chunks = []

    # Chunked reading where the file is large
    for chunk in pd.read_csv(raw_path, chunksize=200_000):
        # Strip whitespace from column names
        chunk.columns = [c.strip() for c in chunk.columns]

        # order = year * 366 + day
        order_series = chunk["year"] * 366 + chunk["day"]

        # label = 0 if label == "white" else 1
        label_series = (chunk["label"] != "white").astype(int)

        transformed = pd.DataFrame({
            "order": order_series,
            "length": chunk["length"],
            "weight": chunk["weight"],
            "count": chunk["count"],
            "looped": chunk["looped"],
            "neighbors": chunk["neighbors"],
            "income": chunk["income"],
            "label": label_series,
        })
        chunks.append(transformed)

    full_df = pd.concat(chunks, ignore_index=True)

    # uniform 10% sample with fixed seed 42 (natural prevalence)
    df = full_df.sample(frac=0.10, random_state=42)
    # Sort by order
    df = df.sort_values(by="order", kind="stable").reset_index(drop=True)

    # Write standard dataset
    df.to_csv(out_path, index=False, na_rep="")

    total_rows = len(df)
    positives = int((df["label"] == 1).sum())
    prevalence = positives / total_rows if total_rows > 0 else 0.0

    print(f"[bitcoinheist] Rows: {total_rows}, Positives: {positives}, Prevalence: {prevalence:.6f} ({prevalence * 100:.4f}%)")

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
    print(f"[bitcoinheist] Wrote 300-row sample to {fixture_path} (positives: {int((sample['label'] == 1).sum())})")


if __name__ == "__main__":
    prep_bitcoinheist()
