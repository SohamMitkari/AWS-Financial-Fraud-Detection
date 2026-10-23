#!/usr/bin/env python3
"""
Dataset preparation script for Elliptic Bitcoin Transaction Dataset.

Raw files:
  - dataset/elliptic/elliptic_txs_features.csv
  - dataset/elliptic/elliptic_txs_classes.csv
Standard format output: dataset/elliptic/elliptic_std.csv
Sample output: tests/fixtures/real/elliptic_sample300.csv
"""

import os
import sys
from pathlib import Path
import pandas as pd


def get_raw_paths(repo_root: Path):
    features_candidates = [
        repo_root / "dataset" / "elliptic" / "elliptic_txs_features.csv",
        repo_root / "dataset" / "elliptic_txs_features.csv",
    ]
    classes_candidates = [
        repo_root / "dataset" / "elliptic" / "elliptic_txs_classes.csv",
        repo_root / "dataset" / "elliptic_txs_classes.csv",
    ]

    feat_path = next((c for c in features_candidates if c.exists()), features_candidates[0])
    cls_path = next((c for c in classes_candidates if c.exists()), classes_candidates[0])
    return feat_path, cls_path


def prep_elliptic() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    feat_path, cls_path = get_raw_paths(repo_root)

    if not feat_path.exists():
        sys.exit(f"Raw file missing: {feat_path} not found. Please place elliptic_txs_features.csv in dataset/elliptic/.")
    if not cls_path.exists():
        sys.exit(f"Raw file missing: {cls_path} not found. Please place elliptic_txs_classes.csv in dataset/elliptic/.")

    out_dir = repo_root / "dataset" / "elliptic"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "elliptic_std.csv"

    fixture_dir = repo_root / "tests" / "fixtures" / "real"
    fixture_dir.mkdir(parents=True, exist_ok=True)
    fixture_path = fixture_dir / "elliptic_sample300.csv"

    # Read classes and filter unknown
    classes_df = pd.read_csv(cls_path)
    classes_df.columns = [c.strip() for c in classes_df.columns]
    # drop class "unknown"; illicit ("1")=1, licit ("2")=0
    classes_df = classes_df[classes_df["class"].astype(str).isin(["1", "2"])].copy()
    classes_df["label"] = (classes_df["class"].astype(str) == "1").astype(int)
    classes_map = dict(zip(classes_df["txId"], classes_df["label"]))

    # elliptic_txs_features.csv has NO header (txId, time step, 165 features)
    feature_cols = [f"f{i}" for i in range(1, 166)]
    col_names = ["txId", "order"] + feature_cols

    chunks = []
    for chunk in pd.read_csv(feat_path, header=None, names=col_names, chunksize=50_000):
        # Join to classes on txId
        chunk = chunk[chunk["txId"].isin(classes_map)].copy()
        chunk["label"] = chunk["txId"].map(classes_map).astype(int)
        # Drop txId
        chunk = chunk.drop(columns=["txId"])
        chunks.append(chunk)

    df = pd.concat(chunks, ignore_index=True)
    # Ensure columns order: order, f1..f165, label
    df = df[["order"] + feature_cols + ["label"]]
    # Sort by order
    df = df.sort_values(by="order", kind="stable").reset_index(drop=True)

    # Write standard dataset
    df.to_csv(out_path, index=False, na_rep="")

    total_rows = len(df)
    positives = int((df["label"] == 1).sum())
    prevalence = positives / total_rows if total_rows > 0 else 0.0

    print(f"[elliptic] Rows: {total_rows}, Positives: {positives}, Prevalence: {prevalence:.6f} ({prevalence * 100:.4f}%)")

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
    print(f"[elliptic] Wrote 300-row sample to {fixture_path} (positives: {int((sample['label'] == 1).sum())})")


if __name__ == "__main__":
    prep_elliptic()
