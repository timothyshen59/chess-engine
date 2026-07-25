"""
pipeline.py

Computes time/window features via DuckDB SQL and writes to Parquet.
Position features removed — transformer uses board tensors directly.

Usage:
  uv run src/features/pipeline.py
  uv run src/features/pipeline.py --processed-dir data/processed --features-dir data/features
"""

import sys
import polars as pl
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from time_features import compute_time_features


def build_features(processed_dir: str, features_dir: str) -> None:
    output_path = f"{features_dir}/features.parquet"

    print("Computing time features via DuckDB ...")
    df = compute_time_features(processed_dir)
    print(f"  {len(df):,} rows loaded")

    Path(features_dir).mkdir(parents=True, exist_ok=True)
    df.write_parquet(output_path, compression="zstd")

    print(f"\nDone. {len(df):,} rows")
    print(f"Columns: {df.columns}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Feature engineering pipeline")
    parser.add_argument("--processed-dir", type=str, default="data/processed")
    parser.add_argument("--features-dir",  type=str, default="data/features")
    args = parser.parse_args()
    build_features(args.processed_dir, args.features_dir)