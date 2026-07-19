"""
pipeline.py

Imports all feature modules and computes features for every move in the dataset (~29 features)

Usage:
  uv run src/features/pipeline.py
  uv run src/features/pipeline.py --processed-dir data/processed --features-dir data/features
"""

import sys
import chess
import polars as pl
import argparse
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from position import compute_position_features
from pawn_structure import compute_pawn_structure_features
from king_safety import compute_king_safety_features
from move_features import compute_move_features
from time_features import compute_time_features





def compute_all_position_features(fens: list[str], ucis: list[str]) -> dict[str, list]:
    """
    Compute all position, pawn, king safety, and move features for every row.
    """
    results: dict[str, list] = {}

    for fen, uci in tqdm(zip(fens, ucis), total=len(fens), desc="Position features"):
        try:
            board = chess.Board(fen)
            move  = chess.Move.from_uci(uci)

            row = {}
            row.update(compute_position_features(board))
            row.update(compute_pawn_structure_features(board))
            row.update(compute_king_safety_features(board))
            row.update(compute_move_features(board, move))

        except Exception:
        
            row = {k: 0 for k in results} if results else {}

        for key, val in row.items():
            if key not in results:
                results[key] = []
            results[key].append(val)

    return results



def build_features(processed_dir: str, features_dir: str) -> None:
    """
    1. Compute time/window features from Parquet via DuckDB SQL
    2. Compute position/pawn/king/move features via python-chess
    3. Join both sets of features and write final Parquet
    """
    output_path = f"{features_dir}/features.parquet"

    print("Computing time features via DuckDB …")
    df = compute_time_features(processed_dir)
    print(f"  {len(df):,} rows loaded")

    print("Computing position/pawn/king/move features …")
    pos_features = compute_all_position_features(
        df["fen_before"].to_list(),
        df["move_uci"].to_list(),
    )

    df = df.with_columns([
        pl.Series(name, values)
        for name, values in pos_features.items()
    ])

    Path(features_dir).mkdir(parents=True, exist_ok=True)
    df.write_parquet(output_path, compression="zstd")

    meta_cols  = {"game_id", "move_number", "color", "fen_before", "move_uci", "time_spent"}
    feat_cols  = [c for c in df.columns if c not in meta_cols]
    print(f"\nDone. {len(df):,} rows")
    print(f"Feature columns ({len(feat_cols)}): {feat_cols}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Feature engineering pipeline")
    parser.add_argument("--processed-dir", type=str, default="data/processed")
    parser.add_argument("--features-dir",  type=str, default="data/features")
    args = parser.parse_args()
    build_features(args.processed_dir, args.features_dir)