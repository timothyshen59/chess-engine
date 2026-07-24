"""
Convert features parquet to WebDataset shards 

Example:
    uv run python -m src.training.preprocessing \
        --features-path data/features/features.parquet \
        --output-dir data/training_store/v1 \
        --games-per-shard 1000 \
        --max-seq-len 128
"""

from __future__ import annotations 

import argparse
import hashlib
import json
import os
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import chess
import numpy as np
import polars as pl
import torch
import webdataset as wds

#Constants 
BOARD_CHANNELS = 17
DEFAULT_MAX_SEQ_LEN = 128
N_TIME_FEATURES = 3

PIECE_CHANNEL = {
    (chess.PAWN, chess.WHITE): 0,
    (chess.KNIGHT, chess.WHITE): 1,
    (chess.BISHOP, chess.WHITE): 2,
    (chess.ROOK, chess.WHITE): 3,
    (chess.QUEEN, chess.WHITE): 4,
    (chess.KING, chess.WHITE): 5,
    (chess.PAWN, chess.BLACK): 6,
    (chess.KNIGHT, chess.BLACK): 7,
    (chess.BISHOP, chess.BLACK): 8,
    (chess.ROOK, chess.BLACK): 9,
    (chess.QUEEN, chess.BLACK): 10,
    (chess.KING, chess.BLACK): 11,
}

REQUIRED_COLUMNS = {
    "game_id",
    "move_number",
    "fen_before",
    "cp_loss",
    "white_elo",
    "black_elo",
}

TIME_FEATURE_COLUMNS = (
    "time_spent_norm",
    "time_pressure_flag",
    "avg_time_spent_5",
)

@dataclass(frozen=True)
class PreprocessConfig:
    features_path: str
    output_dir: str
    max_seq_len: int
    games_per_shard: int
    test_fraction: float
    split_seed: int
    overwrite: bool
    
def fen_to_tensor(fen: str) -> np.ndarray:
    """
    Channels:
      0-11: white/black piece representation
      12: 1 means White to move, 0 is black to move 
      13-16: white-K, white-Q, black-K, black-Q castling-right planes
    """
    tensor = np.zeros((BOARD_CHANNELS, 8, 8), dtype=np.uint8)

    try:
        board = chess.Board(fen)
    except (ValueError, TypeError):
        return tensor

    for square, piece in board.piece_map().items():
        channel = PIECE_CHANNEL[(piece.piece_type, piece.color)]
        rank = chess.square_rank(square)
        file = chess.square_file(square)
        tensor[channel, rank, file] = 1

    if board.turn == chess.WHITE:
        tensor[12, :, :] = 1

    if board.has_kingside_castling_rights(chess.WHITE):
        tensor[13, :, :] = 1

    if board.has_queenside_castling_rights(chess.WHITE):
        tensor[14, :, :] = 1

    if board.has_kingside_castling_rights(chess.BLACK):
        tensor[15, :, :] = 1

    if board.has_queenside_castling_rights(chess.BLACK):
        tensor[16, :, :] = 1

    return tensor


def sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str: 
    digest = hashlib.sha256()
    
    with path.open("rb") as handle: 
        while chunk := handle.read(chunk_bytes): 
            digest.update(chunk) 
            
    return digest.hexdigest() 

def prepare_dataframe(features_path: Path) -> pl.DataFrame:
    df = pl.read_parquet(features_path)

    required = [
        "game_id",
        "move_number",
        "fen_before",
        "cp_loss",
        "white_elo",
        "black_elo",
        *TIME_FEATURE_COLUMNS,
    ]

    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(
            f"features.parquet is missing columns: {sorted(missing)}"
        )

    return (
        df.select(required)
        .drop_nulls( ["game_id", "move_number", "fen_before","white_elo","black_elo"])
        .with_columns(
            pl.col("move_number").cast(pl.Int32),
            pl.col(["cp_loss", "white_elo", "black_elo", *TIME_FEATURE_COLUMNS])
            .cast(pl.Float32)
            .fill_nan(0.0)
            .fill_null(0.0),
        )
    )
    
def split_game_ids(game_ids: list[Any], test_fraction: float, seed: int) -> tuple[set[Any], set[Any]]:
    """Return non-overlapping train and test game-id sets."""
    ids = np.asarray(game_ids, dtype=object)
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)

    n_test = int(len(ids) * test_fraction)
    test_ids = set(ids[:n_test].tolist())
    train_ids = set(ids[n_test:].tolist())

    return train_ids, test_ids


def encode_game(game: pl.DataFrame, max_seq_len: int) -> dict[str, torch.Tensor]: 
    game = game.sort("move_number").head(max_seq_len)
    seq_len = len(game)

    if seq_len == 0:
        raise ValueError("Cannot encode an empty game.")

    boards = np.zeros((max_seq_len, BOARD_CHANNELS, 8, 8), dtype=np.uint8)
    fens = game["fen_before"]
    
    for index, fen in enumerate(fens): 
        boards[index] = fen_to_tensor(fen) 
    
    time_features = np.zeros((max_seq_len, N_TIME_FEATURES), dtype = np.float16)
    
    time_features[:seq_len] = (
        game.select(list(TIME_FEATURE_COLUMNS))
        .to_numpy()
        .astype(np.float16, copy=False)
    )
    
    cp_loss = np.zeros(max_seq_len, dtype=np.float32)
    cp_loss[:seq_len] = game["cp_loss"].to_numpy().astype(np.float32, copy=False)

    attention_mask = np.zeros(max_seq_len, dtype=np.bool_)
    attention_mask[:seq_len] = True
    
    elos = (game.select(["white_elo", "black_elo"]).row(0))
    elos = np.asarray(elos, dtype=np.float32)
    
    return {
        "board_tensors": torch.from_numpy(boards),
        "time_features": torch.from_numpy(time_features),
        "attention_mask": torch.from_numpy(attention_mask),
        "cp_loss": torch.from_numpy(cp_loss),
        "elos": torch.from_numpy(elos),
        "seq_len": torch.tensor(seq_len, dtype=torch.int16),
    }


def iter_games(df: pl.DataFrame, game_ids: set[Any]) -> Iterator[tuple[str, pl.DataFrame]]: 
    """Yield game groups """
    split_df = df.filter(pl.col("game_id").is_in(game_ids))
    
    for game_id, game in split_df.group_by("game_id"): 
        yield str(game_id[0] if isinstance(game_id, tuple) else game_id ), game 

def write_webdataset_split( 
    df: pl.DataFrame,
    game_ids: set[Any], 
    output_dir: Path, 
    split: str, 
    max_seq_len: int, 
    games_per_shard: int,
) -> dict[str, int]: 
    """Tensorize and write split to tar shards"""
    split_dir = output_dir / split 
    split_dir.mkdir(parents=True, exist_ok=True)
    
    pattern = str(split_dir / f"{split}-%06d.tar")
    
    games_written = 0 
    skipped_games = 0 
    
    with wds.ShardWriter(pattern, maxcount=games_per_shard) as sink: 
        for game_id, game in iter_games(df, game_ids): 
            try: 
                sample = encode_game(game, max_seq_len=max_seq_len)
            except Exception as exc: 
                skipped_games += 1 
                print(f"[{split}] Skipping game_id={game_id}: {exc}")
                continue 
                
            sink.write(
                {
                    "__key__": f"game-{game_id}",
                    "pth": sample,
                }
            )
            games_written += 1 

            if games_written % 1000 == 0: 
                print(f"[{split}] wrote {games_written:,} games")

    shard_count = len(list(split_dir.glob(f"{split}-*.tar")))

    return {
        "games_written": games_written,
        "games_skipped": skipped_games,
        "shards_written": shard_count,
    }

def write_manifest(
    output_dir: Path,
    config: PreprocessConfig,
    train_stats: dict[str, int],
    test_stats: dict[str, int],
) -> None:
    manifest = {
        "source": config.features_path,
        "split_seed": config.split_seed,
        "test_fraction": config.test_fraction,
        "max_seq_len": config.max_seq_len,
        "games_per_shard": config.games_per_shard,
        "train": train_stats,
        "test": test_stats,
    }

    with (output_dir / "manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)
        
def prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"{output_dir} exists and is not empty. "
                "Use --overwrite or choose a new version directory."
            )

        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize chess Parquet features into WebDataset shards."
    )

    parser.add_argument(
        "--features-path",
        type=Path,
        required=True,
        help="Input analytical feature Parquet file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Versioned WebDataset output directory.",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=DEFAULT_MAX_SEQ_LEN,
    )
    parser.add_argument(
        "--games-per-shard",
        type=int,
        default=1_000,
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.2,
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete an existing output directory before writing.",
    )

    return parser.parse_args()

def main() -> None:
    args = parse_args()

    if not args.features_path.exists():
        raise FileNotFoundError(args.features_path)

    if not 0.0 < args.test_fraction < 1.0:
        raise ValueError("--test-fraction must be between 0 and 1.")

    if args.max_seq_len <= 0:
        raise ValueError("--max-seq-len must be positive.")

    if args.games_per_shard <= 0:
        raise ValueError("--games-per-shard must be positive.")

    config = PreprocessConfig(
        features_path=str(args.features_path),
        output_dir=str(args.output_dir),
        max_seq_len=args.max_seq_len,
        games_per_shard=args.games_per_shard,
        test_fraction=args.test_fraction,
        split_seed=args.split_seed,
        overwrite=args.overwrite,
    )

    prepare_output_dir(args.output_dir, overwrite=args.overwrite)

    print(f"Loading feature table: {args.features_path}")
    df = prepare_dataframe(args.features_path)

    print(
        f"Clean feature table: {len(df):,} moves across "
        f"{df['game_id'].n_unique():,} games"
    )

    game_ids = df["game_id"].unique().to_list()

    train_game_ids, test_game_ids = split_game_ids(
        game_ids=game_ids,
        test_fraction=config.test_fraction,
        seed=config.split_seed,
    )

    print(
        f"Split: {len(train_game_ids):,} train games, "
        f"{len(test_game_ids):,} test games"
    )

    train_stats = write_webdataset_split(
        df=df,
        game_ids=train_game_ids,
        output_dir=args.output_dir,
        split="train",
        max_seq_len=config.max_seq_len,
        games_per_shard=config.games_per_shard,
    )

    test_stats = write_webdataset_split(
        df=df,
        game_ids=test_game_ids,
        output_dir=args.output_dir,
        split="test",
        max_seq_len=config.max_seq_len,
        games_per_shard=config.games_per_shard,
    )

    write_manifest(
        output_dir=args.output_dir,
        config=config,
        train_stats=train_stats,
        test_stats=test_stats,
    )

    print("\nDone.")
    print(f"Training store: {args.output_dir}")
    print(f"Train: {train_stats}")
    print(f"Test:  {test_stats}")


if __name__ == "__main__":
    main()