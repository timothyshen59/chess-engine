import numpy as np
import polars as pl

from src.training.preprocessing import (
    BOARD_CHANNELS,
    N_TIME_FEATURES,
    encode_game,
    fen_to_tensor,
)


def test_fen_to_tensor_starting_position():
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

    tensor = fen_to_tensor(fen)

    assert tensor.shape == (BOARD_CHANNELS, 8, 8)
    assert tensor.dtype == np.uint8
    assert tensor[:12].sum() == 32
    assert tensor[12].sum() == 64  # White to move
    assert tensor[13].sum() == 64  # White kingside castling
    assert tensor[14].sum() == 64  # White queenside castling
    assert tensor[15].sum() == 64  # Black kingside castling
    assert tensor[16].sum() == 64  # Black queenside castling


def test_encode_game_pads_correctly():
    game = pl.DataFrame(
        {
            "move_number": [0, 1, 2],
            "fen_before": [
                "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1",
                "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 2",
            ],
            "time_spent_norm": [0.5, 1.0, 1.5],
            "time_pressure_flag": [0.0, 0.0, 1.0],
            "avg_time_spent_5": [2.0, 3.0, 4.0],
            "cp_loss": [12.0, 0.0, 35.0],
            "white_elo": [1500.0, 1500.0, 1500.0],
            "black_elo": [1600.0, 1600.0, 1600.0],
        }
    )

    sample = encode_game(game, max_seq_len=8)

    assert sample["board_tensors"].shape == (8, BOARD_CHANNELS, 8, 8)
    assert sample["time_features"].shape == (8, N_TIME_FEATURES)
    assert sample["attention_mask"].shape == (8,)
    assert sample["cp_loss"].shape == (8,)
    assert sample["elos"].shape == (2,)

    assert sample["seq_len"].item() == 3
    assert sample["attention_mask"].tolist() == [
        True, True, True, False, False, False, False, False
    ]

    assert sample["board_tensors"][3:].sum().item() == 0
    assert sample["time_features"][3:].sum().item() == 0
    assert sample["cp_loss"][3:].sum().item() == 0

    assert sample["elos"].tolist() == [1500.0, 1600.0]