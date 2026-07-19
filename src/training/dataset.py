"""
dataset.py

PyTorch Dataset for chess transformer model 

Training Example: 
-Board Tensor per move
-Time features (3 scalars
    time_spent_norm     — time spent / base clock (comparable across time controls)
    time_pressure_flag  — clock < 30s (0 or 1)
    avg_time_spent_5    — rolling avg time over last 5 moves
    

Labels: 
    cp_loss per move - centipawn loss 
    player_elo 

"""

import chess 
import numpy as np 
import torch 
from torch.utils.data import Dataset 
from torch.nn.utils.rnn import pad_sequence

import polars as pl 
from pathlib import Path 

MAX_SEQ_LEN = 128 
BOARD_CHANNELS = 17 #12 piece, 1 turn, 4 castling rights
N_TIME_FEATURES = 3 #time_spent_norm, time_pressure_flag, avg_time_spent_5 

PIECE_CHANNEL = { 
    (chess.PAWN, chess.WHITE): 0, 
    (chess.KNIGHT, chess.WHITE): 1,
    (chess.BISHOP, chess.WHITE): 2,
    (chess.ROOK,   chess.WHITE): 3,
    (chess.QUEEN,  chess.WHITE): 4,
    (chess.KING,   chess.WHITE): 5,
    (chess.PAWN,   chess.BLACK): 6,
    (chess.KNIGHT, chess.BLACK): 7,
    (chess.BISHOP, chess.BLACK): 8,
    (chess.ROOK,   chess.BLACK): 9,
    (chess.QUEEN,  chess.BLACK): 10,
    (chess.KING,   chess.BLACK): 11,

}

TIME_CONTROL_MAP = {"bullet": 0, "blitz": 1, "rapid": 2, "classical": 3, "unknown": -1}

#Board Tensor encoding 

def fen_to_tensor(fen:str )-> np.ndarray: 
    """
    Encode FEN string as 8 x 8 x 17 float32 tensor 
    
    Shape: (17, 8, 8)
    Channels:
      0-11: piece positions (1 = piece present, 0 = empty)
      12:   turn (all 1s = white to move, all 0s = black to move)
      13:   white kingside castling right
      14:   white queenside castling right
      15:   black kingside castling right
      16:   black queenside castling right
 
    """
    tensor = np.zeros((BOARD_CHANNELS, 8 , 8), dtype=np.float32)
    
    try: 
        board = chess.Board(fen) 
        
        for sq, piece in board.piece_map().items(): 
            ch = PIECE_CHANNEL[(piece.piece_type, piece.color)]
            rank = chess.square_rank(sq) 
            file = chess.square_file(sq) 
            tensor[ch, rank, file] = 1.0 

        if board.turn == chess.WHITE: 
            tensor[12, :, :] = 1.0 
        
        if board.has_kingside_castling_rights(chess.WHITE): 
            tensor[13,: ,:] = 1.0 
        if board.has_queenside_castling_rights(chess.WHITE): 
            tensor[14,:,:] = 1.0 
        if board.has_kingside_castling_rights(chess.BLACK): 
            tensor[15,:,:] = 1.0 
        if board.has_queenside_castling_rights(chess.BLACK): 
            tensor[16, :, :] = 1.0 
    except Exception: 
        pass 

    return tensor 

#Chess Game Dataset 
class ChessGameDataset(Dataset): 
    """
    PyTorch Dataset: each item is a complete game 
    
    Each game is a sequence of moves: 
        Each move has 
        - board_tensor: (17, 8, 8) flaot32 
        - time_feature: (3, ) float 32 
        - cp_loss: scalar float32 
        - player_elo: scalar float32
    
    Game length limited by MAX_SEQ_LEN
    """
    
    def __init__(self, features_path: str, split: str = "train", test_fraction: float = 0.2, seed: int = 42): 
        """
        """
        print(f"Loading {features_path}...")
        df = pl.read_parquet(features_path)
        
        needed = ["game_id", "move_number", "color", "fen_before",
                  "cp_loss", "white_elo", "black_elo", "time_spent_norm", 
                  "time_pressure_flag", "avg_time_spent_5"]
        
        if "time_spent_norm" not in df.columns: 
            df = df.with_columns(
                (pl.col("time_spent") / pl.col("time_spent").mean().over("game_id"))
                .alias("time_spent_norm")
            )
        
        available = [c for c in needed if c in df.columns]
        df = df.drop_nulls(subset=available) 
        
        print(f" {len(df):,} moves after dropping nulls")

        game_ids = df["game_id"].unique().to_list() 
        rng = np.random.default_rng(seed=seed) 
        shuffled = rng.permutation(game_ids)
        n_test = int(len(game_ids) * test_fraction) 
        
        if split == "train": 
            keep_games = set(shuffled[n_test:])
        else: 
            keep_games = set(shuffled[:n_test])
        
        df = df.filter(pl.col("game_id").is_in(keep_games))
        print(f"  {split}: {len(df):,} moves from {len(keep_games)} games")
        
        self.games = [] 
        
        for game_id, group in df.group_by("game_id"): 
            group = group.sort("move_number")
            self.games.append(group)
        
        print(f"  {len(self.games)} games loaded")
        
    def __len__(self) -> int: 
        return len(self.games) 

    def __getitem__(self, idx: int) -> dict: 
        """
        Returns dict with tensor for one game 
        """
        game = self.games[idx]
        
        if len(game) > MAX_SEQ_LEN: 
            game = game.head(MAX_SEQ_LEN)
        
        seq_len = len(game) 
        fens = game["fen_before"].to_list() 
        colors = game["color"].to_list() 
        
        board_tensors = np.stack([fen_to_tensor(f) for f in fens]) #Maybe try multiprocesssing 
        
        # Imputes zeros for missing values 
        def get_col(col_name, default=0.0):
            if col_name in game.columns:
                vals = game[col_name].cast(pl.Float32).fill_null(default).to_list()
            else:
                vals = [default] * seq_len
            return vals
        
        time_features = np.array([
            get_col("time_spent_norm"),    # time_spent / base_clock
            get_col("time_pressure_flag"), # clock < 30s
            get_col("time_spent_vs_avg"),  # deviation from game average
        ], dtype=np.float32).T  # (seq_len, 3)

        cp_loss = np.array(
            game["cp_loss"].fill_null(0.0).to_list(),
            dtype=np.float32
        )
 
        white_elo = float(game["white_elo"][0])
        black_elo = float(game["black_elo"][0])
        
        return { 
            "board_tensors": torch.from_numpy(board_tensors), 
            "time_features": torch.from_numpy(time_features), 
            "cp_loss": torch.from_numpy(cp_loss),
            "white_elo": torch.tensor(white_elo), 
            "black_elo": torch.tensor(black_elo), 
            "seq_len": seq_len, 
            
        }
        
      
        
def collate_fn(batch: list[dict]) -> dict: 
    """
    Pad variable-length sequences to max-length 
    Returns attetnion_mask 
    """
    max_seq_len = max(item["seq_len"] for item in batch)
    
     
    board_tensors   = []
    time_features   = []
    cp_losses       = []
    white_elos      = []
    black_elos      = []
    attention_masks = []
    
    for item in batch:
        seq_len = item["seq_len"]
        pad     = max_seq_len - seq_len
 
        # Pad with zeros at end
        bt = item["board_tensors"]
        tf = item["time_features"]
        cp = item["cp_loss"]
 
        if pad > 0:
            bt = torch.cat([bt, torch.zeros(pad, BOARD_CHANNELS, 8, 8)])
            tf = torch.cat([tf, torch.zeros(pad, N_TIME_FEATURES)])
            cp = torch.cat([cp, torch.zeros(pad)])
 
        board_tensors.append(bt)
        time_features.append(tf)
        cp_losses.append(cp)
        white_elos.append(item["white_elo"])
        black_elos.append(item["black_elo"])
 
        attention_masks.append(torch.cat([
            torch.ones(seq_len),
            torch.zeros(pad),
        ]))
 
    return {
        "board_tensors":  torch.stack(board_tensors),    # (B, T, 17, 8, 8)
        "time_features":  torch.stack(time_features),    # (B, T, 3)
        "cp_loss":        torch.stack(cp_losses),        # (B, T)
        "white_elo":      torch.stack(white_elos),       # (B,)
        "black_elo":      torch.stack(black_elos),       # (B,)
        "attention_mask": torch.stack(attention_masks),  # (B, T)
    }
    
 
if __name__ == "__main__":
    from torch.utils.data import DataLoader
 
    ds     = ChessGameDataset("data/features/features.parquet", split="train")
    loader = DataLoader(ds, batch_size=32, shuffle=True, collate_fn=collate_fn, num_workers=0)
    batch  = next(iter(loader))
 
    print("\nBatch shapes:")
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {tuple(v.shape)} {v.dtype}")
        else:
            print(f"  {k}: {v}")
 
    

    
        
        
        
        
    
            
    
