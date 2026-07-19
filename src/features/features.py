"""
features.py — Layer 2: Feature engineering from raw Parquet.
 
Takes the move-level Parquet from ingest.py and computes 5 features
per move that the XGBoost model will train on.
 
Features:
  1. num_legal_moves     — position complexity (from FEN via python-chess)
  2. material_balance    — who's ahead in pieces (from FEN)
  3. king_safety_score   — pawn shield around king (from FEN)
  4. time_pressure_moves — is player low on time (from clock data)
  5. avg_time_spent      — rolling avg time per player per game (DuckDB window)
 
Usage:
  uv run src/features/features.py --processed-dir data/processed --features-dir data/features
"""

import chess 
import duckdb 
import polars as pl 
import argparse 
from pathlib import Path 

#Centipawn Engine Values for pieces. 
PIECE_VALUES = {
    chess.PAWN:   100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK:   500,
    chess.QUEEN:  900,
}

ATTACK_WEIGHTS = {
    chess.PAWN:   1,
    chess.KNIGHT: 2,
    chess.BISHOP: 2,
    chess.ROOK:   3,
    chess.QUEEN:  5,
}

#Ppsitional Features 




#Might have to revise this one, recompute, and add more features 
def compute_king_safety_score(fen:str) -> int: 
    """Computes king safety score based on attacking piece weight and pawn shield
    Positive is "safe" king, negative is king under attack (range -27 to + 3)
    """
    
    board = chess.Board(fen) 
    color = board.turn 
    enemy = not color 
    king = board.king(color) 
    
    if king is None: 
        return 0 
    
    king_file = chess.square_file(king)
    king_rank = chess.square_rank(king) 
    
    shield_rank = king_rank + 1 if color == chess.WHITE else king_rank - 1 
    shield = 0 
    
    if 0 <= shield_rank <= 7: 
        for f in range(max(0, king_file - 1), min(7, king_file + 1) +1 ): 
            sq = chess.square(f, shield_rank) 
            piece = board.piece_at(sq) 
            if piece and piece.piece_type == chess.PAWN and piece.color == color: 
                shield += 1 
                
    #King Zone (area around king) 
    king_zone = set() 
    for dr in [-1,0,1]: 
        for df in [-1,0,1]: 
            r = king_rank + dr 
            f = king_file + df 
            if 0 <= r <= 7 and 0 <= f <= 7: 
                king_zone.add(chess.square(f,r))
    
    #Attack Weight 
    attackers_on_zone = set() 
    for sq in king_zone: 
        for attacker_sq in board.attackers(enemy, sq): 
            attackers_on_zone.add(attacker_sq) 
            
    attack_weight = 0
    for attacker_sq in attackers_on_zone: 
        piece = board.piece_at(attacker_sq) 
        
        if piece and piece.piece_type in ATTACK_WEIGHTS: 
            attack_weight += ATTACK_WEIGHTS[piece.piece_type]
    
    return shield - attack_weight

def compute_position_features(fens: list[str]) -> dict[str, list]: 
    """Compute all 3 position features for a list of FEN strings
    
    Return: 
        dict of lists to add as polar columns
    """
    num_legal, material, king_safety = [], [], [] 
    
    for fen in fens: 
        try: 
            num_legal.append(compute_num_legal_moves(fen))
            material.append(compute_material_balance(fen))
            king_safety.append(compute_king_safety_score(fen))
        except Exception:
            num_legal.append(None)
            material.append(None)
            king_safety.append(None)
            #Fallthrouhg - let XGBoost handle it 
            
    return {
        "num_legal_moves":   num_legal,
        "material_balance":  material,
        "king_safety_score": king_safety,
    }
    
    



def compute_passed_pawns(board: chess.Board) -> int: 
    
#Wdinow Featrues (DuckDB) 

#SQL statement for window 


WINDOW_FEATURES_SQL = """
SELECT 
    game_id, 
    move_number, 
    color, 
    fen_before, 
    move_uci,
    time_spent, 
    time_control_base, 
    
    --Feature 4: time_pressure_flag (30s threshold)
    
    CASE
        WHEN clock_after IS NOT NULL THEN clock_after < 30 
        ELSE NULL 
    END AS time_pressure_flag, 
    
    --Feature 5: avg_time_spent (rolling_avg over last 5 moves, same player) 
    AVG(time_spent) OVER ( 
        PARTITION BY game_id, color 
        ORDER BY move_number 
        ROWS BETWEEN 4 PRECEDING AND CURRENT ROW 
    
    ) AS avg_time_spent 

FROM read_parquet('{input_glob}')

"""

def build_features(processed_dir: str, features_dir: str) -> None: 
    
    input_glob = f"{processed_dir}/*.parquet"
    output_path= f"{features_dir}/features.parquet"
    
    print("Computing window features via DuckDB...")
    sql = WINDOW_FEATURES_SQL.format(input_glob=input_glob)
    df = duckdb.sql(sql).pl() 
    print(f"Loaded {len(df):,} rows")
    print("Computing position features via python-chess …")

    pos_features = compute_position_features(df["fen_before"].to_list())
    df = df.with_columns([
        pl.Series("num_legal_moves",   pos_features["num_legal_moves"]),
        pl.Series("material_balance",  pos_features["material_balance"]),
        pl.Series("king_safety_score", pos_features["king_safety_score"]),
    ])
    
     
    print(f"Writing features -> {output_path}")
    Path(features_dir).mkdir(parents=True, exist_ok=True)
    df.write_parquet(output_path, compression="zstd")
    print(f"Finished feature processing. Data shape:{len(df):,} rows, {df.shape[1]} columns")
    print(f"Columns: {df.columns}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chess feature engineering — Layer 2")
    parser.add_argument("--processed-dir", type=str, default="data/processed",
                        help="Input: directory of raw Parquet from ingest.py")
    parser.add_argument("--features-dir",  type=str, default="data/features",
                        help="Output: directory for feature Parquet")
    args = parser.parse_args()
 
    build_features(args.processed_dir, args.features_dir)