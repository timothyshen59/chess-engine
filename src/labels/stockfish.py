"""
stockfish.py
"""


import chess
import chess.engine 
import polars as pl 
import argparse 
from pathlib import Path 
from tqdm import tqdm 

def cp_loss_to_label(cp_loss: float |None ) -> str: 
    """Convert centipawn loss to move-quality labels """
    if cp_loss is None:
        return "unknown"
    if cp_loss < 10:
        return "good"
    if cp_loss < 25:
        return "inaccuracy"
    if cp_loss < 100:
        return "mistake"
    return "blunder"

#Stockfish Evaluator
def evaluate_move(
    engine: chess.engine.SimpleEngine, 
    fen_before: str, 
    move_uci: str, 
    depth: int, 
) -> float | None: 
    try: 
        board = chess.Board(fen_before) 
        move = chess.Move.from_uci(move_uci) 
        
        limit = chess.engine.Limit(depth=depth) 
        
        info_before = engine.analyse(board,limit) 
        mover = board.turn  # capture BEFORE pushing
        
        score_before = info_before["score"].pov(mover).score(mate_score=10000)
        board.push(move)
        info_after = engine.analyse(board, limit)
        score_after = info_after["score"].pov(mover).score(mate_score=10000)
        
        cp_loss = max(0, score_before - score_after) 
        cp_loss = min(cp_loss, 500)
        
        return float(cp_loss) 
    
    except Exception: 
        return None 

def generate_labels( 
    features_dir: str, 
    labels_dir: str, 
    stockfish_path: str, 
    depth: int, 
    max_rows: int | None, 
) -> None: 
    input_path = f"{features_dir}/features.parquet"
    output_path = f"{labels_dir}/labels.parquet"
    
    print(f"Loading featrues from {input_path}...")
    df = pl.read_parquet(input_path)
    
    if max_rows: 
        df = df.head(max_rows) 
        print(f" Using only first {max_rows} for testing")
    
    print(f"  {len(df):,} positions to evaluate at depth {depth}")

    print(f"Starting stockfish from {stockfish_path}...")
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    
    cp_losses: list[float | None] = [] 
    labels: list[str] = [] 

    for i, row in tqdm(enumerate(df.iter_rows(named=True)), total = len(df), desc="Stockfish"): 
        cp_loss = evaluate_move(
            engine=engine, 
            fen_before = row["fen_before"],
            move_uci   = row["move_uci"], 
            depth      = depth,
        )
        
        cp_losses.append(cp_loss) 
        labels.append(cp_loss_to_label(cp_loss))
    
    engine.quit() 
    
    result = df.select(["game_id", "move_number", "color"]).with_columns([
        pl.Series("centipawn_loss",  cp_losses),
        pl.Series("move_quality",    labels),
    ])
    
    Path(labels_dir).mkdir(parents=True, exist_ok=True)
    result.write_parquet(output_path, compression="zstd")
    
    print("\nLabel distribution:")
    print(result["move_quality"].value_counts().sort("move_quality"))
#CLI 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stockfish label generation — Layer 2.5")
    parser.add_argument("--features-dir",   type=str, default="data/features")
    parser.add_argument("--labels-dir",     type=str, default="data/labels")
    parser.add_argument("--stockfish-path", type=str, default="stockfish",
                        help="Path to Stockfish binary (default: 'stockfish' on PATH)")
    parser.add_argument("--depth",          type=int, default=10,
                        help="Stockfish search depth (10=fast/dev, 20=accurate/slow)")
    parser.add_argument("--max-rows",       type=int, default=None,
                        help="Limit rows for testing (e.g. 500)")
    args = parser.parse_args()
 
    generate_labels(
        features_dir   = args.features_dir,
        labels_dir     = args.labels_dir,
        stockfish_path = args.stockfish_path,
        depth          = args.depth,
        max_rows       = args.max_rows,
    )