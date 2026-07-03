import io 
import re 
import random 
import argparse 

import chess 
import chess.pgn 
import zstandard as zstd 
import polars as pl 
import berserk #Lichess API 
from pathlib import Path 
from dataclasses import dataclass, asdict 
from typing import Iterator 

@dataclass 
class MoveRow: 
    game_id: str 
    move_number: int 
    color: str 
    fen_before: str 
    move_uci: str 
    move_san: str 
    
    white_elo: int 
    black_elo: int 
    result: str 
    
    time_control: str 
    time_control_base: int 
    time_control_inc: int 
    time_control_type: str 
    
    clock_before: float | None 
    clock_after: float | None 
    
    time_spent: float | None 
    time_spent_norm: float | None # time_spent/ time_control_base

#Clock Parser

_CLK_RE = re.compile(r'\[%clk\s+(\d+):(\d+):(\d+)\]') # Regex (extracts h:mm:ss)

def _parse_clock(comment: str | None) -> float | None:
    """Extract seconds from a Lichess clock comment. Returns None if absent."""
    if not comment:
        return None
    
    m = _CLK_RE.search(comment)
    
    if not m:
        return None
    
    h, mn, s = int(m.group(1)), int(m.group(2)), int(m.group(3))
    return float(h * 3600 + mn * 60 + s)

def _parse_time_control(tc: str)-> tuple[int, int, str]: 
    """Parse time control string into (base_seconds, increment, category)"""
    
    try: 
        base, inc = map(int, tc.split("+"))
    except Exception: 
        return 0,0, "unknown" #Edge case for unknowns

    estimated = base + 40 * inc #Lichess categorization formula 
    
    if estimated < 179:
        category = "bullet"
    elif estimated < 479:
        category = "blitz"
    elif estimated < 1499:
        category = "rapid"
    else:
        category = "classical"

    return base, inc, category
    
def _get_clock_before(node: chess.pgn.GameNode, base_seconds: int) -> float: 
    """
    Get current player's clock time before amaking this move. 
    """
    
    if node.parent is None or node.parent.parent is None: 
        return float(base_seconds) # If no previous move
    
    grandparent = node.parent.parent 
    clk = _parse_clock(grandparent.comment)
    
    if clk is None: 
        return float(base_seconds) #Corrupted PGN 

    return clk 


#Game Parser 
def _parse_game(game: chess.pgn.Game) -> list[MoveRow]: 
    """Turn one PGN game into a list of MoveRow objects"""

    headers = game.headers 
    game_id = headers.get("Site", "?").split("/")[-1] 
    result = headers.get("Result", "?")
    time_ctrl = headers.get("TimeControl", "?")
    
    base, inc, tc_type = _parse_time_control(time_ctrl)

    def safe_int(v): #If LiChess returns "?" for unrated players 
        try: return int(v or 0) 
        except: return 0 
        
    white_elo = safe_int(headers.get("WhiteElo"))
    black_elo = safe_int(headers.get("BlackElo"))
    
    rows = [] 
    board = game.board() 
    ply = 0 
    
    for node in game.mainline(): 
        move = node.move 
        
        color = "white" if board.turn == chess.WHITE else "black" 
        fen_before = board.fen() 
        san = board.san(move) 

        clk_before = _get_clock_before(node, base) 
        clk_after = _parse_clock(node.comment) 
        
        if clk_after is not None: 
            raw_spent = clk_before - clk_after 
            time_spent = round(max(0.0, raw_spent), 2)
        else: 
            time_spent = None 
            
        time_spent_norm = (
            round(time_spent / base, 4)
            if time_spent is not None and base > 0
            else None
        )
        
        
        rows.append(MoveRow(
            game_id           = game_id,
            move_number       = ply + 1,
            color             = color,
            fen_before        = fen_before,
            move_uci          = move.uci(),
            move_san          = san,
            white_elo         = white_elo,
            black_elo         = black_elo,
            result            = result,
            time_control      = time_ctrl,
            time_control_base = base,
            time_control_inc  = inc,
            time_control_type = tc_type,
            clock_before      = clk_before,
            clock_after       = clk_after,
            time_spent        = time_spent,
            time_spent_norm   = time_spent_norm,
        ))

        board.push(move)
        ply += 1 
        
    return rows 

#Parquet Writer 

def _flush_to_parquet(rows: list[dict], base_path: str, batch_idx: int) -> None: 
    """Write batch of move rows to compressed Parquest shard """
    shard = f'{base_path}_batch{batch_idx:04d}.parquet'
    df = pl.DataFrame(rows) 
    df.write_parquet(shard, compression="zstd")
    print(f"Wrote {len(df)} rows to -> {shard}")
        
#Game DUmp Processing 
def process_dump(zst_path: Path, processed_dir: str, batch_size: int = 10_000) -> None: 
    output_prefix = f"{processed_dir}/{zst_path.stem}"    
    dctx = zstd.ZstdDecompressor() 
    
    print(f"Processing {zst_path.name}") 
    games_done = 0 
    batch_idx = 0 
    buffer: list[dict] = [] 
    
    with open(zst_path, "rb") as fh: 
        with dctx.stream_reader(fh) as reader: 
            text = io.TextIOWrapper(reader, encoding="utf-8", errors="replace")
            
            while True: 
                game = chess.pgn.read_game(text) 
                
                if game is None: 
                    break 
                
                if game.next() is None: 
                    continue 
                
                buffer.extend(asdict(r) for r in _parse_game(game))
                games_done += 1 
                
                if games_done % batch_size == 0: 
                    _flush_to_parquet(buffer, output_prefix, batch_idx)
                    batch_idx += 1 
                    
                    buffer = [] 
                    print(f" {games_done:} games processed.")
                
    if buffer: 
        _flush_to_parquet(buffer, output_prefix, batch_idx) 
        
    print(f"Finishing processing {games_done} games -> {processed_dir}")
    

#Testing (Sample Games)

def sample_games(n: int, perf_type: str, processed_dir: str) -> None: 
    """Samples n games from random top player on leaderboard via berserk and writes data to parquet"""
    client = berserk.Client()

    print(f"Fetching {perf_type} leaderboard....")
    leaderboard = client.users.get_leaderboard(perf_type, count=50)
    player      = random.choice(leaderboard)["username"]
    print(f"Sampling {n} games from {player}...")

    games_gen = client.games.export_by_player(
        player,
        max=n,
        clocks=True,
        opening=True,
        as_pgn=True,
        perf_type=perf_type,
    )

    buffer: list[dict] = []
    games_done = 0

    for pgn_str in games_gen:
  
        game = chess.pgn.read_game(io.StringIO(pgn_str))
        if game is None or game.next() is None:
            continue
        buffer.extend(asdict(r) for r in _parse_game(game))
        games_done += 1

    out = f"{processed_dir}/sample_{games_done}_games"
    _flush_to_parquet(buffer, out, batch_idx=0)
    print(f"Processed {games_done} games from {player} ")
    

if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description="Chess data ingestion — Layer 1")

    parser.add_argument("--raw-dir",       type=str, default="data/raw",
                        help="File directory of dump files (local path or s3://...)")
    parser.add_argument("--processed-dir", type=str, default="data/processed",
                        help="File directory where Parquet output goes (local path or s3://...)")

    sub = parser.add_subparsers(dest="cmd")

    d = sub.add_parser("dump",     help="Process a .pgn.zst dump file to Parquet")
    d.add_argument("path",         type=Path, help="Path to the .pgn.zst file")
    d.add_argument("--batch-size", type=int,  default=10_000)

    s = sub.add_parser("sample",   help="Download N random games for testing")
    s.add_argument("--n",          type=int,  default=200)
    s.add_argument("--perf-type",  type=str,  default="blitz",
                   help="bullet | blitz | rapid | classical")

    args = parser.parse_args()

    # Only mkdir for local paths — S3 has no real directories
    if not args.processed_dir.startswith("s3://"):
        Path(args.processed_dir).mkdir(parents=True, exist_ok=True)
    if not args.raw_dir.startswith("s3://"):
        Path(args.raw_dir).mkdir(parents=True, exist_ok=True)

    if args.cmd == "dump":
        process_dump(args.path, args.processed_dir, batch_size=args.batch_size)
    elif args.cmd == "sample":
        sample_games(n=args.n, perf_type=args.perf_type, processed_dir=args.processed_dir)
    else:
        parser.print_help()
