 
import io
import json 
import orjson 
import chess
import xxhash
import zstandard as zstd
import polars as pl
import duckdb
import requests
import argparse
from pathlib import Path
from tqdm import tqdm


#Constants 
EVALS_URL  = "https://database.lichess.org/lichess_db_eval.jsonl.zst"
CP_CAP     = 500  # cap mate scores — same as stockfish.py
PARTITION_DEPTH = 2 #First PARTITION_DEPTH bytes we sort parquet shards by 

#FEN Helpers 

def normalize_fen(fen:str) -> str: 
    """Strips half-move clock and fullmove number from FEN string"""
    parts = fen.split() 
    return " ".join(parts[:4])

def fen_to_hash(fen: str) -> int: 
    """Hash a noramlized FEN string to int64 using xxhash. Reduces memory at join time"""
    return xxhash.xxh64(fen).intdigest()

def get_partition(fen_hash: int): 
    """Gets partition of FEN that we're going to key Parquet shards by
    Example: hash 0x3a7f.... -> partition "3a"( depth = 2) 
    """
    return f"{fen_hash:016x}"[:PARTITION_DEPTH]
    
def get_fen_after(fen_before: str, move_uci: str) -> str | None: 
    """Applies move to FEN and returns resulting position FEN 
    
    Return: 
        None if FEN or move is invalid. Otherwise, FEN
    """
    try: 
        board = chess.Board(fen_before) 
        move = chess.Move.from_uci(move_uci)
        board.push(move) 
        
        return normalize_fen(board.fen())
    
    except Exception: 
    
        return None 

def download_evals(evals_dir: str) -> Path: 
    """
    Download the Lichess evaluation .zst file
    https://database.lichess.org/#evals
    
    """
    dest = Path(evals_dir)/ "lichess_db_eval.jsonl.zst"
    Path(evals_dir).mkdir(parents=True, exist_ok=True)
    
    if dest.exists() and dest.stat().st_size > 0: 
        print(f"Evaluation database already exists: {dest}")
        return dest 
    
    print(f"Downloading eval database from {EVALS_URL} …")
    
    with requests.get(EVALS_URL, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(dest, "wb") as f:
            with tqdm(total=total, unit="B", unit_scale=True, desc="Downloading evals") as pbar:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
                            
    print(f"\nSaved to {dest} ({dest.stat().st_size // 1_048_576} MB)")
    return dest 



def flush_batch(fens: list[str], cps: list[int], output_dir: Path) -> None:
    """
    Vectorized batch flush. Computes hash and parttion keys in parallel 
    """
    df = pl.DataFrame({"fen": fens, "cp": pl.Series(cps, dtype=pl.Int32)})

    df = df.with_columns([
        pl.col("fen")
            .map_elements(fen_to_hash, return_dtype=pl.UInt64)
            .alias("fen_hash"),
    ]).with_columns([
        pl.col("fen_hash")
            .map_elements(get_partition, return_dtype=pl.Utf8)
            .alias("partition"),
    ]).drop("fen")  

    for (partition,), group in df.group_by(["partition"]):
        _flush_partition(
            group["fen_hash"].to_list(),
            group["cp"].to_list(),
            output_dir,
            partition,
        )


#Parquet Sharding Functions 
def build_evals_parquet(evals_dir: str, batch_size: int = 500_000) -> None: 
    
    zst_path = Path(evals_dir) / "lichess_db_eval.jsonl.zst"
    output_dir = Path(evals_dir) / "parquet"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Parsing eval database -> Parquet shards in {output_dir}")
    print(f"Output: {output_dir}") 
    
    dctx = zstd.ZstdDecompressor()
    positions = 0 
    batch_fens: list[str] = [] 
    batch_cps: list[int] = [] 
    
    with open(zst_path, "rb") as fh: 
        with dctx.stream_reader(fh) as reader: 
            text = io.TextIOWrapper(reader, encoding="utf-8", errors="replace") 
            
            for line in tqdm(text, desc="Parsing evals"): 
                line = line.strip() 
                
                if not line: 
                    continue 
                
                try: 
                    obj = orjson.loads(line) 
                except orjson.JSONDecodeError: 
                    continue 
                
                fen = obj.get("fen", "")
                evals = obj.get("evals", [])
                
                if not fen or not evals: 
                    continue 
                
                best = max(evals, key=lambda e: e.get("depth", 0))
                pvs = best.get("pvs", [])
                
                if not pvs: 
                    continue 
                
                pv = pvs[0]
                
                if "cp" in pv: 
                    cp = max(-CP_CAP, min(CP_CAP, pv["cp"]))
                elif "mate" in pv: 
                    cp = CP_CAP if pv["mate"] > 0 else -CP_CAP
                else: 
                    continue 
                
                batch_fens.append(normalize_fen(fen))
                batch_cps.append(cp)
                positions += 1 
                
                if len(batch_fens) >= batch_size: 
                    flush_batch(batch_fens, batch_cps, output_dir)
                    batch_fens.clear() 
                    batch_cps.clear() 
                        
                        
    if batch_fens: 
        flush_batch(batch_fens, batch_cps, output_dir)
    
    total_shards = len(list(output_dir.glob("**/*.parquet")))
    
    print(f"Done. {positions:,} positions across {total_shards} shards")


    
def _flush_partition(hashes: list[int], cps: list[int], output_dir: Path, partition: str) -> None:
    """Write one partition buffer to a new numbered Parquest shard. W
    Writes new file each flush """
    
    part_dir = output_dir / f"partition={partition}"
    part_dir.mkdir(parents=True, exist_ok=True)

    shard_idx  = len(list(part_dir.glob("evals_*.parquet")))
    shard_path = part_dir / f"evals_{shard_idx:04d}.parquet"

    # FIX: don't store partition column — it comes from folder name via hive partitioning
    pl.DataFrame({
        "fen_hash": pl.Series(hashes, dtype=pl.UInt64),
        "cp":       pl.Series(cps,    dtype=pl.Int32),
    }).sort("fen_hash").write_parquet(shard_path, compression="zstd")
    
    
def cp_loss_to_label(cp_loss: float | None) -> str:
    if cp_loss is None: return "unknown"
    if cp_loss < 10:    return "good"
    if cp_loss < 25:    return "inaccuracy"
    if cp_loss < 100:   return "mistake"
    return "blunder"


def generate_labels(features_dir: str, labels_dir: str, evals_dir: str) -> None: 
    """Join features Parquest with partitioned evals Parquet via DUckDB"""
    
    features_path = f"{features_dir}/features.parquet"
    evals_glob    = f"{evals_dir}/parquet/**/*.parquet"
    output_path   = f"{labels_dir}/labels.parquet"
    tmp_path      = f"{labels_dir}/moves_with_hashes.parquet"
    
    print("Loading features...")
    df = pl.read_parquet(features_path)
    print(f"{len(df):,} rows...")
    
    print("Computing FEN hashes and partitions (vectorized)...") 
    
    #Vectorize FEN normalization
    df = df.with_columns(
        pl.struct(["fen_before", "move_uci"])
            .map_elements(
                lambda r: get_fen_after(r["fen_before"], r["move_uci"]) or "",
                return_dtype=pl.Utf8,
            )
            .alias("fen_after_norm")
    
    )
    
    #Vectorize hashing and parition key computations
    df = df.with_columns([
        pl.col("fen_before")
          .map_elements(lambda f: fen_to_hash(normalize_fen(f)), return_dtype=pl.UInt64)
          .alias("fen_hash_before"),
        pl.col("fen_after_norm")
          .map_elements(lambda f: fen_to_hash(f) if f else 0, return_dtype=pl.UInt64)
          .alias("fen_hash_after"),
    ]).with_columns([
        pl.col("fen_hash_before")
          .map_elements(get_partition, return_dtype=pl.Utf8)
          .alias("partition_before"),
        pl.col("fen_hash_after")
          .map_elements(get_partition, return_dtype=pl.Utf8)
          .alias("partition_after"),
    ])
    
    Path(labels_dir).mkdir(parents=True, exist_ok=True)
    
    df.select([ 
        "game_id", "move_number", "color",
        "fen_hash_before", "fen_hash_after",
    ]).write_parquet(tmp_path)
    
    print("Joining with eval database via DuckDB ...")
    
    result = duckdb.sql(f"""
    SELECT
        m.game_id,
        m.move_number,
        m.color,
        e_before.cp AS cp_before,
        e_after.cp  AS cp_after,

        CASE
            WHEN e_before.cp IS NULL OR e_after.cp IS NULL THEN NULL
            WHEN m.color = 'white' THEN
                GREATEST(0, LEAST({CP_CAP}, e_before.cp - e_after.cp))
            ELSE
                GREATEST(0, LEAST({CP_CAP}, e_after.cp - e_before.cp))
        END AS centipawn_loss

    FROM read_parquet('{tmp_path}') m
    LEFT JOIN read_parquet('{evals_glob}', hive_partitioning=true) e_before
        ON m.fen_hash_before = e_before.fen_hash
    LEFT JOIN read_parquet('{evals_glob}', hive_partitioning=true) e_after
        ON m.fen_hash_after = e_after.fen_hash
    """).pl()
    
    result = result.with_columns(
        pl.col("centipawn_loss") 
            .map_elements(cp_loss_to_label, return_dtype=pl.Utf8)
            .alias("move_quality")
    )
    
    Path(tmp_path).unlink()
    result.write_parquet(output_path, compression="zstd")
 
    matched = result["centipawn_loss"].drop_nulls().len()
    print(f"\nDone. {len(result):,} labels -> {output_path}")
    print(f"Match rate: {matched:,} / {len(result):,} ({matched/len(result)*100:.1f}%)")
    print("\nLabel distribution:")
    print(result["move_quality"].value_counts().sort("move_quality"))
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lichess eval label generation")
    sub    = parser.add_subparsers(dest="cmd")
 
    b = sub.add_parser("build", help="Download + convert eval database to partitioned Parquet")
    b.add_argument("--evals-dir",       type=str, default="data/evals")
    b.add_argument("--skip-download",   action="store_true")
    b.add_argument("--partition-depth", type=int, default=PARTITION_DEPTH,
                   help="Hex chars for partition key (2=256, 3=4096, 4=65536 buckets)")
 
    l = sub.add_parser("label", help="Join features with evals to generate labels")
    l.add_argument("--features-dir", type=str, default="data/features")
    l.add_argument("--labels-dir",   type=str, default="data/labels")
    l.add_argument("--evals-dir",    type=str, default="data/evals")
 
    args = parser.parse_args()
 
    if args.cmd == "build":
        if hasattr(args, "partition_depth"):
            PARTITION_DEPTH = args.partition_depth
        if not args.skip_download:
            download_evals(args.evals_dir)
        build_evals_parquet(args.evals_dir)
 
    elif args.cmd == "label":
        generate_labels(args.features_dir, args.labels_dir, args.evals_dir)
 
    else:
        parser.print_help()