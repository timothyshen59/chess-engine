"""
extract_buckets.py — Extract rating-stratified chess games with evals from Lichess dumps.

Config file (config/extract.yaml) specifies dump files and settings.
Extracts per_bucket_limit games per file, appending across files.

Example config (config/extract.yaml):
  output_dir: data/raw
  per_bucket_limit: 5000
  buckets:
    - [1000, 1200]
    - [1200, 1500]
    - [1500, 1800]
    - [1800, 2100]
    - [2100, 2400]
    - [2400, 3000]
  dumps:
    - data/raw/lichess_db_standard_rated_2018-06.pgn.zst
    - data/raw/lichess_db_standard_rated_2018-07.pgn.zst
    - data/raw/lichess_db_standard_rated_2018-08.pgn.zst
    - data/raw/lichess_db_standard_rated_2018-09.pgn.zst

Usage:
  uv run src/ingestion/extract_buckets.py --config config/extract.yaml
"""

import io
import argparse
import yaml
import zstandard as zstd
from pathlib import Path
from tqdm import tqdm


def extract_from_dump(
    zst_path:        str,
    output_dir:      str,
    buckets:         list[tuple[int, int]],
    per_bucket_limit: int,
) -> dict[tuple[int, int], int]:
    """
    Stream one .pgn.zst dump and append up to per_bucket_limit games
    per bucket into existing PGN files.

    Always appends — caller manages file creation on first run.

    Returns:
        counts of games extracted per bucket from this dump.
    """
    out_files = {
        (lo, hi): open(f"{output_dir}/games_{lo}_{hi}.pgn", "a")
        for lo, hi in buckets
    }
    counts = {(lo, hi): 0 for lo, hi in buckets}
    done   = set()

    dctx = zstd.ZstdDecompressor()
    with open(zst_path, "rb") as fh:
        with dctx.stream_reader(fh) as reader:
            text = io.TextIOWrapper(reader, encoding="utf-8", errors="replace")

            with tqdm(desc=f"  {Path(zst_path).name}", unit=" games") as pbar:
                game_lines: list[str] = []

                for line in text:
                    if line.startswith('[Event '):
                        # process previously accumulated game
                        if game_lines:
                            pgn_str = ''.join(game_lines)

                            if '%eval' in pgn_str:
                                white_elo = 0
                                for l in game_lines:
                                    if l.startswith('[WhiteElo '):
                                        try:
                                            white_elo = int(l.split('"')[1])
                                        except Exception:
                                            pass
                                        break

                                for (lo, hi) in buckets:
                                    if (lo, hi) in done:
                                        continue
                                    if lo <= white_elo < hi:
                                        out_files[(lo, hi)].write(pgn_str + '\n')
                                        counts[(lo, hi)] += 1
                                        pbar.update(1)
                                        if counts[(lo, hi)] >= per_bucket_limit:
                                            done.add((lo, hi))
                                            out_files[(lo, hi)].flush()
                                            tqdm.write(f"  ✓ {lo}-{hi}: {counts[(lo,hi)]} games from this dump")
                                        break

                        game_lines = [line]

                        if len(done) >= len(buckets):
                            break
                    else:
                        game_lines.append(line)

    for f in out_files.values():
        f.close()

    return counts


def run(config_path: str) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    output_dir        = cfg["output_dir"]
    per_bucket_limit  = cfg["per_bucket_limit"]
    buckets           = [tuple(b) for b in cfg["buckets"]]
    dumps             = cfg["dumps"]

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # clear existing bucket files before starting
    for lo, hi in buckets:
        path = Path(f"{output_dir}/games_{lo}_{hi}.pgn")
        if path.exists():
            path.unlink()

    global_counts = {(lo, hi): 0 for lo, hi in buckets}

    for zst_path in dumps:
        print(f"\nProcessing {Path(zst_path).name} ...")
        print(f"  Target: {per_bucket_limit} games per bucket from this file")

        new_counts = extract_from_dump(
            zst_path         = zst_path,
            output_dir       = output_dir,
            buckets          = buckets,
            per_bucket_limit = per_bucket_limit,
        )

        for bucket, count in new_counts.items():
            global_counts[bucket] += count

        print(f"  Running totals:")
        for (lo, hi), count in global_counts.items():
            print(f"    {lo}-{hi}: {count:,} games")

    print("\n── Final totals ─────────────────────────────────")
    total = 0
    for (lo, hi), count in global_counts.items():
        expected = per_bucket_limit * len(dumps)
        print(f"  {lo}-{hi}: {count:,} / {expected:,} games")
        total += count
    print(f"  Total: {total:,} games across all buckets")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract stratified chess games from Lichess dumps")
    parser.add_argument("--config", type=str, default="config/extract.yaml")
    args = parser.parse_args()
    run(args.config)