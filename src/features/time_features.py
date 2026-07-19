"""
time_features.py — Clock and time management features via DuckDB

Features:
  time_pressure_flag     — clock_after < 30s (bool)
  avg_time_spent         — rolling avg over last 5 moves, same player
  time_spent_vs_avg      — this move's time vs player's game average (deviation)
  time_acceleration      — avg_time_last_3 vs avg_time_last_10 (speeding up?)
"""

import duckdb
import polars as pl

#SQL Statement for extracting time window features 
WINDOW_SQL = """
WITH base AS (
    SELECT
        game_id,
        move_number,
        color,
        fen_before,
        move_uci,
        clock_after,
        time_spent,
        cp_loss,
        white_elo,
        black_elo, 
        time_control_type,
        time_control_base,
        time_spent_norm, 
        
        

        -- time_pressure_flag: < 30s left after move
        CASE
            WHEN clock_after IS NOT NULL THEN clock_after < 30
            ELSE NULL
        END AS time_pressure_flag,


        -- Rolling avg over last 5 moves, same player
        -- PARTITION BY color keeps white and black on separate windows
        AVG(time_spent) OVER (
            PARTITION BY game_id, color
            ORDER BY move_number
            ROWS BETWEEN 4 PRECEDING AND CURRENT ROW
        ) AS avg_time_spent_5,

        -- Rolling avg over last 10 moves, same player
        AVG(time_spent) OVER (
            PARTITION BY game_id, color
            ORDER BY move_number
            ROWS BETWEEN 9 PRECEDING AND CURRENT ROW
        ) AS avg_time_spent_10,

        -- Rolling avg over last 3 moves, same player
        AVG(time_spent) OVER (
            PARTITION BY game_id, color
            ORDER BY move_number
            ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
        ) AS avg_time_spent_3,

        -- Game-level avg time for this player
        AVG(time_spent) OVER (
            PARTITION BY game_id, color
        ) AS game_avg_time_spent

    FROM read_parquet('{input_glob}')
)

SELECT
    game_id,
    move_number,
    color,
    fen_before,
    move_uci,
    time_spent,
    cp_loss, 
    white_elo,
    black_elo, 
    time_control_type,    
    time_control_base,
    time_pressure_flag,
    avg_time_spent_5,
    avg_time_spent_10,
    time_spent_norm, 

    -- time_acceleration: negative = speeding up, positive = slowing down
    (avg_time_spent_3 - avg_time_spent_10) AS time_acceleration,

    -- deviation from player's own game average
   
    (time_spent - game_avg_time_spent) AS time_spent_vs_avg

FROM base
"""


def compute_time_features(processed_dir: str) -> pl.DataFrame:
    """
    Run window SQL against processed Parquet and return a DataFrame
    """
    input_glob = f"{processed_dir}/*.parquet"
    sql        = WINDOW_SQL.format(input_glob=input_glob)
    return duckdb.sql(sql).pl()