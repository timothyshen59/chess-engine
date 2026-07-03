"""
tests/test_ingest.py
 
Small, offline tests for the ingest pipeline.

 
Run with:  make test
           make test-fast   (stop at first failure)
"""

import io 
import chess.pgn 
import pytest 
from src.ingestion.ingest import( 
    _parse_clock, 
    _parse_game, 
    _parse_time_control, 
    _get_block_before, 

)

@pytest.fixture
def short_game_pgn():
    """A real 5-move PGN with clock data (Scholar's mate)"""
    return """
        [Event "Rated Blitz game"]
        [Site "https://lichess.org/testgame1"]
        [White "playerA"]
        [Black "playerB"]
        [WhiteElo "1500"]
        [BlackElo "1450"]
        [TimeControl "180+0"]
        [Result "1-0"]
        
        1. e4 { [%clk 0:03:00] } 1... e5 { [%clk 0:03:00] }
        2. Qh5 { [%clk 0:02:58] } 2... Nc6 { [%clk 0:02:59] }
        3. Bc4 { [%clk 0:02:55] } 3... Nf6 { [%clk 0:02:57] }
        4. Qxf7# { [%clk 0:02:53] } 1-0
    """
    
@pytest.fixture 
def _no_clock_pgn(): 
    """A PGN with no clock data — tests that None is handled gracefully."""
    return """
        [Event "Correspondence game"]
        [Site "https://lichess.org/testgame2"]
        [White "playerC"]
        [Black "playerD"]
        [WhiteElo "1200"]
        [BlackElo "1300"]
        [TimeControl "-"]
        [Result "0-1"]
        
        1. d4 d5 2. c4 e6 3. Nc3 Nf6 0-1
    """
    

@pytest.fixture
def unrated_pgn():
    """A PGN where players have no rating — tests safe_int handles '?'."""
    return """
        [Event "Casual game"]
        [Site "https://lichess.org/testgame3"]
        [White "playerE"]
        [Black "playerF"]
        [WhiteElo "?"]
        [BlackElo "?"]
        [TimeControl "600+0"]
        [Result "1/2-1/2"]
        
        1. e4 { [%clk 0:10:00] } e5 { [%clk 0:10:00] } 1/2-1/2
    """
    
def _read_pgn(pgn_str: str) -> chess.pgn.Game: 
    """Helper function to parse PGN string into object"""
    return chess.pgn.read_game(io.StringIO(pgn_str.strip()))

#----------Test Suite----------

# Parse Clock Tests
def test_parse_clock_normal():
    """Standard clock comment converts to correct seconds."""
    assert _parse_clock("{ [%clk 0:03:00] }") == 180.0
 
def test_parse_clock_with_hours():
    """Hours are included in the total."""
    assert _parse_clock("{ [%clk 1:30:00] }") == 5400.0
 
def test_parse_clock_zero():
    """Flagged (0:00:00) returns 0.0, not None."""
    assert _parse_clock("{ [%clk 0:00:00] }") == 0.0
 
def test_parse_clock_none_input():
    """None input returns None — no crash."""
    assert _parse_clock(None) is None
 
def test_parse_clock_no_clk_tag():
    """Comment without a clock tag returns None."""
    assert _parse_clock("{ just a comment }") is None
 
def test_parse_clock_empty_string():
    """Empty string returns None."""
    assert _parse_clock("") is None
    
    
#Parse Time Control Tests 

def test_time_control_bullet():
    base, inc, category = _parse_time_control("60+0")
    assert base == 60
    assert inc == 0
    assert category == "bullet"
 
def test_time_control_blitz():
    base, inc, category = _parse_time_control("180+0")
    assert base == 180
    assert category == "blitz"
 
def test_time_control_rapid():
    base, inc, category = _parse_time_control("600+0")
    assert base == 600
    assert category == "rapid"
 
def test_time_control_classical():
    base, inc, category = _parse_time_control("1800+0")
    assert base == 1800
    assert category == "classical"
 
def test_time_control_with_increment():
    """600+5 → estimated = 600 + 40*5 = 800 → rapid."""
    base, inc, category = _parse_time_control("600+5")
    assert inc == 5
    assert category == "rapid"
 
def test_time_control_unknown():
    """Malformed string returns safe defaults."""
    base, inc, category = _parse_time_control("-")
    assert base == 0
    assert category == "unknown"
 
# Parse Game Tests 


def test_parse_game_row_count(short_game_pgn):
    """Scholar's mate has 7 half-moves → 7 rows."""
    game = _read_pgn(short_game_pgn)
    rows = _parse_game(game)
    assert len(rows) == 7
 
def test_parse_game_colors_alternate(short_game_pgn):
    """Colors must alternate white/black/white/black..."""
    game = _read_pgn(short_game_pgn)
    rows = _parse_game(game)
    colors = [r.color for r in rows]
    assert colors == ["white", "black", "white", "black", "white", "black", "white"]
 
def test_parse_game_first_move(short_game_pgn):
    """First move should be e4 in UCI and SAN."""
    game = _read_pgn(short_game_pgn)
    rows = _parse_game(game)
    assert rows[0].move_uci == "e2e4"
    assert rows[0].move_san == "e4"
 
def test_parse_game_elo(short_game_pgn):
    """Elo ratings are read correctly from headers."""
    game = _read_pgn(short_game_pgn)
    rows = _parse_game(game)
    assert rows[0].white_elo == 1500
    assert rows[0].black_elo == 1450
 
def test_parse_game_unrated_elo(unrated_pgn):
    """'?' ratings are converted to 0, not a crash."""
    game = _read_pgn(unrated_pgn)
    rows = _parse_game(game)
    assert rows[0].white_elo == 0
    assert rows[0].black_elo == 0
 
def test_parse_game_clock_first_move(short_game_pgn):
    """
    First move clock_before = full base time (180s for 180+0).
    clock_after = 180s (white played instantly).
    time_spent = 0.0
    """
    game = _read_pgn(short_game_pgn)
    rows = _parse_game(game)
    assert rows[0].clock_before == 180.0   # full time, no grandparent
    assert rows[0].clock_after  == 180.0
    assert rows[0].time_spent   == 0.0
 
def test_parse_game_time_spent(short_game_pgn):
    """
    Move 3 is white's Qh5. White had 3:00 after move 1 (e4),
    and 2:58 after Qh5. So time_spent = 2.0s.
    """
    game = _read_pgn(short_game_pgn)
    rows = _parse_game(game)
    # rows[2] is the 3rd half-move (white's Qh5)
    assert rows[2].move_san    == "Qh5"
    assert rows[2].clock_before == 180.0  # white had 3:00 after e4
    assert rows[2].clock_after  == 178.0  # 2:58 = 178s
    assert rows[2].time_spent   == 2.0
 
def test_parse_game_no_clock(no_clock_pgn):
    """Games without clock data have None for all clock fields."""
    game = _read_pgn(no_clock_pgn)
    rows = _parse_game(game)
    for row in rows:
        assert row.clock_after  is None
        assert row.time_spent   is None
 
def test_parse_game_game_id(short_game_pgn):
    """Game ID is extracted from the Site URL correctly."""
    game = _read_pgn(short_game_pgn)
    rows = _parse_game(game)
    assert rows[0].game_id == "testgame1"
 
def test_parse_game_result(short_game_pgn):
    """Result is propagated to every row."""
    game = _read_pgn(short_game_pgn)
    rows = _parse_game(game)
    assert all(r.result == "1-0" for r in rows)
 
def test_parse_game_fen_changes(short_game_pgn):
    """FEN should be different for every move — board is advancing."""
    game = _read_pgn(short_game_pgn)
    rows = _parse_game(game)
    fens = [r.fen_before for r in rows]
    assert len(set(fens)) == len(fens)  # all unique
    