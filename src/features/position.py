import chess

PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,   
}


def compute_num_legal_moves(board: chess.Board) -> int:
    """Compute legal moves available in position for player"""
    return len(list(board.legal_moves))


def compute_material_balance(board: chess.Board) -> int:
    """Compute material balance in centipawns from moving player's perspective"""
    white_material = sum(PIECE_VALUES[pt] * len(board.pieces(pt, chess.WHITE)) for pt in PIECE_VALUES)
    black_material = sum(PIECE_VALUES[pt] * len(board.pieces(pt, chess.BLACK)) for pt in PIECE_VALUES)

    balance = white_material - black_material

    return balance if board.turn == chess.WHITE else -balance


def compute_piece_mobility(board: chess.Board) -> int:
    """Computes difference between player's versus opponent's legal moves"""
    my_moves = len(list(board.legal_moves))
    try:
        board.push(chess.Move.null())
        opp_moves = len(list(board.legal_moves))
        board.pop()
    except Exception:
        return 0

    return my_moves - opp_moves


def compute_bishop_pair(board: chess.Board) -> int:
    """Computes boolean value for if bishop pair is present"""
    return 1 if len(board.pieces(chess.BISHOP, board.turn)) >= 2 else 0


def compute_rook_on_open_file(board: chess.Board) -> int:
    """Computes player's rook's access to open files (no pawns of either color)"""
    color = board.turn
    rooks = board.pieces(chess.ROOK, color)

    count = 0

    for sq in rooks:
        f = chess.square_file(sq)
        file_squares = [chess.square(f, r) for r in range(8)]
      
        has_pawn = any(
            board.piece_at(s) and board.piece_at(s).piece_type == chess.PAWN
            for s in file_squares
        )

        if not has_pawn:
            count += 1

    return count


def compute_center_control(board: chess.Board) -> int:
    """Compute center control for current player

    Pawn control is weighted heavily in this function
    """
    PAWN_WEIGHT = 2
    OCCUPANT_BONUS = 1

    color = board.turn
    center = [chess.E4, chess.E5, chess.D4, chess.D5]

    score = 0

    for sq in center:
        occupant = board.piece_at(sq)

        if occupant and occupant.piece_type == chess.PAWN and occupant.color == color:
            score += OCCUPANT_BONUS

        for attacker_sq in board.attackers(color, sq):
            attacker = board.piece_at(attacker_sq)
            if attacker.piece_type == chess.PAWN:
                score += PAWN_WEIGHT
            else:
                score += 1  # Just +1 for default other pieces

    return score


def compute_rook_on_semi_open_file(board: chess.Board) -> int:
    """Compute semi-open file (file without player's pawn, but with enemy pawn) for position"""

    color = board.turn
    enemy = not color
    rooks = board.pieces(chess.ROOK, color)
    count = 0

    for sq in rooks:
        f = chess.square_file(sq)
        file_squares = [chess.square(f, r) for r in range(8)]
        friendly_pawn = any(
            board.piece_at(s) and board.piece_at(s).piece_type == chess.PAWN
            and board.piece_at(s).color == color
            for s in file_squares
        )
        enemy_pawn = any(
            board.piece_at(s) and board.piece_at(s).piece_type == chess.PAWN
            and board.piece_at(s).color == enemy
            for s in file_squares
        )

        if not friendly_pawn and enemy_pawn:
            count += 1

    return count


def compute_space_control(board: chess.Board) -> int:
    """Computes space control with Stockfish-style function"""
    color = board.turn
    enemy = not color

    files = range(2, 6)
    ranks = range(1, 4) if color == chess.WHITE else range(4, 7)

    space_mask = [chess.square(f, r) for f in files for r in ranks]

    own_pawns = board.pieces(chess.PAWN, color)

    score = 0
    for sq in space_mask:
        occupant = board.piece_at(sq)
        
        if occupant and occupant.piece_type == chess.PAWN and occupant.color == color:
            continue
  
        if board.attackers(enemy, sq) & board.pieces(chess.PAWN, enemy):
            continue

        score += 1

        f = chess.square_file(sq)
        r = chess.square_rank(sq)  

     
        behind_ranks = range(0, r) if color == chess.WHITE else range(r + 1, 8)

        pawn_behind = any(
            chess.square(f, br) in own_pawns for br in behind_ranks
        )
        if pawn_behind and not board.attackers(enemy, sq):
            score += 1

    return score


def compute_position_features(board: chess.Board) -> dict:
    """Compute position features"""
    return {
        "num_legal_moves":        compute_num_legal_moves(board),
        "material_balance":       compute_material_balance(board),
        "piece_mobility":         compute_piece_mobility(board),
        "bishop_pair":            compute_bishop_pair(board),
        "rook_on_open_file":      compute_rook_on_open_file(board),
        "rook_on_semi_open_file": compute_rook_on_semi_open_file(board),
        "center_control":         compute_center_control(board),
        "space_score":            compute_space_control(board),
    }