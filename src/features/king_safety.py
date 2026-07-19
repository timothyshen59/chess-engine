import chess

ATTACK_WEIGHTS = {
    chess.PAWN:   1,
    chess.KNIGHT: 2,
    chess.BISHOP: 2,
    chess.ROOK:   3,
    chess.QUEEN:  5,
}

TROPISM_WEIGHTS = {
    chess.QUEEN:  5,
    chess.ROOK:   3,
    chess.BISHOP: 2,
    chess.KNIGHT: 2,
    chess.PAWN:   1,
}


def compute_king_safety_score(board: chess.Board) -> int:
    """Computes heuristic for king safety based on pawn shield + attack on king position"""
    color = board.turn
    enemy = not color
    king  = board.king(color)
    if king is None:
        return 0

    king_file = chess.square_file(king)
    king_rank = chess.square_rank(king)

    # Pawn shield
    shield_rank = king_rank + 1 if color == chess.WHITE else king_rank - 1
    shield = 0
    if 0 <= shield_rank <= 7:
        for f in range(max(0, king_file - 1), min(7, king_file + 1) + 1):
            sq = chess.square(f, shield_rank)
            piece = board.piece_at(sq)
            if piece and piece.piece_type == chess.PAWN and piece.color == color:
                shield += 1

    # King zone
    king_zone = set()
    for dr in [-1, 0, 1]:
        for df in [-1, 0, 1]:
            r, f = king_rank + dr, king_file + df
            if 0 <= r <= 7 and 0 <= f <= 7:
                king_zone.add(chess.square(f, r))

    # Unique attackers on king zone
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


def compute_king_tropism(board: chess.Board) -> int:
    """Heuristic for how close enemy pieces are (distnace +piece weight) """
    color = board.turn
    enemy = not color
    king  = board.king(color)
    if king is None:
        return 0

    king_file = chess.square_file(king)
    king_rank = chess.square_rank(king)

    tropism = 0
    for pt, weight in TROPISM_WEIGHTS.items():
        for sq in board.pieces(pt, enemy):
            f = chess.square_file(sq)
            r = chess.square_rank(sq)
            distance = abs(f - king_file) + abs(r - king_rank)
            tropism += weight * (14 - distance)

    return -tropism


def compute_open_file_near_king(board: chess.Board) -> int:
    """
    Penalizes open files near king as open lanes for rooks and queens 
    """
    color = board.turn
    enemy = not color
    king  = board.king(color)
    if king is None:
        return 0

    king_file = chess.square_file(king)
    penalty   = 0

    for f in range(max(0, king_file - 1), min(7, king_file + 1) + 1):
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
        if not friendly_pawn:
            penalty += 2 if not enemy_pawn else 1  # open=2, semi-open=1

    return penalty


def compute_pawn_storm(board: chess.Board) -> int:
    """
    Count enemy pawns advancing toward the king.
    """
    color = board.turn
    enemy = not color
    king  = board.king(color)
    if king is None:
        return 0

    king_file = chess.square_file(king)
    king_rank = chess.square_rank(king)
    storm     = 0

    for sq in board.pieces(chess.PAWN, enemy):
        f = chess.square_file(sq)
        r = chess.square_rank(sq)

    
        if abs(f - king_file) > 2:
            continue

        # Distance of pawn from king — closer = more dangerous
        if color == chess.WHITE:
            distance = r - king_rank  # enemy pawns approach from above
        else:
            distance = king_rank - r

        if 0 < distance <= 4:  # within 4 ranks
            storm += (5 - distance)  # closer = higher score

    return storm


def compute_king_safety_features(board: chess.Board) -> dict:
    return {
        "king_safety_score":    compute_king_safety_score(board),
        "king_tropism":         compute_king_tropism(board),
        "open_file_near_king":  compute_open_file_near_king(board),
        "pawn_storm":           compute_pawn_storm(board),
    }