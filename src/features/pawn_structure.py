import chess


def compute_passed_pawns(board: chess.Board) -> int:
    """Computes passed pawns where a pawn is free to move without enemy pawn in same/adjacent files"""
    color = board.turn
    enemy = not color
    pawns = board.pieces(chess.PAWN, color)
    ep    = board.pieces(chess.PAWN, enemy)
    count = 0

    for sq in pawns:
        f = chess.square_file(sq)
        r = chess.square_rank(sq)
        files_to_check = [f_ for f_ in [f-1, f, f+1] if 0 <= f_ <= 7]
        ranks_ahead = range(r + 1, 8) if color == chess.WHITE else range(0, r)

        blocked = any(
            chess.square_file(eq) in files_to_check and
            chess.square_rank(eq) in ranks_ahead
            for eq in ep
        )
        if not blocked:
            count += 1
    return count


def compute_doubled_pawns(board: chess.Board) -> int:
    """Compute pawns on the same file (bad) """
    color = board.turn
    pawns = board.pieces(chess.PAWN, color)
    file_counts = {}
    for sq in pawns:
        f = chess.square_file(sq)
        file_counts[f] = file_counts.get(f, 0) + 1
    return sum(max(0, c - 1) for c in file_counts.values())


def compute_isolated_pawns(board: chess.Board) -> int:
    """Compute isolated pawns with no friendly pawns on either file"""
    color = board.turn
    pawns = board.pieces(chess.PAWN, color)
    pawn_files = set(chess.square_file(sq) for sq in pawns)
    count = 0
    for f in pawn_files:
        if (f - 1) not in pawn_files and (f + 1) not in pawn_files:
            count += sum(1 for sq in pawns if chess.square_file(sq) == f)
    return count


def compute_backward_pawns(board: chess.Board) -> int:
    """Computes pawns that are "backward" - pawns with no supporting structure + going into enemy pawn"""
    color = board.turn
    enemy = not color
    pawns = board.pieces(chess.PAWN, color)
    count = 0

    for sq in pawns:
        f = chess.square_file(sq)
        r = chess.square_rank(sq)

        front_rank = r + 1 if color == chess.WHITE else r - 1
        if not (0 <= front_rank <= 7):
            continue
        front_sq = chess.square(f, front_rank)

        enemy_controls_front = bool(
            board.attackers(enemy, front_sq) & board.pieces(chess.PAWN, enemy)
        )

        support_ranks = range(0, r + 1) if color == chess.WHITE else range(r, 8)
        has_support = any(
            chess.square_file(sq2) in [f - 1, f + 1] and
            chess.square_rank(sq2) in support_ranks
            for sq2 in board.pieces(chess.PAWN, color)
        )

        if enemy_controls_front and not has_support:
            count += 1

    return count


def compute_pawn_advancement(board: chess.Board) -> int:
    """Computes how far pawns have advanced from original ranks"""
    color = board.turn
    pawns = board.pieces(chess.PAWN, color)
    total = 0
    for sq in pawns:
        r = chess.square_rank(sq)
        total += (r - 1) if color == chess.WHITE else (6 - r)
    return total


def compute_pawn_structure_features(board: chess.Board) -> dict:
    return {
        "passed_pawn_count":   compute_passed_pawns(board),
        "doubled_pawn_count":  compute_doubled_pawns(board),
        "isolated_pawn_count": compute_isolated_pawns(board),
        "backward_pawn_count": compute_backward_pawns(board),
        "pawn_advancement":    compute_pawn_advancement(board),
    }