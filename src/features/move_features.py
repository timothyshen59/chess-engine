import chess

PIECE_VALUES = {
    chess.PAWN:   100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK:   500,
    chess.QUEEN:  900,
}
PIECE_TYPE_MAP = {
    chess.PAWN:   1,
    chess.KNIGHT: 2,
    chess.BISHOP: 3,
    chess.ROOK:   4,
    chess.QUEEN:  5,
    chess.KING:   6,
}


def compute_capture_value(board: chess.Board, move: chess.Move) -> int:
    if board.is_capture(move):
        captured = board.piece_at(move.to_square)
        if captured is None and board.is_en_passant(move):
            return PIECE_VALUES[chess.PAWN]
        return PIECE_VALUES.get(captured.piece_type, 0) if captured else 0
    return 0


def compute_is_check(board: chess.Board, move: chess.Move) -> int:
    return 1 if board.gives_check(move) else 0


def compute_is_castling(board: chess.Board, move: chess.Move) -> int:
    return 1 if board.is_castling(move) else 0


def compute_is_promotion(board: chess.Board, move: chess.Move) -> int:
    return 1 if move.promotion is not None else 0


def compute_piece_type_moved(board: chess.Board, move: chess.Move) -> int:
    piece = board.piece_at(move.from_square)
    return PIECE_TYPE_MAP.get(piece.piece_type, 0) if piece else 0


def compute_move_distance(move: chess.Move) -> int:
    from_f = chess.square_file(move.from_square)
    from_r = chess.square_rank(move.from_square)
    to_f   = chess.square_file(move.to_square)
    to_r   = chess.square_rank(move.to_square)
    return abs(to_f - from_f) + abs(to_r - from_r)


def compute_see(board: chess.Board, move: chess.Move) -> int:
    """
    Computes Static Exchange Evaluation (SEE) — estimates material gain/loss
    """
    if not board.is_capture(move):
        return 0

    captured = board.piece_at(move.to_square)
    is_ep = board.is_en_passant(move)
    if captured is None and not is_ep:
        return 0


    gain = PIECE_VALUES[chess.PAWN] if is_ep else PIECE_VALUES.get(captured.piece_type, 0)


    attacker = board.piece_at(move.from_square)
    attacker_value = PIECE_VALUES.get(attacker.piece_type, 0) if attacker else 0

    board.push(move)
    can_recapture = any(m.to_square == move.to_square for m in board.legal_moves)
    if can_recapture:
        gain -= attacker_value
    board.pop()

    return gain


def compute_move_features(board: chess.Board, move: chess.Move) -> dict:
    return {
        "capture_value":    compute_capture_value(board, move),
        "is_check":         compute_is_check(board, move),
        "is_castling":      compute_is_castling(board, move),
        "is_promotion":     compute_is_promotion(board, move),
        "piece_type_moved": compute_piece_type_moved(board, move),
        "move_distance":    compute_move_distance(move),
        "see_score":        compute_see(board, move),
    }