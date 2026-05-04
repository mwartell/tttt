"""Core board representation, symmetry, and minimax for tttt."""

import numpy as np

EMPTY, X, O = np.int8(0), np.int8(1), np.int8(2)

# Cell layout:
#   0 1 2
#   3 4 5
#   6 7 8
#
# Each row is a D4 symmetry transform: new_board[i] = old_board[perm[i]]
SYMMETRIES = np.array(
    [
        [0, 1, 2, 3, 4, 5, 6, 7, 8],  # identity
        [6, 3, 0, 7, 4, 1, 8, 5, 2],  # rotate 90° CW
        [8, 7, 6, 5, 4, 3, 2, 1, 0],  # rotate 180°
        [2, 5, 8, 1, 4, 7, 0, 3, 6],  # rotate 270° CW
        [2, 1, 0, 5, 4, 3, 8, 7, 6],  # reflect left↔right
        [6, 7, 8, 3, 4, 5, 0, 1, 2],  # reflect top↔bottom
        [0, 3, 6, 1, 4, 7, 2, 5, 8],  # reflect main diagonal
        [8, 5, 2, 7, 4, 1, 6, 3, 0],  # reflect anti-diagonal
    ],
    dtype=np.intp,
)

WIN_LINES = [
    (0, 1, 2), (3, 4, 5), (6, 7, 8),  # rows
    (0, 3, 6), (1, 4, 7), (2, 5, 8),  # columns
    (0, 4, 8), (2, 4, 6),              # diagonals
]


def empty_board() -> np.ndarray:
    return np.zeros(9, dtype=np.int8)


def is_won(board: np.ndarray, piece: np.int8) -> bool:
    for a, b, c in WIN_LINES:
        if board[a] == board[b] == board[c] == piece:
            return True
    return False


def canonical(board: np.ndarray) -> tuple[np.ndarray, int]:
    """Return (lexically-greatest equivalent board, symmetry index used)."""
    variants = [board[perm] for perm in SYMMETRIES]
    idx = max(range(8), key=lambda i: tuple(variants[i]))
    return variants[idx].copy(), idx


def board_to_str(board: np.ndarray) -> str:
    return "".join(".xo"[c] for c in board)


def str_to_board(s: str) -> np.ndarray:
    return np.array([".xo".index(c) for c in s], dtype=np.int8)


def to_features(board: np.ndarray) -> np.ndarray:
    """18-bit input: cell i → bits [2i]=X-bit, [2i+1]=O-bit."""
    features = np.zeros(18, dtype=np.float32)
    for i, cell in enumerate(board):
        if cell == X:
            features[2 * i] = 1.0
        elif cell == O:
            features[2 * i + 1] = 1.0
    return features


def to_target(move: int, piece: np.int8 = O) -> np.ndarray:
    """18-bit output: one-hot move encoding for the given piece."""
    target = np.zeros(18, dtype=np.float32)
    if piece == X:
        target[2 * move] = 1.0
    else:
        target[2 * move + 1] = 1.0
    return target


def minimax(board: np.ndarray, x_to_move: bool) -> tuple[int, int]:
    """Return (score, best_move_cell). Score: +1=X wins, -1=O wins, 0=draw.
    best_move_cell is -1 when no moves remain."""
    if is_won(board, X):
        return 1, -1
    if is_won(board, O):
        return -1, -1
    empties = np.where(board == EMPTY)[0]
    if len(empties) == 0:
        return 0, -1

    piece = X if x_to_move else O
    best_move = int(empties[0])
    best_score = -2 if x_to_move else 2

    for move in empties:
        board[move] = piece
        score, _ = minimax(board, not x_to_move)
        board[move] = EMPTY
        if x_to_move and score > best_score:
            best_score, best_move = score, int(move)
        elif not x_to_move and score < best_score:
            best_score, best_move = score, int(move)

    return best_score, best_move


CORNERS = np.array([0, 2, 6, 8], dtype=np.intp)


def generate_o_to_move() -> list[np.ndarray]:
    """All distinct canonical non-terminal boards where O moves next,
    reachable when X always plays its single minimax-best move.

    X's first move is cell 0 (canonical corner representative).
    All subsequent X moves use the single minimax-best move.
    O explores all legal responses."""
    seen: set[tuple] = set()
    results: list[np.ndarray] = []

    def recurse(board: np.ndarray, x_to_move: bool) -> None:
        if x_to_move:
            if not np.any(board == X):
                move = 0  # canonical corner representative
            else:
                _, move = minimax(board.copy(), x_to_move=True)
                if move < 0:
                    return  # terminal
            board[move] = X
            if not is_won(board, X):
                c, _ = canonical(board)
                key = tuple(c)
                if key not in seen:
                    seen.add(key)
                    results.append(c)
                recurse(board, x_to_move=False)
            board[move] = EMPTY
        else:
            for move in np.where(board == EMPTY)[0]:
                board[move] = O
                if not is_won(board, O):
                    recurse(board, x_to_move=True)
                board[move] = EMPTY

    recurse(empty_board(), x_to_move=True)
    return results
