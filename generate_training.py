"""Generate data/training.tsv and data/test.tsv from computed canonical O-to-move positions.

Run with:  uv run python generate_training.py
"""

import csv
import os

import numpy as np

from board import SYMMETRIES, X, board_to_str, generate_o_to_move, minimax, str_to_board


def display(board_str: str) -> str:
    return f"{board_str[:3]}/{board_str[3:6]}/{board_str[6:]}"


def write_tsv(path: str, records: list[dict]) -> None:
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t")
        writer.writerow(["board", "best_move"])
        for r in records:
            writer.writerow([r["board"], r["best_move"]])


def generate_test_records(training_records: list[dict]) -> list[dict]:
    """All non-canonical D4 variants of the training positions.

    For each canonical board C with best_move m, apply each of the 7
    non-identity symmetry permutations.  The best move on the variant V
    is derived via the inverse permutation: if V = C[perm] then the
    cell of V that corresponds to C[m] is inv_perm[m].
    """
    seen: set[str] = {r["board"] for r in training_records}
    test: list[dict] = []
    for rec in training_records:
        c_board = str_to_board(rec["board"])
        c_move = rec["best_move"]
        for j in range(1, 8):  # skip identity (j=0)
            perm = SYMMETRIES[j]
            v_str = board_to_str(c_board[perm])
            if v_str in seen:
                continue
            seen.add(v_str)
            v_move = -1 if c_move < 0 else int(np.argsort(perm)[c_move])
            test.append({"board": v_str, "best_move": v_move})
    return test


def main() -> None:
    print("Generating canonical O-to-move positions…")
    positions = generate_o_to_move()
    print(f"  {len(positions)} canonical positions found")

    first_move = [p for p in positions if list(p).count(X) == 1]
    print(f"\nFirst-move canonical positions ({len(first_move)} expected: 1):")
    for b in sorted(first_move, key=lambda b: tuple(b)):
        print(f"  {display(board_to_str(b))}")

    training: list[dict] = []
    for board in positions:
        _, best_move = minimax(board.copy(), x_to_move=False)
        training.append({"board": board_to_str(board), "best_move": best_move})
    training.sort(key=lambda r: r["board"])

    os.makedirs("data", exist_ok=True)
    write_tsv("data/training.tsv", training)
    print(f"\nWrote {len(training)} records to data/training.tsv")

    test = generate_test_records(training)
    test.sort(key=lambda r: r["board"])
    write_tsv("data/test.tsv", test)
    print(f"Wrote {len(test)} records to data/test.tsv")


if __name__ == "__main__":
    main()
