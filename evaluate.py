"""Evaluate the trained tttt model against the training set.

Run with:  uv run python evaluate.py
"""

import csv

import joblib
import numpy as np

from board import O, str_to_board, to_features

DATA_FILE = "data/training.tsv"
MODEL_FILE = "data/model.joblib"


def display(board_str: str) -> str:
    return f"{board_str[:3]}/{board_str[3:6]}/{board_str[6:]}"


def decode_move(output_vec: np.ndarray, piece: np.int8 = O) -> int:
    """Return the cell index with the highest activation for the given piece."""
    offset = 0 if piece == 1 else 1   # X-bit=0, O-bit=1
    o_bits = output_vec[offset::2]    # bits for every cell's piece
    return int(np.argmax(o_bits))


def main() -> None:
    model = joblib.load(MODEL_FILE)
    print(f"Loaded model from {MODEL_FILE}")

    with open(DATA_FILE, newline="") as fh:
        records = [
            {"board": r["board"], "best_move": int(r["best_move"])}
            for r in csv.DictReader(fh, delimiter="\t")
            if int(r["best_move"]) >= 0
        ]

    X_feat = np.array([to_features(str_to_board(r["board"])) for r in records])
    predictions = model.predict(X_feat)

    correct = total = 0
    wrong_examples: list[dict] = []

    for record, pred_vec in zip(records, predictions):
        predicted_move = decode_move(pred_vec, piece=O)
        expected_move = record["best_move"]
        total += 1
        if predicted_move == expected_move:
            correct += 1
        else:
            wrong_examples.append({
                "board": display(record["board"]),
                "expected": expected_move,
                "predicted": predicted_move,
            })

    pct = 100 * correct / total
    print(f"\nAccuracy: {correct}/{total} ({pct:.1f}%)")

    if wrong_examples:
        print(f"\nIncorrect predictions ({len(wrong_examples)}):")
        for ex in wrong_examples:
            print(f"  {ex['board']}  expected={ex['expected']}  got={ex['predicted']}")
    else:
        print("All predictions correct.")


if __name__ == "__main__":
    main()
