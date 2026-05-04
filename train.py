"""Train the tttt MLP on data/training.tsv and save the model.

Run with:  uv run python train.py
"""

import csv

import joblib
import numpy as np
from sklearn.neural_network import MLPClassifier

from board import O, str_to_board, to_features, to_target

DATA_FILE = "data/training.tsv"
MODEL_FILE = "data/model.joblib"


def load_training_data(path: str) -> tuple[np.ndarray, np.ndarray]:
    with open(path, newline="") as fh:
        records = [
            r
            for r in csv.DictReader(fh, delimiter="\t")
            if int(r["best_move"]) >= 0  # skip full-board draws
        ]

    X_feat = np.array([to_features(str_to_board(r["board"])) for r in records])
    y_target = np.array([to_target(int(r["best_move"]), piece=O) for r in records])
    return X_feat, y_target


def main() -> None:
    print(f"Loading training data from {DATA_FILE}…")
    X_feat, y_target = load_training_data(DATA_FILE)
    print(f"  {X_feat.shape[0]} samples, {X_feat.shape[1]} input features, "
          f"{y_target.shape[1]} outputs")

    model = MLPClassifier(
        hidden_layer_sizes=(16, 16),
        activation="logistic",   # sigmoid units — matches original PDP
        solver="lbfgs",          # recommended for small datasets
        max_iter=5000,
        random_state=42,
    )

    print("Training…")
    model.fit(X_feat, y_target)
    print(f"  Converged: {model.n_iter_} iterations")

    joblib.dump(model, MODEL_FILE)
    print(f"Model saved to {MODEL_FILE}")


if __name__ == "__main__":
    main()
