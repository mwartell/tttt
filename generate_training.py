"""Generate data/training.yaml from computed canonical O-to-move positions.

Run with:  uv run python generate_training.py
"""

import os
import yaml
from board import generate_o_to_move, minimax, board_to_str, X, O

def display(board_str: str) -> str:
    """Pretty-print a board string as three slash-separated rows."""
    return f"{board_str[:3]}/{board_str[3:6]}/{board_str[6:]}"


def main() -> None:
    print("Generating canonical O-to-move positions…")
    positions = generate_o_to_move()
    print(f"  {len(positions)} canonical positions found")

    # Verify first-move positions (X=1, O=0)
    first_move = [p for p in positions if list(p).count(X) == 1]
    print(f"\nFirst-move canonical positions ({len(first_move)} expected: 3):")
    for b in sorted(first_move, key=lambda b: tuple(b)):
        print(f"  {display(board_to_str(b))}")

    # Compute optimal O response for every position
    training = []
    for board in positions:
        _, best_move = minimax(board.copy(), x_to_move=False)
        training.append(
            {
                "board": board_to_str(board),
                "best_move": best_move,
                "display": display(board_to_str(board)),
            }
        )

    # Sort by board string for stable, readable output
    training.sort(key=lambda r: r["board"])

    os.makedirs("data", exist_ok=True)
    with open("data/training.yaml", "w") as fh:
        yaml.dump(
            {"training": training},
            fh,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        )
    print(f"\nWrote {len(training)} records to data/training.yaml")


if __name__ == "__main__":
    main()
