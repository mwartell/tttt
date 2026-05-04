"""Convert a training/test TSV to an HTML board visualization.

Run with:  uv run python visualize.py data/training.tsv > data/training.html
           uv run python visualize.py data/test.tsv     > data/test.html
"""

import csv
import sys

CSS = """
* { box-sizing: border-box; }
body {
    font-family: monospace;
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    padding: 16px;
    background: #f8f8f8;
}
table.board {
    border-collapse: collapse;
    border: 2px solid #444;
}
table.board td {
    width: 28px;
    height: 28px;
    text-align: center;
    font-size: 18px;
    font-weight: bold;
    border: 1px solid #aaa;
}
table.board.no-move {
    background: #c8c8c8;
    border-color: #888;
}
td.x     { color: #2255bb; }
td.o     { color: #777;    }
td.best  { color: #cc2222; }
td.empty { color: #ccc;    }
"""


def board_html(board_str: str, best_move: int) -> str:
    """Render one board position as an HTML table.

    Existing pieces are shown lowercase; the best move is shown as a
    capital O in red.  Full-board positions (best_move == -1) get a
    grey background to indicate no move is available.
    """
    no_move_cls = " no-move" if best_move < 0 else ""
    cells = []
    for i, ch in enumerate(board_str):
        if i == best_move:
            cells.append('<td class="best">O</td>')
        elif ch == "x":
            cells.append('<td class="x">x</td>')
        elif ch == "o":
            cells.append('<td class="o">o</td>')
        else:
            cells.append('<td class="empty">\u00b7</td>')
    rows = "".join(f"<tr>{''.join(cells[r * 3 : (r + 1) * 3])}</tr>" for r in range(3))
    return f'<table class="board{no_move_cls}">{rows}</table>\n'


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else "data/training.tsv"
    with open(path, newline="") as fh:
        records = list(csv.DictReader(fh, delimiter="\t"))

    title = path
    print(f"<!DOCTYPE html>")
    print(f"<html>")
    print(f"<head><meta charset='utf-8'><title>{title}</title>")
    print(f"<style>{CSS}</style>")
    print(f"</head>")
    print(f"<body>")
    for r in records:
        print(board_html(r["board"], int(r["best_move"])), end="")
    print(f"</body>")
    print(f"</html>")


if __name__ == "__main__":
    main()
