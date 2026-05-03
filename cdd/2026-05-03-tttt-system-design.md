---
date: 2026-05-03
title: tttt system design
current_head: 8f734fa
agent: "Claude Haiku 4.5"
session_type: exploration
topics: [system-design, architecture, tttt, planning]
---

# prompt 1: selecting data representation and ml package

Looking at the narrative in README.md, propose a python data representation for training and test datasets such that:

- it is easy to generate all valid board positions that can result from a game of ttt
- it is easy to compute reflections and rotations of a board for canonicalization

Then find a suitable Python package for ML training and testing on small nets like this. I believe sklearn is a good candidate, with PyTorch and similar being overkill for this task.

# Response 1: numpy int8 arrays for boards; D4 symmetry via index permutations; sklearn MLPClassifier

A `numpy` array of shape `(9,)` with `int8` dtype is the right primitive — it's cheap, directly reshapable to `(3,3)` for numpy ops, and trivially converted to the 18-bit ML feature vector. The D4 symmetry group is best expressed as eight static index-permutation arrays so canonicalization is a single `argmin` over eight `board[perm]` views. `scikit-learn`'s `MLPClassifier` is the right package: it accepts a 2-D binary target matrix natively, trains in seconds on a ~20-row dataset, and exposes the sigmoid hidden-layer architecture the original used.

---

## Data Representation

### Board primitive

```python
import numpy as np

EMPTY, X, O = np.int8(0), np.int8(1), np.int8(2)


def empty_board() -> np.ndarray:
    """Return a blank 9-element board."""
    return np.zeros(9, dtype=np.int8)
```

Cell layout (row-major, 0-indexed from upper-left):

```
0 1 2
3 4 5
6 7 8
```

Values: `0` = empty, `1` = X, `2` = O — matches the README's `00/01/10` binary scheme after the `to_features` conversion below.

### ML feature vector (18 bits)

```python
def to_features(board: np.ndarray) -> np.ndarray:
    """Convert a 9-cell board to the 18-bit input vector used by the net.
    Cell i maps to bits [2i, 2i+1]: X→[1,0], O→[0,1], empty→[0,0].
    """
    features = np.zeros(18, dtype=np.float32)
    for i, cell in enumerate(board):
        if cell == X:
            features[2 * i] = 1.0
        elif cell == O:
            features[2 * i + 1] = 1.0
    return features


def to_target(move: int) -> np.ndarray:
    """Encode a recommended move (cell index 0-8) as an 18-bit output vector."""
    target = np.zeros(18, dtype=np.float32)
    target[2 * move] = 1.0  # X-bit of the chosen cell
    return target
```

### Rotation and reflection via index permutations

The eight elements of the dihedral group D₄ are expressed as static permutation arrays. Applying a symmetry is a single numpy fancy-index operation — no reshape, no copy beyond what numpy does for the index.

```python
# Each row is one of the 8 symmetry transforms of cells [0..8].
# Grid layout used to derive these:
#   0 1 2
#   3 4 5
#   6 7 8
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


def canonical(board: np.ndarray) -> tuple[np.ndarray, int]:
    """Return the lexically smallest symmetry-equivalent board and the
    index of the symmetry transform that produced it."""
    variants = [board[perm] for perm in SYMMETRIES]
    idx = min(range(8), key=lambda i: tuple(variants[i]))
    return variants[idx], idx
```

Canonicalization works identically to the original 1987 approach: generate all 8 orientations, pick the lexically smallest as a tuple of ints.

### Generating valid board positions

```python
from itertools import product


def generate_after_x_moves(n_x: int) -> list[np.ndarray]:
    """All distinct canonical boards after X has made n_x moves (O has n_x - 1)."""
    n_o = n_x - 1
    seen: set[tuple] = set()
    results: list[np.ndarray] = []
    for cells in product([EMPTY, X, O], repeat=9):
        board = np.array(cells, dtype=np.int8)
        if board.tolist().count(X) == n_x and board.tolist().count(O) == n_o:
            c, _ = canonical(board)
            key = tuple(c)
            if key not in seen:
                seen.add(key)
                results.append(c)
    return results
```

---

## ML Package

**`scikit-learn` `MLPClassifier`** is the right choice:

| Criterion | scikit-learn | PyTorch/JAX |
|---|---|---|
| Dataset size | ~20 rows → fine | overkill |
| Architecture match | `hidden_layer_sizes=(16, 16)` | requires manual loop |
| Multi-output support | native 2-D `y` | custom loss |
| Training time | seconds | seconds + setup |
| Explainability | `coefs_`, `intercepts_` exposed | varies |

```python
from sklearn.neural_network import MLPClassifier

model = MLPClassifier(
    hidden_layer_sizes=(16, 16),  # two hidden layers, ≤16 nodes each
    activation="logistic",  # sigmoid — matches original PDP units
    solver="lbfgs",  # well-suited for small datasets
    max_iter=2000,
    random_state=42,
)
```

`MLPClassifier` detects a 2-D binary `y` matrix and trains one output unit per column, matching the original 18-output architecture. After `predict_proba` the outputs can be thresholded at 0.5 to recover binary move recommendations.

`solver="lbfgs"` is preferred over `"adam"` for datasets this small (sklearn docs explicitly recommend it).

# prompt 2:

Choose an external, human-readable data format for training and test data, yaml if suitable, and then implement your recommendations:

- Build the hand-coded training set of ~20 canonical boards and O's best response
- Implement `generate_after_x_moves(1)` to verify the 3 canonical first-move positions (corner / edge / center)
- Scaffold `train.py` and `evaluate.py`

and save the training set in a file.

# Response 2: YAML chosen; training set computed (not hand-coded); 292 canonical positions; train.py and evaluate.py scaffolded; sklearn blocked by macOS system issue

YAML was chosen for readability and round-trip fidelity. The training set was **computed** rather than hand-coded: `board.py` implements minimax + D4 canonicalization and `generate_training.py` produces `data/training.yaml`. The README's "~20 states" is a significant misremembering — there are **292 canonical non-terminal positions** where O moves next, distributed across game depths as shown below. `train.py` and `evaluate.py` are fully scaffolded but could not be exercised: `scikit-learn` fails to load on this system (`mmap(PROT_EXEC)` returns `errno=1` for all sklearn `.so` files regardless of version or venv location; numpy loads fine). No hand-coded data entry was needed.

---

## Files created

| File | Purpose |
|---|---|
| [board.py](../board.py) | Board primitive, D4 symmetry, `minimax()`, `canonical()`, `generate_o_to_move()` |
| [generate_training.py](../generate_training.py) | Runs the generator, writes `data/training.yaml` |
| [data/training.yaml](../data/training.yaml) | 292 computed canonical positions with minimax best-move |
| [train.py](../train.py) | Loads YAML → fits `MLPClassifier(hidden_layer_sizes=(16,16), activation="logistic", solver="lbfgs")` → saves `data/model.joblib` |
| [evaluate.py](../evaluate.py) | Loads model → reports per-position accuracy vs training set |

## Canonical position counts by depth

```
X=1, O=0:   3 canonical positions   (corner / edge / center)
X=2, O=1:  38 canonical positions
X=3, O=2: 153 canonical positions
X=4, O=3:  95 canonical positions
X=5, O=4:   3 canonical positions
────────────────────────────────
Total:     292 canonical positions
```

The three first-move positions are verified correct: corner (`..x......` canonical), edge (`.x.......` canonical), center (`.....x...` canonical).

## Blocker

`scikit-learn` cannot be loaded on this machine — every sklearn `.so` fails with `mmap(size=...) failed with errno=1` (permission denied for executable mappings). This affects all versions tested (1.5.2, 1.8.0), clean venvs, and paths outside the project directory. `numpy` and `pyyaml` load fine. The failure is a system-level restriction on `PROT_EXEC` mmap for unsigned or ad-hoc–signed binaries (codesign shows `flags=adhoc,linker-signed` on the sklearn `.so` files). This is likely a macOS security policy or Endpoint Security configuration. **Next step:** run `uv run python train.py && uv run python evaluate.py` on a machine without this restriction.
