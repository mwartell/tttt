# tic-tac-toe pdp simulation

In 1987 I had written a Parallel Distributed Processing program to learn the optimal move in tic-tac-toe for a given board position.

Following McClelland and Rumelhart (1986), and using their PDP library on a 25MHz x386 PC, it had 18 binary inputs representing the current board, and 18 binary outputs representing the best next move. Each cell of the 9x9 board could either be 00 for empty, 01, for X and 10 for O.

I had created a hand-coded training set containing all possible board states after X had moved. I had canonicalized the boards across rotation and reflection so as to see if the trained net could extrapolate from the training set from the canonical boards it learned to all possible boards. For example if X's first move is into a corner cell, there are four possible board positions with X in the four corners, this canonicalizes into one state with an X in the upper left cell. After performing all rotations and reflections, I simply chose the version that was lexically smallest.

## nomenclature

The cells of a tic-tac-toe board are number from 0 to 8, starting from upper left and proceeding right to left and down the rows. For example, a board position of

```
xo.
...
...
```

is represented as the string `xo.......` which can be displayed as `xo./.../...` where the slashes help a human reader separate the rows. The binary encoding of that board state is `011000_000000_000000` with underscores for visual clarity.

## network

With the input and output layers fixed by the task, I used two hidden layers of small dimension because training could take a whole night given the available technology at the time. I think the hidden layers had a maximum node count of 16 each, but it might have been less or not much more.

## also sprach zarathustra

For the set of canonical boards where X has already moved I hand coded O's best move as the desired output. I think there are about 20 canonical states.

I had used the canonical form because I was curious to see if trained on just the subset of all states, it could extrapolate to valid states not in the training set. I set the supervised learning to run overnight and found that it learned all but two of the desired outputs. I fiddled with the code and then ran it again overnight. The second training returned the same two erroneous moves as the night before. So I decoded the two bad moves and found that I had made an error in the training data and the network was responding with a better move.

I walked home that night astounded that with only about 18 valid training states, the net settled to better answers than I had. I don't think I'm anthropomorphizing that.

This repo exists because in the past 39 years, we've gotten better at ML explainability and training of those tiny nets likely takes minutes rather than hours.
