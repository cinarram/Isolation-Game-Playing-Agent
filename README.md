
# Building an intelligent Game-playing Agent for the game of Isolation

![Example game of isolation](viz.gif)

## Synopsis

This project developed an adversarial search agent to play the game "Isolation".  Isolation is a deterministic, two-player game of perfect information in which the players alternate turns moving a single piece from one cell to another on a board.  Whenever either player occupies a cell, that cell becomes blocked for the remainder of the game.  The first player with no remaining legal moves loses, and the opponent is declared the winner.  These rules are implemented in the `isolation.Board` class.

This project uses a version of Isolation where each agent is restricted to L-shaped movements (like a knight in chess) on a rectangular grid (like a chess or checkerboard).  The agents can move to any open cell on the board that is 2-rows and 1-column or 2-columns and 1-row away from their current position on the board. Movements are blocked at the edges of the board (the board does not wrap around), however, the player can "jump" blocked or occupied spaces (just like a knight in chess).

Additionally, agents will have a fixed time limit each turn to search for the best move and respond.  If the time limit expires during a player's turn, that player forfeits the match, and the opponent wins.

The gist of the development work is contained in the `game_agent.py` file that implements Minimax search with Alpha-Beta Pruning and Iterative Deepening for a strong AI player.  Additional files include example Players and evaluation functions and the game board class.


## Where Artificial Intelligence comes in

The meat of the AI part is in the `game_agent.py` file that implements the following:

- `MinimaxPlayer.minimax()`: minimax search
- `AlphaBetaPlayer.alphabeta()`: minimax search with alpha-beta pruning
- `AlphaBetaPlayer.get_move()`: iterative deepening search
- `custom_score()`: position evaluation heuristic No. 1
- `custom_score_2()`: position evaluation heuristic No. 2
- `custom_score_3()`: position evaluation heuristic No. 3

The custom heuristics are:
(1) A late-game longest-path approach, where the longest paths for both players is calculated and a board situation where the active player has a longer path is favoured;
(2) A simple MyMoves vs OpponentMoves function, where the opponent's number of moves is weighted double -- this rather simple evaluation function has shown to be quite effective;
(3) A heuristic focussing on the importance of the centre of the board and tries to maximise moves that keep the player close the centre.

All these heuristics are independent of each other -- evaluating all of them, heuristic #2 has shown to perform the best.


## Game Visualization

The `isoviz` folder contains a modified version of chessboard.js that can animate games played on a 7x7 board.  In order to use the board, you must run a local webserver by running `python -m http.server 8000` from your project directory (you can replace 8000 with another port number if that one is unavailable), then open your browser to `http://localhost:8000` and navigate to the `/isoviz/display.html` page.  Enter the move history of an isolation match (i.e., the array returned by the Board.play() method) into the text area and run the match.  Refresh the page to run a different game.  (Feel free to submit pull requests with improvements to isoviz.)


