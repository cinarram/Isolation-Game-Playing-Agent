import random


class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass


def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """

    if game.is_loser(player):
        return float("-inf")
    if game.is_winner(player):
        return float("inf")

    # Longest Path Heuristic (used towards end game)

    game_phase = len(game.get_blank_spaces()) # high if early, low if late in game
    max_phase = game.width*game.height

    def longestPath(player,game,path=0,longest=0):
        moves = game.get_legal_moves(player)
        if path > longest:
            longest = path
        if len(moves) == 0:
            path = 0
        for move in moves:
            new_board = game.forecast_move(move)
            longestPath(player,new_board,path+1,longest)
        return longest

    if (game_phase<15): # only feasible to calculate late-game
        game_phase = abs(game_phase-max_phase) # low if early, high if late in game
        return (longestPath(player,game)-longestPath(game.get_opponent(player),game))
    else:
        opponent = game.get_opponent(player)
        return float(len(game.get_legal_moves(player)))-2.0*float(len(game.get_legal_moves(opponent)))


def custom_score_2(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    if game.is_loser(player):
        return float("-inf")
    if game.is_winner(player):
        return float("inf")

    # Aim to maximise your own available moves vs the opponent (Factor 2)

    opponent = game.get_opponent(player)
    return float(len(game.get_legal_moves(player)))-2.0*float(len(game.get_legal_moves(opponent)))


def custom_score_3(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    if game.is_loser(player):
        return float("-inf")
    if game.is_winner(player):
        return float("inf")
    game_phase = len(game.get_blank_spaces()) # high if early, low if late

    # Heuristic tries to take advantage of the center and shadowing if possible, otherwise stick to the centre and maximise number of moves 

    # (*0) Calculate the (theoretical) centre
    center = (game.width / 2., game.height / 2.)
    opponent = game.get_opponent(player)
    loc_player = game.get_player_location(player)
    loc_opponent = game.get_player_location(opponent)
    if game.width % 2 != 0 and game.height % 2 != 0:
        trueCentre = True
        loc_mirror = tuple(abs(x-(game.width-1)) for x in loc_player) # the mirrored location of the player across the axes
    else:
        trueCentre = False
    # (1) Always take the centre!
    if loc_player == center:
        return float("inf")
    # (2) If opponent has the centre, avoid a position within knight's movement at all costs to avoid shadowing
    if loc_opponent == center:
        r, c = center
        directions = [(-2, -1), (-2, 1), (-1, -2), (-1, 2),(1, -2), (1, 2), (2, -1), (2, 1)]
        avoidable_positions = [(r + dr, c + dc) for dr, dc in directions]
        if loc_player in avoidable_positions:
            return float("-inf")
    # (3) If we can shadow the opponent, we should!
    if trueCentre:
        if center not in game.get_blank_spaces() and loc_opponent == loc_mirror and len(game.get_legal_moves(player)) == len(game.get_legal_moves(opponent)):
            return float("inf")
    # (4) Finally, we simply return number of moves active player can make minus number of moves opponent can make minus the distance from the centre, weighted by the game phase
    w, h = center
    y, x = loc_player
    dist = float((h - y)**2 + (w - x)**2)
    return (float(len(game.get_legal_moves(player)))-2.0*float(len(game.get_legal_moves(opponent)))-dist)*game_phase
    
        


class IsolationPlayer:
    """Base class for minimax and alphabeta agents -- this class is never
    constructed or tested directly.

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """
    def __init__(self, search_depth=3, score_fn=custom_score, timeout=20.):
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout


class MinimaxPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using depth-limited minimax
    search
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        For fixed-depth search, this function simply wraps the call to the
        minimax method, but this method provides a common interface for all
        Isolation agents, and you will replace it in the AlphaBetaPlayer with
        iterative deepening search.

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        if not game.get_legal_moves():
            return (-1,-1)
        best_move = game.get_legal_moves()[0]

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            return self.minimax(game, self.search_depth)

        except SearchTimeout:
            pass  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        return best_move

    def minimax(self, game, depth):
        """Implement depth-limited minimax search algorithm

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves
        """

        def min_value(game, traversed_depth=1):
            if self.time_left() < self.TIMER_THRESHOLD:
                raise SearchTimeout()
            if traversed_depth==depth:
                return self.score(game,self)
            traversed_depth+=1
            v = float("inf")
            for m in game.get_legal_moves():
                if self.time_left() < self.TIMER_THRESHOLD:
                    raise SearchTimeout()
                v = min(v, max_value(game.forecast_move(m),traversed_depth))
            return v

        def max_value(game, traversed_depth=1):
            if self.time_left() < self.TIMER_THRESHOLD:
                raise SearchTimeout()
            if traversed_depth==depth:
                return self.score(game,self)
            traversed_depth+=1
            v = float("-inf")
            for m in game.get_legal_moves():
                if self.time_left() < self.TIMER_THRESHOLD:
                    raise SearchTimeout()
                v = max(v, min_value(game.forecast_move(m),traversed_depth))
            return v
            
        return max(game.get_legal_moves(), key=lambda m: min_value(game.forecast_move(m)))


class AlphaBetaPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using iterative deepening minimax
    search with alpha-beta pruning
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        **********************************************************************
        NOTE: If time_left() < 0 when this function returns, the agent will
              forfeit the game due to timeout
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        if not game.get_legal_moves():
            return (-1,-1)
        best_move = game.get_legal_moves()[0]
        search_depth = 1
	
        try:
            while(self.time_left()>self.TIMER_THRESHOLD):
                last_move = best_move
                best_move = self.alphabeta(game,search_depth)
                search_depth += 1

        except SearchTimeout:
            best_move = last_move
            return best_move
	
        # Return the best move from the last completed search iteration
        return best_move

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf")):
        """Depth-limited minimax search with alpha-beta pruning

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves
        """

        def min_value(game, depth, alpha=float("-inf"), beta=float("inf")):
            if self.time_left() < self.TIMER_THRESHOLD:
                raise SearchTimeout()
            if depth == 0 or not game.get_legal_moves():
                return self.score(game,self)
            v = float("inf")
            for m in game.get_legal_moves():
                v = min(v, max_value(game.forecast_move(m), depth - 1, alpha, beta))
                if v<=alpha:
                    return v
                beta = min(beta,v)
            return v

        def max_value(game, depth, alpha=float("-inf"), beta=float("inf")):
            if self.time_left() < self.TIMER_THRESHOLD:
                raise SearchTimeout()
            if depth == 0 or not game.get_legal_moves():
                return self.score(game,self)
            v = float("-inf")
            for m in game.get_legal_moves():
                v = max(v, min_value(game.forecast_move(m), depth - 1, alpha, beta))
                if v>=beta:
                    return v
                alpha = max(alpha,v)
            return v
            
        best_val = float("-inf")
        best_move = (-1,-1)
        for move in game.get_legal_moves():
            v = min_value(game.forecast_move(move), depth - 1, alpha, beta)
            if v>best_val:
                best_val = v
                best_move = move
            alpha=max(alpha, best_val)
        return best_move

