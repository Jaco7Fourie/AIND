"""Finish all TODO items in this file to complete the isolation project, then
test your agent's strength against a set of known agents using tournament.py
and include the results in your report.
"""
import random


class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass


def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    This heuristic is based on a more aggressive version improved_score.
    It ads a larger weight to the opponents available moves. Additionally is also adds a penalty that increases
    as the player moves further away from the centre.

    This should be the best heuristic function for your project submission.

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
    PENALTY_SCALE = 0.5
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    opponent = game.get_opponent(player)
    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(opponent))
    pw, ph = game.get_player_location(player)
    w, h = game.width / 2., game.height / 2.
    # use the Manhattan distance metric for performance reasons.
    distance = abs(w - pw) + abs(h - ph)
    return float(own_moves - 2 * opp_moves) - PENALTY_SCALE * distance




def custom_score_2(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    This is 'improved_score' but weighted to limit opponent's moves more and play more aggressively.
    It also uses a distance metric to balance the overly aggressive behaviour of the increased weight by
    penalising the play as it moves closer to the opponent

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

    opponent = game.get_opponent(player)
    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(opponent))
    current_location1 = game.get_player_location(player)
    current_location2 = game.get_player_location(opponent)
    # using the sum-of-absolute-difference (SAD) distance the longest distance two players can be is the sum of the
    # width and height of the board. Scale our penalty so that it does not dominate the main heuristic
    SCALING_FACTOR = 4 # 2
    # the sum-of-absolute-difference (SAD) distance between the players
    distance = abs(current_location1[0] - current_location2[0]) + abs(current_location1[1] - current_location2[1])
    return float(own_moves - 2*opp_moves) + distance*SCALING_FACTOR


def custom_score_3(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Having noticed that the best heuristic 'improved_score' often fails against the 'open_score' heuristic by
    ending up in a corner this heuristic tries to improve on 'improved_score' by adding a penalty to corners

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
    PENALTY_AMOUNT = 3
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
    current_location = game.get_player_location(player)
    corner_penalty = PENALTY_AMOUNT if (current_location[0] == 0 and current_location[1] == 0) or \
                                       (
                                       current_location[0] == game.height - 1 and current_location[1] == game.width - 1) \
        else 0
    return float(own_moves - 2 * opp_moves) - corner_penalty


class IsolationPlayer:
    """Base class for minimax and alphabeta agents -- this class is never
    constructed or tested directly.

    ********************  DO NOT MODIFY THIS CLASS  ********************

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
    def __init__(self, search_depth=3, score_fn=custom_score, timeout=15.):
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout


class MinimaxPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using depth-limited minimax
    search. You must finish and test this player to make sure it properly uses
    minimax to return a good move before the search time limit expires.
    """

    def __str__(self):
        return "MinimaxPlayer: {}".format(self.score.__name__)

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        **************  YOU DO NOT NEED TO MODIFY THIS FUNCTION  *************

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
        best_move = (-1, -1)

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            return self.minimax(game, self.search_depth)

        except SearchTimeout:
            pass  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        return best_move

    def max_value(self, node, depth):
        """
        Determine max value of this node by recursively going through the whole tree
        :param node: the node to search from
        :param depth: the maximum depth to search to from this node
        :return: the maximum value that can be obtained from this node as measured by the evaluation function (self.score)
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        # check to see if we are done
        end_val = node.utility(self)
        if end_val != 0:
            return end_val
        if depth <= 0:
            return self.score(node, self)
        # get all possible next moves
        next_moves = node.get_legal_moves()
        max_val = float("-inf")
        for next_move in next_moves:
            next_node = node.forecast_move(next_move)
            val = self.min_value(next_node, depth-1)
            if val > max_val:
                max_val = val
        return max_val

    def min_value(self, node, depth):
        """
        Determine min value of this node by recursively going through the whole tree
        :param node: the node to search from
        :param depth: the minimum depth to search to from this node
        :return: the minimum value that can be obtained from this node as measured by the evaluation function (self.score)
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        # check to see if we are done
        end_val = node.utility(self)
        if end_val != 0:
            return end_val
        if depth <= 0:
            return self.score(node, self)
        # get all possible next moves
        next_moves = node.get_legal_moves()
        min_val = float("inf")
        for next_move in next_moves:
            next_node = node.forecast_move(next_move)
            val = self.max_value(next_node, depth - 1)
            if val < min_val:
                min_val = val
        return min_val

    def minimax(self, game, depth):
        """Implement depth-limited minimax search algorithm as described in
        the lectures.

        This should be a modified version of MINIMAX-DECISION in the AIMA text.
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Minimax-Decision.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

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

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        # check to see if we are done
        legal_moves = game.get_legal_moves()
        if not legal_moves:
            return -1, -1
        max_val = float("-inf")
        best_move = legal_moves[0] # just to initialise to valid move
        for next_move in legal_moves:
            next_node = game.forecast_move(next_move)
            val = self.min_value(next_node, depth - 1)
            if val > max_val:
                max_val = val
                best_move = next_move
        return best_move


class AlphaBetaPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using iterative deepening minimax
    search with alpha-beta pruning. You must finish and test this player to
    make sure it returns a good move before the search time limit expires.
    """

    def __str__(self):
        return "AlphaBetaPlayer: {}".format(self.score.__name__)

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        Modify the get_move() method from the MinimaxPlayer class to implement
        iterative deepening search instead of fixed-depth search.

        **********************************************************************
        NOTE: If time_left() < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
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
        best_move = (-1, -1)
        # print(game.to_string())
        i = 1
        while True:
            try:
                # The try/except block will automatically catch the exception
                # raised when the timer is about to expire.
                best_move = self.alphabeta(game, i)
                # print("Best move ={} at depth={}".format(best_move, i))
                i += 1

            except SearchTimeout:
                # print("Timeout! Best move ={} at depth={}".format(best_move, i-1))
                return best_move # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        # return best_move

    def min_max_value(self, node, depth, alpha, beta, maximising=True):
        """
        Determine min or max value of this node by recursively going through the whole tree
        also perform the alpha-beta pruning step to minimise search time
        :param beta: Beta limits the upper bound of search on maximizing layers
        :param alpha: Alpha limits the lower bound of search on minimizing layers
        :param maximising: True if this is a maximising node, False if it should minimise
        :param node: the node to search from
        :param depth: the minimum depth to search to from this node
        :return: the minimum value that can be obtained from this node as measured by the evaluation function (self.score)
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        # debug
        #print("depth={:4<d} alpha={:4f} beta={:4f}".format(depth, alpha, beta))

        # check to see if we are done
        # get all possible next moves
        next_moves = node.get_legal_moves()
        if depth == 0 or not node.get_legal_moves():
            return self.score(node, self)

        term_val = float("-inf") if maximising else float("inf")
        for next_move in next_moves:
            next_node = node.forecast_move(next_move)
            val = self.min_max_value(next_node, depth - 1, alpha, beta, False) if maximising \
                else self.min_max_value(next_node, depth - 1, alpha, beta, True)
            if (maximising and val > term_val) or (not maximising and val < term_val):
                term_val = val
            if maximising:
                alpha = max(alpha, term_val)
                if term_val >= beta:
                    # debug
                    # print("pruned v > beta --> depth={:<4d} alpha={:4f} beta={:4f}".format(depth, alpha, beta))
                    return term_val
            else:
                beta = min(beta, term_val)
                if term_val <= alpha:
                    # debug
                    # print("pruned v < alpha --> depth={:<4d} alpha={:4f} beta={:4f}".format(depth, alpha, beta))
                    return term_val
        return term_val

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf")):
        """Implement depth-limited minimax search with alpha-beta pruning as
        described in the lectures.

        This should be a modified version of ALPHA-BETA-SEARCH in the AIMA text
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Alpha-Beta-Search.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

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

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        # check to see if we are done
        if game.utility(self) != 0:
            return -1, -1
        # get all possible next moves
        next_moves = game.get_legal_moves()
        best_move = next_moves[0]  # just to initialise to valid move
        max_val = float("-inf")
        for next_move in next_moves:
            next_node = game.forecast_move(next_move)
            val = self.min_max_value(next_node, depth - 1, alpha, beta, False)
            if val > max_val:
                max_val = val
                best_move = next_move
            alpha = max(alpha, max_val)
            if max_val >= beta:  # this only happens when a sure win situation was found (beta is always inf)
                return best_move
        # print("best score={}".format(max_val))
        return best_move

