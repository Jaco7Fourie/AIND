"""This file is provided as a starting template for writing your own unit
tests to run and debug your minimax and alphabeta agents locally.  The test
cases used by the project assistant are not public.
"""

import unittest

import isolation
import game_agent
import sample_players

from importlib import reload


@unittest.skip("skip basic test for now")
class IsolationTest(unittest.TestCase):
    """Unit tests for isolation agents"""

    def setUp(self):
        reload(game_agent)
        self.player1 = sample_players.GreedyPlayer()
        self.player2 = game_agent.AlphaBetaPlayer(score_fn=sample_players.open_move_score, timeout=20)
        self.game = isolation.Board(self.player1, self.player2)

    def test_minimax(self):
        """play a game to completion"""
        winner, history, outcome = self.game.play(time_limit=150)
        print("\nWinner: {}\nOutcome: {}".format(winner, outcome))
        print(self.game.to_string())
        print("Move history:\n{!s}".format(history))
        assert winner == self.player2


@unittest.skip("skip heuristic 1 for now")
class HeuristicTest1(unittest.TestCase):
    """Unit tests for isolation agents"""

    def setUp(self):
        reload(game_agent)
        self.player1 = game_agent.AlphaBetaPlayer(score_fn=sample_players.open_move_score, timeout=20)
        self.player2 = game_agent.AlphaBetaPlayer(score_fn=sample_players.improved_score, timeout=20)
        self.game = isolation.Board(self.player1, self.player2)

    def test_minimax(self):
        """play a game to completion"""
        winner, history, outcome = self.game.play(time_limit=150)
        print("\nWinner: {}\nOutcome: {}".format(winner, outcome))
        print(self.game.to_string())
        print("Move history:\n{!s}".format(history))
        assert winner == self.player2


class HeuristicTest2(unittest.TestCase):
    """Unit tests for isolation agents"""

    def setUp(self):
        reload(game_agent)
        self.player1 = game_agent.AlphaBetaPlayer(score_fn=sample_players.improved_score, timeout=20)
        self.player2 = game_agent.AlphaBetaPlayer(score_fn=game_agent.custom_score_3, timeout=20)
        self.game = isolation.Board(self.player1, self.player2)

    def test_minimax(self):
        """play a game to completion"""
        winner, history, outcome = self.game.play(time_limit=150)
        print("\nWinner: {}\nOutcome: {}".format(winner, outcome))
        print(self.game.to_string())
        print("Move history:\n{!s}".format(history))
        assert winner == self.player2


if __name__ == '__main__':
    unittest.main()
