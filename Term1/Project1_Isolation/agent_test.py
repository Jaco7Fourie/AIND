"""This file is provided as a starting template for writing your own unit
tests to run and debug your minimax and alphabeta agents locally.  The test
cases used by the project assistant are not public.
"""

import unittest

import isolation
import game_agent
import sample_players

from importlib import reload


class IsolationTest(unittest.TestCase):
    """Unit tests for isolation agents"""

    def setUp(self):
        reload(game_agent)
        self.player1 = sample_players.GreedyPlayer()
        self.player2 = game_agent.MinimaxPlayer(score_fn=sample_players.open_move_score)
        self.game = isolation.Board(self.player1, self.player2)

    def test_minimax(self):
        """play a game to completion"""
        winner, history, outcome = self.game.play(time_limit=150)
        print("\nWinner: {}\nOutcome: {}".format(winner, outcome))
        print(self.game.to_string())
        print("Move history:\n{!s}".format(history))


if __name__ == '__main__':
    unittest.main()
