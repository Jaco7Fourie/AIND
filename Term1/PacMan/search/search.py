# search.py
# ---------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

"""
In search.py, you will implement generic search algorithms which are called 
by Pacman agents (in searchAgents.py).
"""

import util


class SearchProblem:
    """
  This class outlines the structure of a search problem, but doesn't implement
  any of the methods (in object-oriented terminology: an abstract class).
  
  You do not need to change anything in this class, ever.
  """

    def getStartState(self):
        """
     Returns the start state for the search problem 
     """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
       state: Search state
    
     Returns True if and only if the state is a valid goal state
     """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
       state: Search state
     
     For a given state, this should return a list of triples, 
     (successor, action, stepCost), where 'successor' is a 
     successor to the current state, 'action' is the action
     required to get there, and 'stepCost' is the incremental 
     cost of expanding to that successor
     """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
      actions: A list of actions to take
 
     This method returns the total cost of a particular sequence of actions.  The sequence must
     be composed of legal moves
     """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
  Returns a sequence of moves that solves tinyMaze.  For any other
  maze, the sequence of moves will be incorrect, so only use this for tinyMaze
  """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]


def depthFirstSearch(problem):
    """
  Search the deepest nodes in the search tree first
  [2nd Edition: p 75, 3rd Edition: p 87]
  
  Your search algorithm needs to return a list of actions that reaches
  the goal.  Make sure to implement a graph search algorithm 
  [2nd Edition: Fig. 3.18, 3rd Edition: Fig 3.7].
  
  To get started, you might want to try some of these simple commands to
  understand the search problem that is being passed in:
  
  print "Start:", problem.getStartState()
  print "Is the start a goal?", problem.isGoalState(problem.getStartState())
  print "Start's successors:", problem.getSuccessors(problem.getStartState())
  """

    state_stack = util.Stack()
    path_stack = util.Stack()
    state_stack.push((problem.getStartState(), None, 0))
    path_stack.push([])
    while not state_stack.isEmpty():
        vertex = state_stack.pop()
        path = path_stack.pop()
        next_steps = problem.getSuccessors(vertex[0])
        for v in next_steps:
            if v[0] not in set([p[0] for p in path]):
                if problem.isGoalState(v[0]):
                    return [p[1] for p in path] + [v[1]]
                else:
                    state_stack.push(v)
                    path_stack.push(path+[v])



def breadthFirstSearch(problem):
    """
  Search the shallowest nodes in the search tree first.
  [2nd Edition: p 73, 3rd Edition: p 82]
  """
    state_stack = util.Queue()
    path_stack = util.Queue()
    state_stack.push((problem.getStartState(), None, 0))
    path_stack.push([])
    while not state_stack.isEmpty():
        vertex = state_stack.pop()
        path = path_stack.pop()
        next_steps = problem.getSuccessors(vertex[0])
        for v in next_steps:
            if v[0] not in set([p[0] for p in path]):
                if problem.isGoalState(v[0]):
                    return [p[1] for p in path] + [v[1]]
                else:
                    state_stack.push(v)
                    path_stack.push(path+[v])


def uniformCostSearch(problem):
    "Search the node of least total cost first. "
    state_stack = util.PriorityQueue()
    path_stack = util.PriorityQueue()
    state_stack.push((problem.getStartState(), None, 0), 0)
    path_stack.push([], 0)
    while not state_stack.isEmpty():
        vertex = state_stack.pop()
        path = path_stack.pop()
        next_steps = problem.getSuccessors(vertex[0])
        for v in next_steps:
            if v[0] not in set([p[0] for p in path]):
                if problem.isGoalState(v[0]):
                    return [p[1] for p in path] + [v[1]]
                else:
                    cost = vertex[2] + v[2]
                    v_costed = (v[0], v[1], cost)
                    state_stack.push(v_costed, cost)
                    path_stack.push(path+[v_costed], cost)


def nullHeuristic(state, problem=None):
    """
  A heuristic function estimates the cost from the current state to the nearest
  goal in the provided SearchProblem.  This heuristic is trivial.
  """
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    "Search the node that has the lowest combined cost and heuristic first."
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
