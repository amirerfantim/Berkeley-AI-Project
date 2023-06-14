# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        foodDistances = [util.manhattanDistance(newPos, foodPos) for foodPos in newFood.asList()]

        minFoodDistance = min(foodDistances) if foodDistances else 0

        ghostDistances = [util.manhattanDistance(newPos, ghostState.getPosition()) for ghostState in newGhostStates]
        minGhostDistance = min(ghostDistances)

        if minFoodDistance:
            minFoodDistance = 1 / minFoodDistance
        if minGhostDistance:
            minGhostDistance = 1.85 / minGhostDistance

        avg_scared = sum(newScaredTimes) / len(newScaredTimes)

        for i in range(len(newGhostStates)):
            ghost = newGhostStates[i]
            if not newScaredTimes[i] and manhattanDistance(ghost.getPosition(), newPos) <= 1:
                return float('-inf')

        return successorGameState.getScore() + minFoodDistance - minGhostDistance + 5 * avg_scared


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):

        def minimax_search(state, depth, agent_index):

            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)

            if agent_index == 0:
                return max_value(state, depth, agent_index)
            else:
                return min_value(state, depth, agent_index)

        def max_value(state, depth, agent_index):
            best_value = -float('inf')
            for action in state.getLegalActions(agent_index):
                successor_state = state.generateSuccessor(agent_index, action)
                value = minimax_search(successor_state, depth, 1)
                best_value = max(best_value, value)
            return best_value

        def min_value(state, depth, agent_index):
            best_value = float('inf')
            for action in state.getLegalActions(agent_index):
                successor_state = state.generateSuccessor(agent_index, action)
                if agent_index == state.getNumAgents() - 1:
                    value = minimax_search(successor_state, depth + 1, 0)
                else:
                    value = minimax_search(successor_state, depth, agent_index + 1)
                best_value = min(best_value, value)
            return best_value

        best_action = None
        best_value = -float('inf')
        for action in gameState.getLegalActions(0):  # Pacman is the max player
            successor_state = gameState.generateSuccessor(0, action)
            value = minimax_search(successor_state, 0, 1)  # Start with depth 0
            if value > best_value:
                best_action = action
                best_value = value

        return best_action


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        alpha = -float('inf')
        beta = float('inf')
        best_action = None
        best_value = -float('inf')

        def alpha_beta_search(state, depth, agent_index, alpha, beta):

            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)

            if agent_index == 0:
                return max_value(state, depth, agent_index, alpha, beta)
            else:
                return min_value(state, depth, agent_index, alpha, beta)

        def max_value(state, depth, agent_index, alpha, beta):
            best_value = -float('inf')
            for action in state.getLegalActions(agent_index):
                successor_state = state.generateSuccessor(agent_index, action)
                value = alpha_beta_search(successor_state, depth, 1, alpha, beta)
                if value > best_value:
                    best_value = value
                if best_value > beta:
                    return best_value
                if best_value > alpha:
                    alpha = best_value
            return best_value

        def min_value(state, depth, agent_index, alpha, beta):
            best_value = float('inf')
            for action in state.getLegalActions(agent_index):
                successor_state = state.generateSuccessor(agent_index, action)
                if agent_index == state.getNumAgents() - 1:
                    value = alpha_beta_search(successor_state, depth + 1, 0, alpha, beta)
                else:
                    value = alpha_beta_search(successor_state, depth, agent_index + 1, alpha, beta)
                if best_value > value:
                    best_value = value
                if best_value < alpha:
                    return best_value
                if best_value < beta:
                    beta = best_value
            return best_value

        for action in gameState.getLegalActions(0):
            successor_state = gameState.generateSuccessor(0, action)
            value = alpha_beta_search(successor_state, 0, 1, alpha, beta)
            if value > best_value:
                best_action = action
                best_value = value
            if best_value > alpha:
                alpha = best_value
        return best_action


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):

        return self.maximizer(gameState, 0, 0)[1]

    def maximizer(self, gameState, depth, agentIndex):

        if depth == self.depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState), ''

        best_val, best_action = -float('inf'), None
        for action in gameState.getLegalActions(agentIndex):
            state = gameState.generateSuccessor(agentIndex, action)
            value = self.randomizer(state, depth + 1, 1)
            if value > best_val:
                best_action = action
            best_val = max(best_val, value)

        return best_val, best_action

    def randomizer(self, gameState, depth, index):

        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)

        sum = 0
        for action in gameState.getLegalActions(index):
            state = gameState.generateSuccessor(index, action)
            if index != gameState.getNumAgents() - 1:
                value = self.randomizer(state, depth, index + 1)
            else:
                value = self.maximizer(state, depth, 0)[0]
            sum += value / len(gameState.getLegalActions(index))

        return sum

    def is_terminal_state(self, gameState, depth):

        return depth == self.depth or gameState.isWin() or gameState.isLose()

    def get_legal_actions(self, gameState, agent_index):

        return gameState.getLegalActions(agent_index)

    def generate_successor(self, gameState, agent_index, action):
        return gameState.generateSuccessor(agent_index, action)


def betterEvaluationFunction(currentGameState):
    """
     Your evaluation function should return a score to maximize for the given
     `currentGameState`. You can use any information available to you in the
     currentGameState, including the food, capsules, scared ghosts, Pacman's
     position, and ghosts' positions.
     """
    pacmanPos = currentGameState.getPacmanPosition()
    food = currentGameState.getFood()
    capsules = currentGameState.getCapsules()
    ghosts = currentGameState.getGhostStates()
    ghostStates = currentGameState.getGhostStates()
    scaredTimers = [ghostState.scaredTimer for ghostState in ghostStates]


    foodDistances = [manhattanDistance(pacmanPos, foodPos) for foodPos in food.asList()]
    minFoodDist = min(foodDistances) if foodDistances else 0

    capsuleDistances = [manhattanDistance(pacmanPos, capPos) for capPos in capsules]
    minCapDist = min(capsuleDistances) if capsuleDistances else 0

    ghostDistances = [manhattanDistance(pacmanPos, ghost.getPosition()) for ghost in ghosts]
    minGhostDist = min(ghostDistances) if ghostDistances else 0
    isGhostScared = any(ghost.scaredTimer > 0 for ghost in ghosts)

    score = currentGameState.getScore()
    capsules = currentGameState.getCapsules()
    capsule_count = len(capsules)
    all_times = 0
    for x in scaredTimers:
        all_times += x

    if minGhostDist <= 1 and not isGhostScared:
        score -= 1000
    else:
        score += 15 / (minFoodDist + 1)
        score += 10 / (minCapDist + 1)
        if isGhostScared:
            score += 100 / (minGhostDist + 1)
        score += 22 / (capsule_count + 1)
        score += 2 * all_times
        score -= len(food.asList())
    return score


# Abbreviation
better = betterEvaluationFunction
