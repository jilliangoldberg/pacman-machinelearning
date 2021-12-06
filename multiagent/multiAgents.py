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
from game import AgentState, Directions
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
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        score = 0
        goAfterGhost = any([times > 0 for times in newScaredTimes])
        if goAfterGhost:
            score += 200

        foodScores = []
        for food in newFood.asList():
            if food:
                foodScores.append(1/(manhattanDistance(newPos, food)))

        return successorGameState.getScore() + max(foodScores,default=0) + score

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

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        # above code from the simple one for reference
        ## ATTEMPT 3 ##
        # pacman is the maximizer, ghosts are the minimizers
        # call get max with numAgents
        # in get max, call the get min function (all the successors are the ghosts)
        # checker to see if index is zero for agent, then see when it hits num of agents
            #  then call min again to check the ghost again

        # plan
        # def main():
            # call maxValue()
        # def max():
            # call minGhost(ghost) to get min value
        # def min():
            # if the ghost == numAgents, call max on the next state and depth+=1
            # if not, call min on ghost+=1

        if gameState.isWin() or gameState.isLose():
            return Directions.STOP

        pacMoves = gameState.getLegalActions(0)
        numAgents = gameState.getNumAgents() - 1
        evalFunc = self.evaluationFunction

        def minimax(agent, state, depth):
            if depth == self.depth or state.isWin() or state.isLose():
                return evalFunc(state)
            nextActions = state.getLegalActions(agent)
            if len(nextActions) == 0:
                return evalFunc(state)
            nextStates = [state.generateSuccessor(agent, action) for action in nextActions]
            if agent == 0:
                results = [minimax(agent + 1, s, depth) for s in nextStates]
                return max(results)
            if agent == numAgents:
                results = [minimax(0, s, depth + 1) for s in nextStates]
            else:
                results = [minimax(agent + 1, s, depth) for s in nextStates]
            return min(results)

        bestScore = float('-inf')
        bestAction = Directions.STOP
        for action in pacMoves:
            state = gameState.generateSuccessor(0, action)
            score = minimax(1, state, 0)
            if score > bestScore:
                bestScore = score
                bestAction = action
        return bestAction
    
# def maxValuePacman(self, state, action, depth):
#     v = float('-inf')
#     evalFunc = self.evaluationFunction
#     if state.isWin() or state.isLose() or depth == self.depth:
#         return evalFunc(state)
#     # call eval function when you hit the max depth of the tree
#     nextPacMoves = state.getLegalActions(0) # gets the actions pac can take
#     if nextPacMoves == 0:
#         return evalFunc(state)
#     # generates the states if the ghost moved a certain way
#     nextPacStates = [state.generateSuccessor(0, action) for action in nextPacMoves]
#     vals = [minValueGhost(self, s, action, 1, depth) for s in nextPacStates]
#     return max(vals)

# def minValueGhost(self, state, pacAction, ghostIndex, depth):
#     v = float('inf')
#     evalFunc = self.evaluationFunction
#     numAgents = state.getNumAgents() - 1
#     if state.isWin() or state.isLose():
#         return evalFunc(state)
#     # recurse until ghosts == numagents
#     nextGhostMoves = state.getLegalActions(ghostIndex) # gets the actions the ghost can take
#     if nextGhostMoves == 0:
#         return evalFunc(state)
#     # generates the states if the ghost moved a certain way
#     nextGhostStates = [state.generateSuccessor(ghostIndex, action) for action in nextGhostMoves]
#     if ghostIndex == numAgents:
#         if depth == self.depth:
#             return min([evalFunc(nextState) for nextState in nextGhostStates])
#         else:
#             return min([maxValuePacman(self, nextState, pacAction, depth + 1) for nextState in nextGhostStates])
#     # means that there are still other ghosts to evaluate!
#     return min([minValueGhost(self, nextState, pacAction, ghostIndex + 1, depth) for nextState in nextGhostStates])

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        if gameState.isWin() or gameState.isLose():
            return Directions.STOP

        pacMoves = gameState.getLegalActions(0)
        numAgents = gameState.getNumAgents() - 1
        evalFunc = self.evaluationFunction

        def alphaBeta(agent, state, depth, alpha, beta):
            nextActions = state.getLegalActions(agent)
            if depth == self.depth or state.isWin() or state.isLose() or len(nextActions) == 0:
                return evalFunc(state), Directions.STOP
            # pacman
            if agent == 0:
                bestV = float('-inf')
                bestAction = Directions.STOP
                for action in nextActions:
                    value, nextAction = alphaBeta(1, state.generateSuccessor(agent, action), depth, alpha, beta)
                    if value > bestV:
                        bestV = value
                        bestAction = action
                    alpha = max(alpha, bestV)
                    if bestV > beta:
                        return bestV, bestAction
                return bestV, bestAction
            # ghosts
            else:
                bestV = float('inf')
                bestAction = Directions.STOP
                for action in nextActions:
                    if agent == numAgents:
                        value, nextAction = alphaBeta(0, state.generateSuccessor(agent, action), depth + 1, alpha, beta)
                    else:
                        value, nextAction = alphaBeta(agent + 1, state.generateSuccessor(agent, action), depth, alpha, beta)
                    if value < bestV:
                        bestV = value
                        bestAction = action
                    beta = min(beta, bestV)
                    if bestV < alpha:
                        return bestV, bestAction
                return bestV, bestAction
        
        alpha, beta = float('-inf'), float('inf')
        bestValue, bestAction = alphaBeta(0, gameState, 0, alpha, beta)
        return bestAction

# def maxValue(agent, state, depth, alpha, beta):
#             nextActions = state.getLegalActions(agent)
#             if depth == self.depth or state.isWin() or state.isLose() or len(nextActions) == 0:
#                 return evalFunc(state), Directions.STOP
#             bestV = float('-inf')
#             bestAction = Directions.STOP
#             for action in nextActions:
#                 state = state.generateSuccessor(agent, action)
#                 value, action = minValue(1, state, depth, alpha, beta)
#                 if value > bestV:
#                     bestV = bestV
#                     bestAction = action
#                 if bestV > beta:
#                     return bestV, bestAction
#                 alpha = max(alpha, bestV)
#             return bestV, bestAction

#         def minValue(agent, state, depth, alpha, beta):
#             nextActions = state.getLegalActions(agent)
#             if depth == self.depth or state.isWin() or state.isLose() or len(nextActions) == 0:
#                 return evalFunc(state), Directions.STOP
#             bestV = float('inf')
#             bestAction = Directions.STOP
#             for action in nextActions:
#                 state = state.generateSuccessor(agent, action)
#                 if agent == numAgents:
#                     value, action = alphaBeta(0, state, depth + 1, alpha, beta)
#                     bestV = min(bestV, value)
#                 else:
#                     value, action = alphaBeta(agent + 1, state, depth, alpha, beta)
#                     bestV = min(bestV, value)
#                 if value > bestV:
#                     bestV = bestV
#                     bestAction = action
#                 if bestV < alpha:
#                     return bestV, bestAction
#                 beta = min(beta, bestV)
#             return bestV, bestAction

# def alpha_beta(agent, state, depth, alpha, beta):
#     if depth == self.depth or state.isWin() or state.isLose():
#         return evalFunc(state)
#     nextActions = state.getLegalActions(agent)
#     if len(nextActions) == 0:
#         return evalFunc(state)
#     nextStates = [state.generateSuccessor(agent, action) for action in nextActions]
#     if agent == 0:
#         # results = [alpha_beta(agent + 1, s, depth) for s in nextStates if alpha_beta(agent + 1, s, depth) < alpha]
#         value = float('-inf')
#         for s in nextStates:
#             value = max(value, alpha_beta(agent + 1, s, depth, alpha, beta))
#             if value > beta:
#                 return value
#             alpha = max(alpha, value)
#         return value
#     value = float('inf')
#     if agent == numAgents:
#         for s in nextStates:
#             value = min(value, alpha_beta(0, s, depth + 1, alpha, beta))
#             if value < alpha:
#                 return value
#             beta = min(beta, value)
#         return value
#     for s in nextStates:
#         value = min(value, alpha_beta(agent + 1, s, depth, alpha, beta))
#         if value < alpha:
#             return value
#         beta = min(beta, value)
#     return value

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        if gameState.isWin() or gameState.isLose():
            return Directions.STOP

        pacMoves = gameState.getLegalActions(0)
        numAgents = gameState.getNumAgents() - 1
        evalFunc = self.evaluationFunction

        def expectimax(agent, state, depth):
            if depth == self.depth or state.isWin() or state.isLose():
                return evalFunc(state)
            nextActions = state.getLegalActions(agent)
            if len(nextActions) == 0:
                return evalFunc(state)
            nextStates = [state.generateSuccessor(agent, action) for action in nextActions]
            # ghost! max it!
            if agent == 0:
                results = [expectimax(agent + 1, s, depth) for s in nextStates]
                return max(results)
            if agent == numAgents:
                results = [expectimax(0, s, depth + 1) for s in nextStates]
            else:
                results = [expectimax(agent + 1, s, depth) for s in nextStates]
            return sum(results) / len(results)

        bestScore = float('-inf')
        bestAction = Directions.STOP
        for action in pacMoves:
            state = gameState.generateSuccessor(0, action)
            score = expectimax(1, state, 0)
            if score > bestScore:
                bestScore = score
                bestAction = action
        return bestAction

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    pos = currentGameState.getPacmanPosition()
    foodList = currentGameState.getFood().asList()
    numAgents = currentGameState.getNumAgents()
    ghostStates = currentGameState.getGhostStates()
    scaredGhostTimes = [ghostState.scaredTimer for ghostState in ghostStates]

    STATE_SCORE = 0

    # distance from ghosts
    if numAgents > 1:
        clostestGhostDistance = min([manhattanDistance(pos, ghost.getPosition()) for ghost in ghostStates])
        if clostestGhostDistance <= 1:
            return float('-inf')
        STATE_SCORE -= 1/clostestGhostDistance
    
    # scared ghosts
    goAfterGhost = any([times > 0 for times in scaredGhostTimes])
    if goAfterGhost:
        STATE_SCORE += 200
    
    # distance from food
    curr = pos
    for food in foodList:
        closestFood = min(foodList, key= lambda x: manhattanDistance(x, curr))
        STATE_SCORE += 1/manhattanDistance(curr, closestFood)
        curr = closestFood
        foodList.remove(closestFood)
    
    # score
    STATE_SCORE += (8 * currentGameState.getScore())

    # extra food scores!
    foodScores = []
    for food in foodList:
        if food:
            foodScores.append(1/(manhattanDistance(pos, food)))
    STATE_SCORE += max(foodScores,default=0)

    return STATE_SCORE

# Abbreviation
better = betterEvaluationFunction
