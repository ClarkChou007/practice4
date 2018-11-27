# ghostAgents.py
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


from game import Agent
from game import Actions
from game import Directions
import random
from util import manhattanDistance
import util

class GhostAgent( Agent ):
    def __init__( self, index ):
        self.index = index

    def getAction( self, state ):
        dist = self.getDistribution(state)
        if len(dist) == 0:
            return Directions.STOP
        else:
            return util.chooseFromDistribution( dist )

    def getDistribution(self, state):
        "Returns a Counter encoding a distribution over actions from the provided state."
        util.   raiseNotDefined()
    def evaluationFunction(self, currentGameState, action):

        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        foodDist = float('inf')
        for food in newFood.asList():
            dist = manhattanDistance(food, newPos)
            if (dist < foodDist):
                foodDist = dist

        ghostScore = 0
        for state in newGhostStates:
            dist = manhattanDistance(newPos, state.getPosition())
            if dist == 0 and state.scaredTimer > 0:
                ghostScore += 100000000000
            elif state.scaredTimer > 0 and dist < state.scaredTimer:
                ghostScore += (1 / (1 + dist))
            elif dist < 3:
                ghostScore -= dist / 100

        score = ghostScore

        return score


class RandomGhost( GhostAgent ):
    "A ghost that chooses a legal action uniformly at random."
    def getDistribution( self, state ):
        dist = util.Counter()
        for a in state.getLegalActions( self.index ): dist[a] = 1.0
        dist.normalize()
        return dist

class DirectionalGhost( GhostAgent ):
    "A ghost that prefers to rush Pacman, or flee when scared."
    def __init__( self, index, prob_attack=0.8, prob_scaredFlee=0.8 ):
        self.index = index
        self.prob_attack = prob_attack
        self.prob_scaredFlee = prob_scaredFlee

    def getDistribution( self, state ):
        # Read variables from state
        ghostState = state.getGhostState( self.index )
        legalActions = state.getLegalActions( self.index )
        pos = state.getGhostPosition( self.index )
        isScared = ghostState.scaredTimer > 0

        speed = 1
        if isScared: speed = 0.5

        actionVectors = [Actions.directionToVector( a, speed ) for a in legalActions]
        newPositions = [( pos[0]+a[0], pos[1]+a[1] ) for a in actionVectors]
        pacmanPosition = state.getPacmanPosition()

        # Select best actions given the state
        distancesToPacman = [manhattanDistance( pos, pacmanPosition ) for pos in newPositions]
        if isScared:
            bestScore = max( distancesToPacman )
            bestProb = self.prob_scaredFlee
        else:
            bestScore = min( distancesToPacman )
            bestProb = self.prob_attack
        bestActions = [action for action, distance in zip( legalActions, distancesToPacman ) if distance == bestScore]

        # Construct distribution
        dist = util.Counter()
        for a in bestActions : dist[a] = bestProb / len(bestActions)
        for a in legalActions: dist[a] += ( 1-bestProb ) / len(legalActions)
        dist.normalize()
        return dist

class MinimaxGhost(GhostAgent):

    """
      Your minimax agent (question 1)

      useage: python2 pacman.py -p ExpectimaxAgent -l specialNew -g MinimaxGhost -a depth=4
              python2 pacman.py -l specialNew -g MinimaxGhost

    """
    "*** YOUR CODE HERE ***"
    def __init__(self,index , evalFun= 'scoreEvaluationFunctionGhost',depth = 2):
        self.index = index
        self.evaluationFunction = util.lookup(evalFun,globals())
        self.depth = int(depth)

    def getAction( self, state ):
        startdepth = 1
        index = self.index  # ghost
        Actions = self.getLegalActions(self.index)
        Move = [action for action in Actions if action is not Directions.STOP]
        evaluation = [self.maxMethod(state, startdepth, index, index) for action in Actions if
                      action is not Directions.STOP]
        score = min(evaluation)
        bestIndex = [index for index in range(len(evaluation)) if evaluation[index] == score]
        return Move[random.choice(bestIndex)]

    def maxMethod(self, state, depth, agent, ghost):
        if depth >= self.depth and depth != 1:
            eval = self.evaluationFunction(state)
            return eval
        else:
            v = -100000000000
            if len(state.getLegalActions(agent)) > 0:
                for action in state.getLegalActions(agent):
                    v = max(v, self.minMethod(state.generateSuccessor(agent, action), depth, ghost, 0))
            return v

    def minMethod(self, state, depth, agent, ghost):
        if depth >= self.depth and depth != 1:
            eval = self.evaluationFunction(state)
            return eval
        else:
            if len(state.getLegalActions(agent)) > 0:
                v = 100000000000
            else:
                v = self.evaluationFunction(state)
            for action in state.getLegalActions(agent):
                if agent == state.getNumAgent() - 1:
                    v = max(v, self.maxMethod(state.generateSuccessor(agent, action), depth + 1, ghost, ghost))
                else:
                    v = min(v, self.maxMethod(state.generateSuccessor(agent, action), depth, ghost, ghost))
            return v


def betterEvaluationFunctionGhost(currentGameState):
    """
        Ghost evaluation function
    """

    position = list(currentGameState.getPacmanPosition())
    foodPos = currentGameState.getFood().asList()
    foodList = []

    for food in foodPos:
        pacmanDist = manhattanDistance(position, food)
        foodList.append(pacmanDist)

    if not foodList:
        foodList.append(0)

    nearestPelletDist = min(foodList)
    return currentGameState.getScore() + (-1) * nearestPelletDist
def scoreEvaluationFunctionGhost (currentGameState):
    return currentGameState.getScore()

# Abbreviation
ghostEval = betterEvaluationFunctionGhost

