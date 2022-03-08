# ==============================================
# 	LIBRARIES
# ==============================================

from operator import ne
import torch
import torch.nn.functional as F
from torch import nn, optim

import numpy as np
import math
import random
from random import choices

from time import time

import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

from copy import copy, deepcopy



# ==============================================
# 	GAME MODULE
# ==============================================


# ==============================================
# 	Game array description:
# ==============================================

# [...]


# ==============================================
# 	The end of the game
# ==============================================

# Return True if the current state is an ending state, and False otherwise
def isOver(state):
	return True

# Return the outcome of the game	
def gameScore(state):
	# Initialize result
	res = 0

	# Evaluate the result
	# [...]

	# Return the result
	return res


# ==============================================
#	Rotations of the board
# ==============================================

# Teaching the network all symmetric positions may be beneficial
def allSymmetries(state):
	# Initialize result
	res = []
	
	# Evaluate all symmetries of the current position
	# [...]

	# Return the result
	return res


# ==============================================
#	Basic state
# ==============================================

def startState():
	# Return the starting state, according to the chosen gamestate description model
	return []

# Return a random valid state of the game, useful for analyzing the performance of the agent
def randomState():
	# Return a random state
	return []


# ==============================================
#	Possible moves
# ==============================================


# Given the current position of the game, player whose turn it is and a valid move
# return the corresponding next position
def move(state, player, _move):
	
	# Evaluate a tuple (nextState, nextPlayer) where:
	# - nextState:  represents the position after 'player' perfomrms '_move' on 'state'
	# - nextPlayer: represents the player whose turn it will be after the move

	return (state, player) 


def qmove(state, player, _move):

	return (state, player, 0)


# Return a list containing all possible positions together with corresponding player-to-play,
# given the current position is 'state' and it is 'player's turn.
def possibleMoves(state, player):
	
	# For a given 'state' and 'player' return a list containing
	#	move(state, player, _move)
	# for all valid moves '_move'.

	return []



# ==============================================
#	Drawing and printing states
# ==============================================

# Print the current position to the console
def printState(state):
	res = ""

	# Create a string representation of the current position
	print(res)


# Draw the current position (e.g. using pyplot)
def drawState(state):

	# [...]
	plt.show()
