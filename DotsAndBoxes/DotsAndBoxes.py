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


# The state of the game is defined as a 1D array of size 24
# representing the state (0 - empty, 1 - taken) of the edges,
# where the egdes with given numbers are:

# 	# - 0 - # - 1 - # - 2 - #
# 	|		|		|		|
# 	3		4		5		6
# 	|		|		|		|
# 	# - 7 - # - 8 - # - 9 - #
#  	|		|		|		|
# 	10		11		12		13
# 	|		|		|		|
# 	# - 14- # - 15- # - 16- #
# 	|		|		|		|
# 	17		18		19		20
# 	|		|		|		|
# 	# - 21- # - 22- # - 23- # 

# Move i corresponds to placing an edge with number i


# ==============================================
# 	The end of the game
# ==============================================

# Define a state representing invalid choice of moves
def wrongMoveState():
	return np.zeros(24)+2

# Define the punishment for invalid moves (in case something goes wrong in the code)
def punishment():
	return -100

# Return True if the current state is an ending state, and False otherwise
def isOver(state):
	if state[0]==2:
		return True
	if 0 in state:
		return False
	return True

# Return the outcome of the game
# # In our case, the result will be calculated during the game
# (filling a box will be rewarded immediately after the move),
# so there is no need to specify additional reward for the outcome
# of the game	
def gameScore(state):
	return 0


# ==============================================
#	Initial state
# ==============================================

def startState():
	# Return the starting state, according to the chosen gamestate description model
	return np.zeros(24)

# Return a random valid state of the game, useful for analyzing the performance of the agent
def randomState():
	return np.random.choice([0,1], 24)


# ==============================================
#	Possible moves
# ==============================================

# Reutrn all lists of indexes of edges forming 1x1 squarees on the grid
def squareIds():
	return np.array([[0, 3, 4, 7],
					[1, 4, 5, 8],
					[2, 5, 6, 9],
					[7, 10, 11, 14],
					[8, 11, 12, 15],
					[9, 12, 13, 16],
					[14, 17, 18, 21],
					[15, 18, 19, 22],
					[16, 19, 20, 23]])

# Calculate the number of filled squares in the current state
def squares(state):
	# Initialize the counter
	count = 0
	# Check every possible square
	for sqIds in squareIds():
		if sum(state[sqIds])==4:
			count += 1
	# Return the result
	return count

# Given the current position of the game, player whose turn it is and a valid move
# return the corresponding next position
def move(state, player, _move):
	
	# Evaluate a tuple (nextState, nextPlayer, reward) where:
	# - nextState:  represents the position after 'player' perfomrms '_move' on 'state'
	# - nextPlayer: represents the player whose turn it will be after the move

	# Initialize new state
	newState = np.array(deepcopy(state))

	# Check if the move is legal
	if newState[_move]!=0:
		return (wrongMoveState(), player, punishment())
	
	# Place the line
	newState[_move]=1

	# Check the number of closed squares for the current and next state:
	curr_squares = squares(state)
	next_squares = squares(newState)

	# If the number of filled squares has changed, it's still player's turn
	if curr_squares!=next_squares:
		return (newState, player, next_squares-curr_squares)

	# Otherwise, the reward is 0 and we move to the second player
	return (newState, -player, 0) 


# Return a list containing all possible moves for a given position
def possibleActions(state):
	return np.where(state==0)[0]



# ==============================================
#	Drawing and printing states
# ==============================================

# Print the current position to the console
def printState(state):
	res = ""
	for i in range(3):
		res += "#"
		for j in range(3):
			if state[7*i+j]==0:
				res += "   "
			else:
				res += "---"
			res += "#"
		res += "\n"
		for j in range(3):
			if state[7*i+3+j]==0:
				res += " "
			else:
				res += "|"
			if state[7*i+j]==1 and state[7*i+j+3]==1 and state[7*i+j+4]==1 and state[7*i+j+7]==1:
				res += "@@@"
			else:
				res += "   "
		if state[7*i+6]==0:
			res += " "
		else:
			res += "|"
		res += "\n"
	res += "#"
	for j in range(3):
		if state[21+j]==0:
			res += "   "
		else:
			res += "---"
		res += "#"
	res += "\n"
	print(res)



# ==============================================
#	Preparing input for the network
# ==============================================

def prepareInput(state):
	input = deepcopy(state)
	return np.insert(input, [3, 10, 17, 24], 0)