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

# The state is represented by a tuple (pos, n_moves) where:
# 	n_moves:	represents the number of moves that has been made
# 	pos:		1D array of size 12, whose entries represent the map:

#  0  |  1  |  2  |  3
# ----------------------
#  4  |  5  |  6  |  7
# ----------------------
#  8  |  9  |  10 |  11

# Obstacles are marked as -1
# The current position of the player is marked as 1
# The remaining squares are labelled as 0

# For the initial state, pos is:

#  0  |  0  |  0  |  0
# ----------------------
#  0  |  -1 |  0  |  0
# ----------------------
#  0  |  1  |  0  |  -1

# which can be represented as an array: [0,0,0,0,0,-1,0,0,0,1,0,-1]
# This means that there are obstacles on squars 5 and 11, and initially,
# the player is on square 9
 
# Each turn, the player can choose between performing one of the following actions:
#	0: move LEFT
#	1: move RIGHT
#	2: move UP
#	3: move DOWN

# If the action is impossible (either because of an obstacle or the end of the board),
# then we do not move, but this procedure costs us -100
  
# The game finishes when the player reaches square 3, 4 or 7, or after 20 moves

# In the former case, the reward is:

# - finishing at square 3:	10
# - finishing at square 4:	1
# - finishing at square 7:	-10



# ==============================================
# 	The end of the game
# ==============================================

# Set the punishment for illegal moves
def punishment():
	return -1

# Return True if the current state is an ending state, and False otherwise
def isOver(state):
	pos, n_moves = state
	return (pos[3]==1 or pos[4]==1 or pos[7]==1 or n_moves>=20)

# Return the outcome of the game	
def gameScore(state):
	return 0


# ==============================================
#	Basic state
# ==============================================

def startState():
	# Return the starting state, according to the chosen gamestate description model
	return ([0,0,0,0,0,-1,0,0,0,1,0,-1], 0)

# Return a random valid state of the game, useful for analyzing the performance of the agent
def randomState():
	# Return a random state
	res = [0,0,0,0,0,-1,0,0,0,0,0,-1]
	res[np.random.randint(0,5)+6*np.random.randint(0,2)] = 1
	n_moves = np.random.randint(4,20)
	return (res, n_moves)


# ==============================================
#	Possible moves
# ==============================================

def reward(square):
	if square == 3:
		return 10
	if square == 4:
		return 1
	if square == 7:
		return -10
	return 0

# Given the current position of the game, player whose turn it is and a move
# return the corresponding next position
def move(state, player, _move):
	
	# Evaluate a tuple (nextState, nextPlayer, reward) where:
	# - nextState:  represents the position after 'player' perfomrms '_move' on 'state'
	# - nextPlayer: represents the player whose turn it will be after the move
	#				In this case, it is always 1, as there is only one player
	# - reward:		the reward for performing _move on state

	# Get the current position and the number of moves that has been made
	pos, n_moves = state
	# Prepare an array for future position
	new_pos = deepcopy(pos)
	# Get the current player's position
	curr_p = np.argmax(new_pos)

	# Check if the move was valid
	# If we wanted to move LEFT:
	if _move == 0 and curr_p%4!=0 and pos[curr_p-1] == 0:
		new_pos[curr_p] 	= 0
		new_pos[curr_p-1] 	= 1
		return ((new_pos, n_moves+1), player, reward(curr_p-1))
	# If we wanted to move RIGHT:
	if _move == 1 and curr_p%4!=3 and pos[curr_p+1] == 0:
		new_pos[curr_p] 	= 0
		new_pos[curr_p+1] 	= 1
		return ((new_pos, n_moves+1), player, reward(curr_p+1))
	# If we wanted to move UP:
	if _move == 2 and curr_p>3 and pos[curr_p-4] == 0:
		new_pos[curr_p] 	= 0
		new_pos[curr_p-4] 	= 1
		return ((new_pos, n_moves+1), player, reward(curr_p-4))
	# If we wanted to move DOWN:
	if _move == 3 and curr_p<8 and pos[curr_p+4] == 0:
		new_pos[curr_p] 	= 0
		new_pos[curr_p+4] 	= 1
		return ((new_pos, n_moves+1), player, reward(curr_p+4))

	# Otherwise, the move was invalid:
	return ((pos, n_moves+1), player, punishment())


# ==============================================
#	Drawing and printing states
# ==============================================

# Print the current position to the console
def printState(state):
	
	pos, n_moves = state
	print("Number of moves played:\t", n_moves, "\n")
	# Create a string representation of the current position
	print("Position:")
	print("------")
	for i in range(3):
		s="|"
		for j in range(4):
			if pos[4*i+j]==0:
				s+=" "
			elif pos[4*i+j]==1:
				s+="X"
			elif pos[4*i+j]==-1:
				s+= "@"
			else:
				s+="?"
		print(s+"|")
	print("------")