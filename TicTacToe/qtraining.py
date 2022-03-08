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

# Import the file containing game specification
from TicTacToe import isOver, gameScore, allSymmetries, startState, randomState, qmove, possibleMoves, printState, drawState



# ==============================================
# 	PARAMETERS
# ==============================================

# Exploration rate
exploration_rate = 0.05

# Number of games played during training
games_num = 10000

# Q learning rate
q_rate = 0.99

# Neural Network Learning rate
learning_rate = 0.0005

# Size of the remembered history
history_size = 200

# Size of the batch recorded in each step
batch_size = 4

# Should the algorithm consider all symmetries of the current position
record_symmetries = False


# ------- SAVING OPTIONS -------------

# Path to the initial model; leave "" to start from a random model
load_path = "qNet.pth"

# Should the algorithm save models after each epoch (after games_num//100 games)
partial_models = True

# Path where the final model will be saved
save_path = "qNet.pth"


# ==============================================
# 	NEURAL NETWORK MODULE
# ==============================================

class QNet(nn.Module):

	def __init__(self):
		super(QNet, self).__init__()
		# Layers
		# size of data: (1, 3, 3):
		self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
		# size of data: (16, 3, 3):
		self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
		# size of data: (32, 3, 3):
		self.conv3 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
		# size of data: (32, 3, 3):
		self.linr1 = nn.Linear(32*9, 128)
		# size of data: (64)
		self.linr2 = nn.Linear(128, 9)
	
	def forward(self, data):
		x = data.view(-1, 1, 3, 3)

		# Convolutional layers
		x = F.relu(self.conv1(x))
		x = F.relu(self.conv2(x))
		x = F.relu(self.conv3(x))

		x = x.reshape(-1, 288)
		
		# Linear layers
		x = torch.tanh(self.linr1(x))
		x = self.linr2(x)

		return x


# Initialize network
valueNetwork 	= QNet()
targetNetwork 	= QNet()
# Loss function
criterion = nn.HuberLoss()
# Optimizer
optimizer = optim.Adam(valueNetwork.parameters(), lr = learning_rate)




# ==============================================
# 	TEACHING MODULE
# ==============================================

# Initialize the history
history = []


# Get the value network's evaluation of the position
def values(state):
	with torch.no_grad():
		inputSt = torch.FloatTensor(state)
		return valueNetwork(inputSt)[0]

# Get the target network's evaluation of the position
def target_values(state):
	with torch.no_grad():
		inputSt = torch.FloatTensor(state)
		return targetNetwork(inputSt)[0]

# Get the value network's evaluation of an action
def value(state, action):
	with torch.no_grad():
		inputSt = torch.FloatTensor(state)
		return valueNetwork(inputSt)[0][action]

# Get the target network's evaluation of an action
def target_value(state, action):
	with torch.no_grad():
		inputSt = torch.FloatTensor(state)
		return targetNetwork(inputSt)[0][action]


# Record a state using target equal to the corresponding result
# We can choose between either recording only the given state or its all symmetries
def record(states, actions, values):

	avg_loss = 0

	# Record the state
	avg_loss += recordSingle(states, actions, values)

	return avg_loss/len(values)

# Record a single state using value as target
def recordSingle(states, actions, values):
	# Initialize input and output
	inputs 		= torch.tensor(states).float()
	targets 	= torch.tensor(values).float()
	
	# Feed the network
	optimizer.zero_grad()
	output 	= valueNetwork(inputs)

	# Recover the predictions
	predictions = torch.gather(output, 1, torch.tensor(actions).unsqueeze(1))
	
	# Evaluate loss and do backpropagation
	loss 	= criterion(targets, predictions)
	loss.backward()
	optimizer.step()

	return loss.mean()


# Generate a random batch from the positions in history
def generateBatch():

	# Get a radnom permutation of the history
	permutation = np.array(range(len(history)))
	np.random.shuffle(permutation)
	permutation = permutation[:batch_size]

	# Initialize the inputs and targets arrays
	inputs 	= []
	actions = []
	targets	= []

	# Create the corresponding batch
	for i in permutation:

		# Take a record
		state, action, reward, playerSwitch, newState = history[i]

		# Add the state to inputs
		inputs.append(state)
		# Add the action to actions
		actions.append(action)	
		# Set the target
		if isOver(newState):
			targets.append(reward)
		else:
			target = reward
			outs = target_values(newState)
			if playerSwitch:
				target -= max(outs)
			else:
				target += max(outs)
			targets.append(reward)

	return inputs, actions, targets


# Generate a random batch from the positions in history and train the network
def teachOnBatch():

	# Upadte the history
	global history
	history = history[-history_size:]

	# Generate a batch
	inputs, actions, targets = generateBatch()

	# Train the network
	avg_loss = record(inputs, actions, targets)

	return avg_loss


# Play a game, beginning as 'player' from the position 'state with exploration rate equal to 'exp_rate'
def playout(state, player, exp_rate, debug=False):
	
	# Train the network on the data generated during the game
	if isOver(state):
		val = gameScore(state)
		return -val

	vals = list(values(state))

	# Choose the best move
	bestVal = max(vals)
	
	# Initialize the value of next move
	nextMove = 0

	# Search for the corresponging best move
	i = 0
	for i in range(len(vals)):
		if bestVal == vals[i]:
			nextMove = i
			break
		i += 1
	
	# With probability exp_rate choose a random move
	exploring = False
	if choices([0,1], [exp_rate, 1-exp_rate])[0] == 0:
		nextMove = random.choice(range(len(vals)))
		exploring = True
	
	# Extract the next state and player
	nextState, nextPlayer, reward = qmove(state, player, nextMove)
	
	# Proceed with the game and get the result
	val = -q_rate*playout(nextState, nextPlayer, exp_rate, debug)
		
	# Print some stuff if debug mode is ON
	if debug:
		print(state)
		print(values(state))
		print(val)
		print(exploring)
		print("===============================")
		
	history.append((state, nextMove, reward, nextPlayer!=player, nextState))

	return val


# Train the model by playing n_games//100 games in each of 100 epochs
# After each epoch, save the current model to folder 'models/'
def trainModel(n_games):

	# Load the best version of the network
	if load_path != "":
		valueNetwork.load_state_dict(torch.load(load_path, map_location=torch.device('cpu')))
	
	# For each of 100 epochs
	for epoch in range(100):

		# Update the target network
		targetNetwork.load_state_dict(valueNetwork.state_dict())

		print("===================   EPOCH  {0:3d}/100   ====================".format(epoch))
		print("Games played:{0:6d}/{1:d}".format(epoch*(n_games//100), n_games))

		# Initialise an array for games results
		values = []

		# Initialize the average loss
		avg_loss = 0

		# Play n_games//100 games and teach the network
		for game in range(n_games//100):

			# Get the result of a played game
			val = playout(startState(), 1, exploration_rate)
			values.append(val)

			# Teach the network on a randomly selected batch from the history
			avg_loss += teachOnBatch()
		
		# Print the average result after the epoch
		print(f"Average score:\t{np.mean(values)}")
		# Print the average loss after the epoch
		print(f"Average loss: \t{avg_loss/(n_games//100)}\n")

		# Save the network model after the epoch
		if partial_models:
			torch.save(valueNetwork.state_dict(), f'models/qNet-{epoch}.pth')

	# In the end, save the final model as 'valueNet.pth'
	torch.save(valueNetwork.state_dict(), save_path)



# ==============================================
# 	RUN THE TRAINING
# ==============================================

trainModel(games_num)
playout(startState(), 1, 0, True)

for i in range(3):
	inp, act, tar = generateBatch()
	print(np.array(inp))
	print(np.array(act))
	print(np.array(tar))
