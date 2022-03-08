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
from DotsAndBoxes import isOver, gameScore, startState, randomState, move, printState, possibleActions, prepareInput



# ==============================================
# 	PARAMETERS
# ==============================================

# Exploration rate
exploration_rate = 0.7

# Decay of exploration rate (guarantees better choices as the game progresses)
exp_decay = exploration_rate/100

# Number of games played during training
games_num = 100000

# Q learning rate
q_rate = 0.99

# Neural Network Learning rate
learning_rate = 0.0005

# Size of the remembered history
history_size = 100

# Size of the batch recorded in each step
batch_size = 4

# Should the algorithm consider all symmetries of the current position
record_symmetries = False


# ------- SAVING OPTIONS -------------

# Path to the initial model; leave "" to start from a random model
load_path = ""

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
		# Convolutional layers
		self.conv1 = nn.Conv2d(1, 8, 2)
		self.conv2 = nn.Conv2d(8, 32, 3, padding=1)
		self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
		# Linear layers
		self.linr1 = nn.Linear(18*64, 512)
		self.linr2 = nn.Linear(512, 24)
		# Dropouts
		self.convdrop = nn.Dropout(0.1)
		self.linrdrop = nn.Dropout(0.5)

	def forward(self, data):
		# Prepare the data in the right shape
		x = data.view(-1, 1, 7, 4)
		# Run through convolutional layers
		x = F.relu(self.conv1(x))
		x = F.relu(self.conv2(x))
		x = F.relu(self.conv3(x))
		# Flatten x
		x = torch.flatten(x, 1)
		# Dropout
		x = self.convdrop(x)
		# Run through linear layer
		x = F.relu(self.linr1(x))
		# Dropout
		x = self.linrdrop(x)
		# Evaluate the final result
		return self.linr2(x)

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
		inputSt = torch.FloatTensor(prepareInput(state))
		return valueNetwork(inputSt)[0]

# Get the target network's evaluation of the position
def target_values(state):
	with torch.no_grad():
		inputSt = torch.FloatTensor(prepareInput(state))
		return targetNetwork(inputSt)[0]

# Get the value network's evaluation of an action
def value(state, action):
	with torch.no_grad():
		inputSt = torch.FloatTensor(prepareInput(state))
		return valueNetwork(inputSt)[0][action]

# Get the target network's evaluation of an action
def target_value(state, action):
	with torch.no_grad():
		inputSt = torch.FloatTensor(prepareInput(state))
		return targetNetwork(inputSt)[0][action]


# Record states using target equal to the corresponding result
def record(states, actions, values):
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
		# Add the position to inputs
		inputs.append(prepareInput(state))
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

	# Get the possible actions for the current position
	possible_actions = possibleActions(state)
	# and calculate Q(state, action) for each of them
	vals = values(state).detach().numpy()[possible_actions]
	# Evaluate the best move according to the network
	nextMove = possible_actions[np.argmax(vals)]

	# With probability exp_rate choose a random move
	exploring = False
	if choices([0,1], [exp_rate, 1-exp_rate])[0] == 0:
		nextMove = random.choice(possible_actions)
		exploring = True

	# Extract the next state and player
	nextState, nextPlayer, reward = move(state, player, nextMove)

	# Proceed with the game and get the result
	val = reward
	result = q_rate*playout(nextState, nextPlayer, max(0.01, exp_rate-exp_decay), debug)

	# If the player changed during the move, we need to substract the result of the game
	# instead of adding it to the reward:
	#	if nextPlayer == player:	Q(state, move) = reward(move) + max {Q(next_state, a)}
	#	if nextPlayer != player:	Q(state, move) = reward(move) - max {Q(next_state, a)}
	if nextPlayer == player:
		val += result
	else:
		val -= result

	# Print some stuff if debug mode is ON
	if debug:
		printState(state)
		print(np.array(values(state)))
		print(val, result, reward)
		print(exploring)
		print("===============================")

	# Add a new record to the history
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

		# Measure time spent during this epoch
		start_time = time()

		# Play n_games//100 games and teach the network
		for game in range(n_games//100):

			# Get the result of a played game
			val = playout(startState(), 1, exploration_rate)
			values.append(val)

			# Teach the network on 2 randomly selected batches from the history
			avg_loss += (teachOnBatch() + teachOnBatch())/2

		# Measure time spent during the epoch
		end_time = time()

		# Print the average result after the epoch
		print(f"Average score:\t{np.mean(values)}")
		# Print the average loss after the epoch
		print(f"Average loss: \t{avg_loss/(n_games//100)}")
		# Print the time spent during this epoch
		print(f"Time spent:\t{(end_time-start_time)}s\n")

		# Save the network model after the epoch
		if partial_models:
			torch.save(valueNetwork.state_dict(), f'models/qNet-{epoch}.pth')

	# In the end, save the final model as 'valueNet.pth'
	torch.save(valueNetwork.state_dict(), save_path)



# ==============================================
# 	RUN THE TRAINING
# ==============================================

trainModel(games_num)
valueNetwork.load_state_dict(torch.load("qNet.pth", map_location=torch.device('cpu')))
playout(startState(), 1, 0, True)
