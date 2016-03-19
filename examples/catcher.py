#%%
import argparse

# Parameters
parser = argparse.ArgumentParser(description='Catcher')
parser.add_argument('--grid', dest='grid', type=int, default=11, help='Game grid size.')
parser.add_argument('--hidden', dest='hidden', type=int, default=100, help='Number of neuron in the hidden layer.')
parser.add_argument('--memory', dest='memory', type=int, default=500, help='Experience replay memory length')
parser.add_argument('--epoch', dest='epoch', type=int, default=1000, help='Number for epochs')
parser.add_argument('--batch', dest='batch', type=int, default=50, help='Batch size retrieved from experience replay meomory')
parser.add_argument('--epsilon', dest='epsilon', type=float, default=0.1, help='Exploration rate epsilon')
parser.add_argument('--gamma', dest='gamma', type=float, default=0.9, help='Discount rate gamma')
parser.add_argument('--save', dest='save', type=str, default='catcher_agent.h5', help='Discount rate gamma')
parser.add_argument('--output', dest='output', type=str, default='catcher_output.gif', help='Path to save output animation.')
args = parser.parse_args()

from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD

from x.environment import Catcher
from x.models import KerasModel
from x.memory import ExperienceReplay
from x.agent import DiscreteAgent

num_actions = 3
nb_filters, nb_rows, nb_cols = 32, 3, 3

# keras model
keras_model = Sequential()
keras_model.add(Dense(args.hidden, activation="relu", input_dim=args.grid**2))
keras_model.add(Dense(args.hidden, activation="relu"))
keras_model.add(Dense(num_actions))

# X wrapper for Keras
model = KerasModel(keras_model)

# Memory
M = ExperienceReplay(memory_length=args.memory)

# Agent
A = DiscreteAgent(model, M)
# SGD optimizer + MSE cost + MAX policy = Q-learning as we know it
A.compile(optimizer=SGD(lr=0.2), loss="mse", policy_rule="max")

# To run an experiment, the Agent needs an Enviroment to iteract with
catcher = Catcher(grid_size=args.grid)
A.learn(catcher, epoch=args.epoch, batch_size=args.batch)

# Test the agent following the learned policy
A.play(catcher, epoch=100, visualize={'filepath': args.output, 'n_frames': 270, 'gray': True})
#%%