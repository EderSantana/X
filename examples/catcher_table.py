#%%
import argparse

# Parameters
parser = argparse.ArgumentParser(description='Catcher')
parser.add_argument('--grid', dest='grid', type=int, default=11, help='Game grid size.')
parser.add_argument('--memory', dest='memory', type=int, default=500, help='Experience replay memory length')
parser.add_argument('--epoch', dest='epoch', type=int, default=1000, help='Number for epochs')
parser.add_argument('--batch', dest='batch', type=int, default=50, help='Batch size retrieved from experience replay meomory')
parser.add_argument('--epsilon', dest='epsilon', type=float, default=0.1, help='Exploration rate epsilon')
parser.add_argument('--gamma', dest='gamma', type=float, default=0.9, help='Discount rate gamma')
parser.add_argument('--save', dest='save', type=str, default='catcher_agent.h5', help='Discount rate gamma')
parser.add_argument('--output', dest='output', type=str, default='catcher_output.gif', help='Path to save output animation.')
args = parser.parse_args()


from x.environment import Catcher
from x.models import TableModel
from x.memory import ExperienceReplay
from x.agent import DiscreteAgent

import numpy as np

num_actions = 3
nb_rows, nb_cols = args.grid, args.grid
state_dim = 3
state_dim_values=(np.r_[0:nb_rows],np.r_[0:nb_cols],np.r_[1:nb_cols-1])

# To run an experiment, the Agent needs an Enviroment to iteract with
env = Catcher(grid_size=args.grid, output_type='position')

# Create Table Model
model = TableModel(state_dim=3, num_actions=num_actions)

# Memory
M = ExperienceReplay(memory_length=args.memory)

# Agent
agent = DiscreteAgent(model, M)

# Configure and build table model
agent.compile(state_dim_values, lr=0.2, policy_rule="max")

agent.learn(env, epoch=args.epoch, batch_size=args.batch)

# Test the agent following the learned policy
pl_epoch = 5
agent.play(env, epoch=pl_epoch, visualize={'filepath': args.output, 'n_frames': pl_epoch*(nb_rows-1), 'gray': True})
