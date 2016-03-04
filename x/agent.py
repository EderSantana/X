# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division

from .model import Model
from .memory import Memory


class Agent(Model, Memory):
    """Base Agent class

    Parameters
    ----------
    model : :obj:`Model`
        A learning model. Ex: neural network or table
    memory : :obj:`Memory`
        Model's memory for storing experiences for replay and such.

    Attributes
    ----------
    model : :obj:`Model`
        A learning model. Ex: neural network or table
    memory : :obj:`Memory`
        Model's memory for storing experiences for replay and such.
    """
    def __init__(self, model, memory):
        self.model = model
        self.memory = memory

    def build(self, optimizer="sgd", loss="mse",
              experience=None):
        self.model.build(optimizer, loss)
        self.memory.reset(experience)

    def values(self, observation, train=False):
        return self.model.values(observation, train)

    def max_values(self, observation, train=False):
        return self.model.max_values(observation, train)

    def act(self, observation, train=False):
        return self.model.act(observation, train)

    def update(self, batch_size=1, exp_batch_size=1, gamma=0.9, callback=None):
        inputs, targets = self.memory.get_batch(
            self.model, batch_size=batch_size, exp_batch_size=exp_batch_size,
            gamma=gamma, callback=callback)
        loss = self.model.update(inputs, targets)
        return loss

    @property
    def num_actions(self):
        return self.model.num_actions

    @property
    def input_shape(self):
        return self.model.input_shape

    def reset(self):
        self.memory.reset()

    def remember(self, prev_state, action, reward, next_state, game_over):
        self.memory.remember(prev_state, action, reward,
                             next_state, game_over)

    def get_batch(self, model, batch_size=1, exp_batch_size=0,
                  gamma=0.9, callback=None):
        self.memory.get_batch(model, batch_size, exp_batch_size,
                              gamma, callback)
