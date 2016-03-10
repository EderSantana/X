# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division

import numpy as np

from keras.utils.generic_utils import Progbar
from agnez.video import make_gif


class Agent(object):
    """Base Agent class

    Parameters
    ----------
    model : :obj:`Model`
        A learning model. Ex: neural network or table
    memory : :obj:`Memory`
    """
    def __init__(self, model, memory):
        self.model = model
        self.memory = memory


class DiscretAgent(Agent):
    """Single Discret action Agent

    Parameters
    ----------
    model : :obj:`Model`
        A learning model. Ex: neural network o} table
    memory : :obj:`Memory`
        Model's memory for storing experiences for replay and such.
    epsilon : callable
        A rule to define if model explore or exploit
        TODO: generalize this to a class that controls if it should explore and
        define custom explorations rules

    """
    def __init__(self, model, memory, epsilon=None):
        super(DiscretAgent, self).__init__(model, memory)
        if epsilon is None:
            self.epsilon = lambda *args: .1

    def compile(self, optimizer="sgd", loss="mse", policy_rule="max",
                experience=None):
        self.model.compile(optimizer, loss, policy_rule)
        self.memory.reset(experience)

    def values(self, observation, train=False):
        return self.model.values(observation, train)

    def max_values(self, observation, train=False):
        return self.model.max_values(observation, train)

    def policy(self, observation, train=False):
        if train and np.random.randint <= self.epsilon():
            return np.random.randint(0, self.num_actions)
        else:
            return self.model.policy(observation, train)

    def update(self, batch_size=1, exp_batch_size=0, gamma=0.9, callback=None):
        inputs, targets = self.get_batch(
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
        return self.memory.get_batch(model, batch_size, exp_batch_size,
                                     gamma, callback)

    def learn(self, env, epoch=1, batch_size=1, exp_batch_size=0,
              gamma=0.9, reset_memory=False, verbose=1, callbacks=None):
        """Train Agent to play Enviroment env

        Parameters
        ----------
        env : :obj:`Enviroment`
            The enviroment the agent learn to play
        epoch : int
            number of complete episodes to play
        batch_size : int
            number of experiences to replay per step
        exp_batch_size : int
            number of experiences to replay from the consolidated
            :attr:`ExperienceReplayexperience.experience`.
        gamma : float
            discount factor
        reset_memory : bool
            if we should restart :attr:`ExperienceReplay.memory` before
            starting the game.
        verbose : int
            controls how much should we print
        callbacks : list of callables
            TODO: Add callback support

        """
        print("Learning started!")
        print("[Environment]: {}".format(env.description))
        print("[Model]: {}".format(self.model.description))
        print("[Memory]: {}".format(self.memory.description))
        if reset_memory:
            self.reset()
        progbar = Progbar(epoch)
        rewards = 0
        for e in xrange(epoch):
            # reset enviroment
            env.reset()
            game_over = False
            loss = 0

            # get initial observation, start game
            obs_t = env.observe()
            # Run an episonde
            while not game_over:
                obs_tm1 = obs_t
                action = self.policy(obs_tm1)

                # apply action, get rewards and new state
                obs_t, reward, game_over = env.update(action)
                rewards += reward

                # store experience
                self.remember(obs_tm1, action, reward, obs_t, game_over)

                # adapt model
                loss += self.update(batch_size=batch_size,
                                    exp_batch_size=exp_batch_size,
                                    gamma=gamma)
            if verbose == 1:
                progbar.add(1, values=[("loss", loss), ("rewards", rewards)])

    def play(self, env, epoch=1, batch_size=1, visualize=None, verbose=1):
        print("Free play started!")
        frames = np.zeros((0, ) + env.observe_image().shape[1:])
        frames = frames.transpose(0, 2, 3, 1)
        rewards = 0
        progbar = Progbar(epoch)

        for e in xrange(epoch):
            env.reset()
            game_over = False
            loss = 0
            # get initial observation, start game
            obs_t = env.observe()
            while not game_over:
                obs_tm1 = obs_t

                # get next action
                action = self.policy(obs_tm1, train=False)

                # apply action, get rewareds and new state
                obs_t, reward, game_over = env.update(action)
                rewards += reward

                frame_t = env.observe_image().transpose(0, 2, 3, 1)
                frames = np.concatenate([frames, frame_t], axis=0)

            if verbose == 1:
                progbar.add(1, values=[("loss", loss), ("rewards", rewards)])

        if visualize:
            print("Making gif!")
            frames = np.repeat(frames, 3, axis=-1)
            make_gif(frames[:-visualize['n_frames']],
                     filepath=visualize['filepath'], gray=visualize['gray'])
            print("See your gif at {}".format(visualize['filepath']))
