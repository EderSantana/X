# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division

import numpy as np


class Memory(object):
    """Base Memory class

    Attributes
    ----------
    """
    def __init__(self, *args, **kwargs):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def remember(self, prev_state, action, reward, next_state, game_over):
        raise NotImplementedError

    def get_batch(batch_size=1):
        raise NotImplementedError


class ExperienceReplay(Memory):
    """Experience Replay table

    Parameters
    ----------
    memory_length : int
        memory length, how many memories are hold

    Attributes
    ----------
    memory : list
        list with elements [previous_state, action, reward,
        next_state, game_over]
    experience : list
        Same as `memory` but with pre-loaded memories from a previous
        experiment or human play. This list is not changed during training.
    remember : callable
        Stores a new set of elements [previous_state, action, reward,
        next_state, game_over] to `memory`.
    get_batch : callable
        Sample elements from `memory` and `experience` and calculates input and
        desired outputs for training a :obj:`Model`.
    """
    def __init__(self, memory_length=1, experience=None):
        self.memory_length = memory_length
        self.experience = experience
        self.memory = list()

    def reset(self, experience=None):
        if experience:
            self.experience = experience
        self.memory = list()

    def remember(self, prev_state, action, reward, next_state, game_over):
        self.memory.append({
            'prev_state': prev_state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'game_over': game_over
        })
        if len(self.memory) > self.memory_length:
            del self.memory[0]

    def get_batch(self, model, batch_size=1, exp_batch_size=0,
                  gamma=0.9, callback=None):
        """Get Batch
        Get input, target samples from memory.

        Parameters
        ----------
        model : :obj:Model
           The model that will be trained, used to calculate future values for
           TD-learning.
        batch_size : int
           Number of samples from :attr:`ExperienceReplay.memory`
        exp_batch_size : int
           Number of samples from :attr:`ExperienceReplay.experience`
        gamma : float
           \gamma discount factor of future rewards
        callback : callable
           A callaback to calculate target values from :obj:Model and
           `next_state`. If None, we use a default Q-learning target:
           t = r_t + \gamma argmax_{a'} Q(s', a')

        Returns
        -------
        inputs : `numpy.array`
           Input observations of "current" states
        targets : `numpy.array`
           Target values of the "future" states
        """

        batch_mem = min(batch_size, len(self.memory))
        if exp_batch_size > 0:
            batch_exp = min(exp_batch_size, len(self.experience))
        else:
            batch_exp = 0
        bsize = batch_mem + batch_exp
        inputs = np.zeros((bsize, ) + model.input_shape)
        actions = np.zeros((bsize, 1))
        targets = np.zeros((bsize, 1))

        # sample from memory
        rlst = np.random.randint(0, len(self.memory), size=batch_mem)
        for i, idx in enumerate(rlst):
            prev_state = self.memory[idx]['prev_state']
            reward = self.memory[idx]['reward']
            next_state = self.memory[idx]['next_state']
            inputs[i] = prev_state.reshape((1, ) + model.input_shape)
            actions[i] = self.memory[idx]['action']
            next_state = next_state.reshape((1, ) + model.input_shape)
            if callback:
                targets[i] = callback(model, next_state)
            else:
                #print('next state:')
                #print next_state
                #print('targets[i]:')                
                #print(targets[i])
                targets[i] = reward + gamma * model.max_values(next_state, train=True)

        # sample from experience
        if not self.experience and exp_batch_size > 0:
            return inputs, targets, actions
        else:
            rlst = np.random.randint(0, len(self.memory), size=batch_exp)
            for k, idx in enumerate(rlst):
                prev_state = self.memory[idx]['prev_state']
                reward = self.memory[idx]['reward']
                next_state = self.memory[idx]['next_state']
                inputs[i+k] = prev_state.reshape((1, ) + model.input_shape)
                actions[i+k] = self.memory[idx]['action']
                next_state = next_state.reshape((1, ) + model.input_shape)
                if callback:
                    targets[i+k] = callback(model, next_state)
                else:
                    targets[i+k] = reward + gamma * model.max_values(
                        next_state, train=False)
            return inputs, targets, actions

    @property
    def description(self):
        dstr = "Experience Replay \n\t Memory length: {}"
        return dstr.format(self.memory_length)
