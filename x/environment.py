# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division

import numpy as np


class Environment(object):
    """Base Environment class

    Attributes
    ----------
    observe : callable
        Get observation of the Environment's state
    update : callable
        Update Environment's state from a given action
    reset : callable
        Reset Environment to an initial state
    reward : callable
        Reward in the current state, returns (new_observation, reward, is_over)
    is_over : bool
        If the Environment is in a terminal state
    state : list
        Inner representation of the Environment state

    """
    def __init__(self):
        raise NotImplementedError

    def observe(self):
        raise NotImplementedError

    def update(self, action):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def reward(self):
        raise NotImplementedError

    @property
    def is_over(self):
        raise NotImplementedError

    @property
    def state(self):
        raise NotImplementedError

    @property
    def description(self):
        raise NotImplementedError


class Catcher(Environment):
    """Catcher

    Agent is a 3 pixel `basket` in the bottom of a square grid of size
    `grid_size`. Single pixels `fruits` fall from the top and return +1 reward
    if catch and -1 else.

    Attributes
    ----------
    grid_size : int
        Size of the square grid
    output_type: str
        Either give state description as raw 'pixels', or as the location of 
        the fruit and basket 'position'. The 'pixels' state space size is
        2**(grid_size**2), while the 'position' state space size is 
        grid_size**3.
    """
    def __init__(self, grid_size=10, output_shape=None, output_type='pixels'):
        self.grid_size = grid_size
        self.output_type = output_type

        if output_shape is None:
            if output_type == 'pixels':
                output_shape = (grid_size**2, )
            elif output_type == 'position':
                output_shape = (3, )
        self.output_shape = output_shape
        self.reset()

    def _update_state(self, action):
        """
        Input: action and states
        Ouput: new states and reward
        """
        state = self.state
        if action == 0:  # left
            action = -1
        elif action == 1:  # stay
            action = 0
        else:
            action = 1  # right
        f0, f1, basket = state[0]
        new_basket = min(max(1, basket + action), self.grid_size-1)
        f0 += 1
        out = np.asarray([f0, f1, new_basket])
        out = out[np.newaxis]

        assert len(out.shape) == 2
        self._state = out

    def _draw_state(self):
        """Convert state description into a square image
        """
        im_size = (self.grid_size,)*2
        state = self.state[0]
        canvas = np.zeros(im_size)
        canvas[state[0], state[1]] = 1  # draw fruit
        canvas[-1, state[2]-1:state[2] + 2] = 1  # draw basket
        return canvas

    def reward(self):
        fruit_row, fruit_col, basket = self.state[0]
        if fruit_row == self.grid_size-1:
            if abs(fruit_col - basket) <= 1:
                return 1
            else:
                return -1
        else:
            return 0

    def observe_image(self):
        canvas = self._draw_state()
        return canvas.reshape(1, 1, self.grid_size, self.grid_size)

    def observe(self):
        if self.output_type == 'pixels':
            canvas = self._draw_state()
            out = canvas.reshape((1, ) + self.output_shape)
        if self.output_type == 'position':
            out = self.state[0]
        return out

    def update(self, action):
        self._update_state(action)
        reward = self.reward()
        game_over = self.is_over
        return self.observe(), reward, game_over

    def reset(self):
        n = np.random.randint(0, self.grid_size-1, size=1)
        m = np.random.randint(1, self.grid_size-2, size=1)
        self._state = np.asarray([0, n, m])[np.newaxis]

    @property
    def state(self):
        return self._state

    @property
    def is_over(self):
        if self.state[0, 0] == self.grid_size-1:
            return True
        else:
            return False

    @property
    def description(self):
        return "Catch game with grid size {}".format(self.grid_size)


class Snake(Environment):

    _apa = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]  # All possible actions
    _grow = 0  # Snake does not yet need to grow any segments

    def __init__(self, grid_size, snake_length=3):
        self.grid_size = grid_size
        self.snake_length = snake_length

        xs = range(1, grid_size - 1)  # X and y coordinates of empty cells
        self._inner_cells = [(x, y) for x in xs for y in xs]

        self.reset()

    def _rand_empty_cell(self):
        empties = set(self._inner_cells)
        empties.difference_update(set(self.snake))
        empties = list(empties)
        np.random.shuffle(empties)
        return empties[0]

    def observe(self):
        frame = np.zeros((self.grid_size, ) * 2)
        frame[[0, -1]] = 1.
        frame[:, [0, -1]] = 1.
        for segment in self.snake:
            frame[segment[1], segment[0]] = 1.
        frame[self.fruit[1], self.fruit[0]] = .5
        return frame

    def update(self, action):
        if self.is_over:
            raise RuntimeWarning, 'Game over'
        if action >= len(self._apa):
            raise ValueError, 'Action not available'

        action = self._apa[action]
        if not sum(action):
            action = self.actions[0]  # Repeat previous action
        self.actions.insert(0, action)
        self.actions.pop()

        if self._grow > 0:
            self.snake.append(self.snake[-1])
            self.actions.append((0, 0))
            self._grow -= 1

        for ix, act in enumerate(self.actions):
            x, y = self.snake[ix]
            delta_x, delta_y = act
            self.snake[ix] = x + delta_x, y + delta_y

        self._reward = 0

        # Snake either ate itself, or hit into wall
        if len(self.snake) > len(set(self.snake)):
            self._reward = -1
        elif not set(self.snake).issubset(set(self._inner_cells)):
            self._reward = -1

        if self.fruit in self.snake:
            self._grow += 1
            self._reward = len(self.snake) - self.snake_length + 1
            self.fruit = self._rand_empty_cell()

        return self.state, self.reward(), self.is_over

    def reset(self):
        center = self.grid_size // 2
        self.snake = [(x, center) for x in range(center, center + self.snake_length)]
        self.actions = [(-1, 0)] * self.snake_length
        self.fruit = self._rand_empty_cell()

    def reward(self):
        if not hasattr(self, '_reward'):
            self._reward = 0

        return self._reward

    @property
    def is_over(self):
        if self.reward() < 0:
            return True
        return False

    @property
    def state(self):
        return self.observe()

    @property
    def description(self):
        return "Snake game with grid size {}".format(self.grid_size)


class Ale(Environment):
    pass
    