from __future__ import absolute_import
import numpy as np
from keras import backend as K
from keras.utils.generic_utils import get_from_module


class Policy(object):
    """Base Policy class
    """
    def __init__(self):
        pass

    def __call__():
        raise NotImplementedError

    def policy(*args, **kwargs):
        raise NotImplementedError

    def max(*args, **kwargs):
        raise NotImplementedError


class Max(Policy):
    def __call__(self, values):
        return K.max(values, axis=-1, keepdims=True)

    @classmethod
    def policy(self, values):
        return np.argmax(values, axis=-1)[np.newaxis].T

    @classmethod
    def max(self, values):
        return np.max(values, axis=-1)[np.newaxis].T


# aliases
max = Max


def get(identifier):
    return get_from_module(identifier, globals(), 'policy', instantiate=True)
