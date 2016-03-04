# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division

import numpy as np
import keras.backend as K
from keras import optimizers, objectives


class Model(object):
    """Base Model class

    Attributes
    ----------
    build : callable
        Initialize model weights or values
    values : callable
        return action values given input observation
    max_values : callable
        np.max of `values`
    act : callable
        Return action given an input enviroment observation as np.argmax of
        `values`
    num_action : int
        Number of possible actions
    input_shape : :list:`int`
        List with dimensions of the input observation
    """
    def __init__(self, *args, **kwargs):
        raise NotImplementedError

    def build(self):
        raise NotImplementedError

    def values(self, observation, train=False):
        raise NotImplementedError

    def max_values(self, observation, train=False):
        raise NotImplementedError

    def act(self, observation, train=False):
        raise NotImplementedError

    def update(self, inputs, targets):
        raise NotImplementedError

    @property
    def num_actions(self):
        raise NotImplementedError

    @property
    def input_shape(self):
        raise NotImplementedError


class KerasModel(Model):
    def __init__(self, keras_model):
        """Keras model wrapper

        Attributes
        ----------
        model : :obj:`Keras.model.Sequential`
            A Keras Sequential neural network model.
        """
        self.keras_model = keras_model

    def build(self, optimizer="sgd", loss="mse"):
        """Initialize model weights and compile functions
        """
        kmodel = self.keras_model
        kmodel.build()

        # Get optimizer and loss
        kmodel.optimizer = optimizers.get(optimizer)
        kmodel.loss = objectives.get(loss)

        # inputs
        kmodel.X_train = kmodel.get_input(train=True)
        kmodel.X_test = kmodel.get_input(train=False)

        # outputs
        _values = kmodel.get_output(train=True)
        _values_test = kmodel.get_output(train=False)
        _max_values = K.expand_dims(K.max(_values, axis=-1), dim=-1)
        _max_values_test = K.expand_dims(K.max(_values_test, axis=-1), dim=-1)

        # targets
        kmodel.y = K.placeholder(ndim=K.ndim(_max_values))

        # Eval losses
        train_loss = K.mean(kmodel.loss(kmodel.y, _max_values))
        test_loss = K.mean(kmodel.loss(kmodel.y, _max_values_test))

        # Eval updates
        updates = kmodel.optimizer.get_updates(kmodel.trainable_weights,
                                               kmodel.constraints,
                                               train_loss)
        updates += kmodel.updates

        # Compile functions
        self._train = K.function([kmodel.X_train, kmodel.y], [train_loss],
                                 updates=updates)
        self._values_test = K.function([kmodel.X_test], [_values_test],
                                       updates=kmodel.state_updates)
        self._test = K.function([kmodel.X_test, kmodel.y], [test_loss],
                                updates=kmodel.state_updates)
        self._values = K.function([kmodel.X_train], [_values],
                                  updates=kmodel.state_updates)

        # Compile action functions
        X = self.keras_model.get_input(train=True)
        Y = self.keras_model.get_output(train=True)
        self._act_train = K.function([X], [Y])

    def values(self, observation, train=False):
        if train:
            vals = self._values([observation])[0]
        else:
            vals = self._values_test([observation])[0]
        return vals

    def max_values(self, observation, train=False):
        vals = self.values(observation, train)
        return np.max(vals, axis=-1)[np.newaxis].T

    def act(self, observation, train=False):
        vals = self.values(observation, train)[0]
        return np.argmax(vals, axis=-1)[np.newaxis].T

    def update(self, inputs, targets):
        loss = self._train([inputs, targets])[0]
        return loss

    @property
    def num_actions(self):
        return self.keras_model.output_shape[-1]

    @property
    def input_shape(self):
        return self.keras_model.input_shape[1:]
