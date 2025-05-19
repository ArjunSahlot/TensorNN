"""
This file contains optimizers that help tune your neural network.
Optimizers enable us to improve our neural network efficiently.
"""

#
#  TensorNN
#  Python machine learning library/framework made from scratch.
#  Copyright Arjun Sahlot 2021
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.
#

from abc import ABC, abstractmethod

import numpy as np

from .tensor import Tensor
from .utils import TensorNNObject

__all__ = [
    "Optimizer",
    "SGD",
    "RMSProp",
    "Adam"
]


class Optimizer(ABC, TensorNNObject):
    """
    Base optimizer class. All optimizers should inherit from this.
    """

    model: TensorNNObject
    learning_rate: float

    def __init__(self, learning_rate: float = 0.01) -> None:
        """
        Initialize the optimizer.

        :param learning_rate: the learning rate of the optimizer
        """
        self.learning_rate = learning_rate
    
    def register(self, model: TensorNNObject) -> None:
        """
        Register the optimizer with the model.

        :param model: the model to register the optimizer with
        """
        self.model = model
    
    def set_lr(self, lr: float) -> None:
        """
        Set the learning rate of the optimizer.

        :param lr: the learning rate to set
        """
        self.learning_rate = lr
    
    def reset(self) -> None:
        """
        Reset the optimizer.
        """
        pass
    
    @abstractmethod
    def step(self) -> None:
        """
        Perform a step of optimization.
        """
        pass


class SGD(Optimizer):
    """
    Stochastic gradient descent optimizer. But, this is actually mini-batch stochastic
    gradient descent. You can make it standard SGD by setting batch_size to 1 in training.
    
    This optimizer is the most basic optimizer and is the most widely used optimizer. It
    works by taking the gradient of the loss function with respect to the weights and biases
    of the model and adjusting them in the opposite direction of the gradient to minimize
    the loss function.
    
    This implementation of SGD also supports momentum. Momentum is a technique which helps
    the model 'walk' down the loss function as a ball would roll down a hill. Successive steps
    in the same direction are accelerated, and the model is able to escape local minima and
    saddle points more easily. To turn off momentum, set the momentum parameter to 0.
    """

    momentum: float
    velocities_w: list[Tensor]
    velocities_b: list[Tensor]

    def __init__(self, learning_rate: float = 0.001, momentum: float = 0.9) -> None:
        """
        Initialize the optimizer.

        :param learning_rate: the learning rate of the optimizer
        :param momentum: the momentum of the optimizer
        """
        super().__init__(learning_rate)
        self.momentum = momentum
        self.velocities_w = []
        self.velocities_b = []

    def register(self, model: TensorNNObject) -> None:
        super().register(model)
        for layer in model.layers:
            self.velocities_w.append(np.zeros_like(layer.weights))
            self.velocities_b.append(np.zeros_like(layer.biases))

    def step(self) -> None:
        for i, layer in enumerate(self.model.layers):
            self.velocities_w[i] = self.momentum * self.velocities_w[i] + (1 - self.momentum) * layer.grad_weights
            self.velocities_b[i] = self.momentum * self.velocities_b[i] + (1 - self.momentum) * layer.grad_biases
            # print(layer.grad_weights, layer.grad_biases)
            layer.step(-self.learning_rate*self.velocities_w[i], -self.learning_rate*self.velocities_b[i])


class RMSProp(Optimizer):
    """
    Root Mean Square Propagation optimizer.
    """

    def __init__(self, learning_rate: float = 0.001, decay: float = 0.9, epsilon: float = 1e-8) -> None:
        """
        Initialize the optimizer.

        :param learning_rate: the learning rate of the optimizer
        :param decay: the decay rate of the optimizer
        :param epsilon: the epsilon value of the optimizer
        """
        super().__init__(learning_rate)
        self.decay = decay

        self.acc_w = []
        self.acc_b = []
        self.epsilon = epsilon
    
    def register(self, model: TensorNNObject) -> None:
        super().register(model)
        for layer in model.layers:
            self.acc_w.append(np.zeros_like(layer.weights))
            self.acc_b.append(np.zeros_like(layer.biases))
    
    def step(self) -> None:
        for i, layer in enumerate(self.model.layers):
            self.acc_w[i] = self.decay * self.acc_w[i] + (1 - self.decay) * np.square(layer.grad_weights)
            self.acc_b[i] = self.decay * self.acc_b[i] + (1 - self.decay) * np.square(layer.grad_biases)
            w_off = -self.learning_rate * layer.grad_weights / (np.sqrt(self.acc_w[i]) + self.epsilon)
            b_off = -self.learning_rate * layer.grad_biases / (np.sqrt(self.acc_b[i]) + self.epsilon)
            layer.step(w_off, b_off)


class Adam(Optimizer):
    """
    Adaptive Moment Estimation, or famously known as Adam, is the most widely used optimizer in the
    machine learning community. By combining the ideas of momentum and RMSProp, people have found that Adam
    works surprisingly well for a lot of situations and is able to converge faster than other optimizers.
    """

    def __init__(self, learning_rate: float = 0.001, momentum: float = 0.9, decay: float = 0.999, epsilon: float = 1e-8) -> None:
        """
        Initialize the optimizer.

        :param learning_rate: the learning rate of the optimizer
        :param momentum: the momentum of the optimizer
        :param decay: the decay rate of the optimizer
        :param epsilon: the epsilon value of the optimizer
        """
        super().__init__(learning_rate)

        self.acc_w = []
        self.acc_b = []
        self.vel_w = []
        self.vel_b = []
        self.momentum = momentum
        self.decay = decay
        self.epsilon = epsilon

        self.iter = 0

    def register(self, model: TensorNNObject) -> None:
        super().register(model)
        for layer in model.layers:
            self.acc_w.append(np.zeros_like(layer.weights))
            self.acc_b.append(np.zeros_like(layer.biases))
            self.vel_w.append(np.zeros_like(layer.weights))
            self.vel_b.append(np.zeros_like(layer.biases))
    
    def step(self) -> None:
        self.iter += 1
        for i, layer in enumerate(self.model.layers):
            self.vel_w[i] = self.momentum * self.vel_w[i] + (1 - self.momentum) * layer.grad_weights
            self.vel_b[i] = self.momentum * self.vel_b[i] + (1 - self.momentum) * layer.grad_biases
            self.acc_w[i] = self.decay * self.acc_w[i] + (1 - self.decay) * np.square(layer.grad_weights)
            self.acc_b[i] = self.decay * self.acc_b[i] + (1 - self.decay) * np.square(layer.grad_biases)

            # error correction
            # self.vel_w[i] /= 1 - np.power(self.momentum, self.iter)
            # self.vel_b[i] /= 1 - np.power(self.momentum, self.iter)
            # self.acc_w[i] /= 1 - np.power(self.decay, self.iter)
            # self.acc_b[i] /= 1 - np.power(self.decay, self.iter)

            w_off = -self.learning_rate * self.vel_w[i] / (np.sqrt(self.acc_w[i]) + self.epsilon)
            b_off = -self.learning_rate * self.vel_b[i] / (np.sqrt(self.acc_b[i]) + self.epsilon)
            layer.step(w_off, b_off)
