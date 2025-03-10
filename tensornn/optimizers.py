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
from .nn import NeuralNetwork

__all__ = [
    "Optimizer",
    "SGD",
]


class Optimizer(ABC, TensorNNObject):
    """
    Base optimizer class. All optimizers should inherit from this.
    """

    model: NeuralNetwork
    learning_rate: float

    def __init__(self, learning_rate: float = 0.01) -> None:
        """
        Initialize the optimizer.

        :param learning_rate: the learning rate of the optimizer
        """
        self.learning_rate = learning_rate
    
    def register(self, model: NeuralNetwork) -> None:
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

    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.9) -> None:
        """
        Initialize the optimizer.

        :param learning_rate: the learning rate of the optimizer
        :param momentum: the momentum of the optimizer
        """
        super().__init__(learning_rate)
        self.momentum = momentum
        self.velocities_w = []
        self.velocities_b = []
    
    def register(self, model: NeuralNetwork) -> None:
        super().register(model)
        for layer in model.layers:
            self.velocities_w.append(np.zeros_like(layer.weights))
            self.velocities_b.append(np.zeros_like(layer.biases))

    def step(self) -> None:
        for i, layer in enumerate(self.model.layers):
            self.velocities_w[i] = self.momentum * self.velocities_w[i] - self.learning_rate * layer.grad_weights
            self.velocities_b[i] = self.momentum * self.velocities_b[i] - self.learning_rate * layer.grad_biases
            layer.step(self.velocities_w[i], self.velocities_b[i])
