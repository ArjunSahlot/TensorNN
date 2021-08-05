"""
This file contains different types of layers used in neural networks.
Layers need to be able to propagate their inputs forward.
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
from typing import Optional

import numpy as np

from .tensor import Tensor
from .activation import Activation


__all__ = [
    "Layer",
    "Dense",
    "Flatten"
]


class Layer(ABC):
    """
    Abstract base layer class. All layer classes should inherit from this.

    A neural network is composed of layers. A set of inputs are moved from one layer to another.
    Each layer has its own way of calculating the output its own inputs(outputs of previous layer).
    Some layer also have a few tweakable parameters, tweaking these parameters will allow the network
    to learn and adapt the inputs to the correct outputs.
    """

    @abstractmethod
    def forward(self, inputs: Tensor) -> Tensor:
        """
        Calculate a forwards pass of this layer.

        Inheritance:
        A dense layer would connect all inputs to outputs using weights and biases, but maybe your layer
        does something else.

        :param inputs: outputs from the previous layer
        :returns: the output calculated after this layer
        """

    def __iter__(self):
        yield self


class Dense(Layer):
    """
    Each neuron is connected to all neurons in the previous layer. Output is
    calculated by: (output of previous layer * weights) + biases.
    """

    def __init__(
        self,
        num_inputs: int,
        num_neurons: int,
        activation: Optional[Activation] = None,
        zero_biases: bool = True
    ) -> None:
        """
        Initialize dense layer.

        :param num_inputs: the number of neurons/outputs in the previous layer
        :param num_neurons: the number of neurons in this layer/number of outputs of this layer
        :param activation: the activation function applied before the layer output is calculated
        :param zero_biases: whether or not the biases should be initialized to 0, if your network dies try setting this to False
        """

        self.weights: Tensor = Tensor(np.random.randn(num_inputs, num_neurons))
        if zero_biases:
            self.biases: Tensor = Tensor(np.zeros((1, num_neurons)))
        else:
            self.biases: Tensor = Tensor(np.random.randn(1, num_neurons))

        self.activation: Optional[Activation] = activation

    def forward(self, inputs: Tensor) -> Tensor:
        if self.activation is not None:
            activated = self.activation.forward(inputs)
        else:
            activated = inputs

        # @ is __matmul__ from python3.5+, https://www.python.org/dev/peps/pep-0465/
        return activated @ self.weights + self.biases


class Flatten(Layer):
    """

    """
