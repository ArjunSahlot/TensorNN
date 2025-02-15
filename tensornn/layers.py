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
from .activation import Activation, NoActivation
from .utils import atleast_2d, TensorNNObject


__all__ = [
    "Layer",
    "Dense",
    "flatten",
]


class Layer(ABC, TensorNNObject):
    """
    Abstract base layer class. All layer classes should inherit from this.

    A neural network is composed of layers. A set of inputs are moved from one layer to another.
    Each layer has its own way of calculating the output of its own inputs(outputs of previous layer).
    Some layers also have a few tweakable parameters, tweaking these parameters will allow the network
    to learn and adapt to the inputs to produce the correct outputs.
    """

    neurons: int
    weights: Tensor
    biases: Tensor
    activation: Activation
    gradients: Tensor

    def __init__(
        self,
        num_neurons: int,
        activation: Activation = NoActivation(),
    ) -> None:
        """
        Initialize a TensorNN Layer

        :param num_neurons: the number of neurons in this layer/number of outputs of this layer
        :param activation: the activation function applied before the layer output is calculated, defaults to NoActivation
        """
        self.neurons = num_neurons
        self.activation: Activation = activation

        self.input = False
        self.biases: Tensor = Tensor(np.random.randn(1, num_neurons))
        self.gradients = None
        self.weights = None  # Initialized on Layer.register()

    @abstractmethod
    def forward(self, inputs: Tensor) -> Tensor:
        """
        Calculate a forwards pass of this layer, before and after activation.

        :param inputs: outputs from the previous layer
        :returns: the output calculated after this layer before and after activation
        """

    @abstractmethod
    def register(self, prev: int) -> None:
        """
        Number of inputs in the previous layer. This is called whenever the NeuralNetwork is
        registered(NeuralNetwork.register()) with the optimizer and loss, it calls this method
        for all layers giving information to it. If your layer doesn't need this, you don't need
        to implement this.

        :param prev: number of neurons in previous layer
        :returns: Nothing
        """

    def reset_gradients(self) -> None:
        """
        Reset the gradients of the layer. This is called at the start of each epoch.
        """
        self.gradients = Tensor(np.zeros(self.weights.shape))



class Dense(Layer):
    """
    Each neuron is connected to all neurons in the previous layer. Output is
    calculated by: (output of previous layer * weights) + biases.
    """

    def __init__(
        self,
        num_neurons: int,
        activation: Activation = NoActivation(),
        # zero_biases: bool = True,
    ) -> None:
        """
        Initialize dense layer.

        :param num_neurons: the number of neurons in this layer/number of outputs of this layer
        :param activation: the activation function applied before the layer output is calculated
        # :param zero_biases: whether or not the biases should be initialized to 0, if your network dies try setting this to False
        #TODO have stuff like zero_biases go in a dictionary like config or options
        """
        super().__init__(num_neurons, activation)
        
        # Used for backpropagation
        self.inputs: Tensor
        self.d_weights: Tensor
        self.d_biases: Tensor

    def forward(self, inputs: Tensor) -> Tensor:
        if self.input:
            return inputs, self.activation.forward(inputs)

        # I don't like the way it's converting from 1D to 2D and back to 1D, but it makes it
        # easier to understand for the user. Might change to sacrifice simplicity for performance.
        inputs = atleast_2d(inputs)
        self.inputs = inputs
        values = inputs @ self.weights + self.biases
        values = values[0]
        return values, self.activation.forward(values)
    
    def backward(self, deriv_cost: Tensor) -> Tensor:
        if self.input:
            return

        deriv_cost = self.activation.derivative(self.inputs @ self.weights + self.biases)

        self.d_weights = self.inputs.T @ deriv_cost
        # print("d_weights")
        # print(self.d_weights)
        self.d_biases = np.sum(deriv_cost, axis=0)
        # print("d_biases")
        # print(self.d_biases)
        return deriv_cost @ self.weights.T

    def register(self, prev: Optional[int]) -> None:
        if prev is None:
            self.input = True
        else:
            self.weights: Tensor = Tensor(np.random.randn(prev, self.neurons))

    def __repr__(self) -> str:
        return f"TensorNN.{self.__class__.__name__}(neurons={self.neurons}, activation={self.activation})"
