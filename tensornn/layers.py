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
        self.biases: Tensor = None
        self.grad_weights: Tensor = None
        self.grad_biases: Tensor = None
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
        self.grad_weights = np.zeros_like(self.weights)
        self.grad_biases = np.zeros_like(self.biases)


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
        self.calculated: Tensor
        self.grad_weights: Tensor
        self.grad_biases: Tensor
    
    def forward(self, inputs: Tensor) -> Tensor:
        if self.input:
            return inputs, self.activation.forward(inputs)

        self.inputs = inputs
        values = inputs @ self.weights + self.biases
        self.calculated = values
        return values, self.activation.forward(values)
    
    def backward(self, accumulated_gradient: Tensor) -> Tensor:
        if self.input:
            return

        dc_da = accumulated_gradient
        da_dz = self.activation.derivative(self.calculated)

        dc_dz = dc_da * da_dz

        self.grad_weights = self.inputs.T @ dc_dz
        self.grad_biases = dc_dz.sum(axis=0)

        return dc_dz @ self.weights.T

    def step(self, adjust_w: Tensor, adjust_b: Tensor) -> None:
        if self.input:
            return

        self.weights += adjust_w
        self.biases += adjust_b

    def register(self, prev: Optional[int]) -> None:
        if prev is None:
            self.input = True
        else:
            self.weights = Tensor(np.random.randn(prev, self.neurons))
            self.biases = Tensor(np.random.randn(1, self.neurons))

    def __repr__(self) -> str:
        return f"TensorNN.{self.__class__.__name__}(neurons={self.neurons}, activation={self.activation})"
