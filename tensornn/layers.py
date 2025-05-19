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
from typing import Optional, Tuple

import numpy as np

from .tensor import Tensor
from .activation import Activation, NoActivation
from .utils import atleast_2d, TensorNNObject


__all__ = [
    "Layer",
    "Dense",
    "Input",
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

        self.weights: Optional[Tensor] = None
        self.biases: Optional[Tensor] = None
        self.grad_weights: Tensor = Tensor(0)
        self.grad_biases: Tensor = Tensor(0)

    @abstractmethod
    def forward(self, inputs: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Calculates and returns a forward pass of this layer, before and after activation.

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

    @abstractmethod
    def backward(self, accumulated_gradient: Tensor) -> Tensor:
        """
        Backpropagation step. This is called when the loss is calculated and the gradients are
        propagated back to this layer.

        :param accumulated_gradient: the gradient from the next layer
        :returns: the gradient for the previous layer
        """

    @abstractmethod
    def step(self, adjust_w: Tensor, adjust_b: Tensor) -> None:
        """
        Update the weights and biases of this layer.

        :param adjust_w: the adjustment to be made to the weights
        :param adjust_b: the adjustment to be made to the biases
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
        parameter_init: tuple[str, str] = ("default", "default"),
        # zero_biases: bool = True,
    ) -> None:
        """
        Initialize dense layer.

        :param num_neurons: the number of neurons in this layer/number of outputs of this layer
        :param activation: the activation function applied before the layer output is calculated
        :param parameter_init: the initialization method for the weights and biases.
        #TODO have stuff like zero_biases go in a dictionary like config or options
        """
        super().__init__(num_neurons, activation)
        
        # Used for backpropagation
        self.inputs: Tensor
        self.calculated: Tensor
        self.grad_weights: Tensor
        self.grad_biases: Tensor
        self.w_init: str = parameter_init[0]
        self.b_init: str = parameter_init[1]
    
    def forward(self, inputs: Tensor) -> Tuple[Tensor, Tensor]:
        self.inputs = inputs
        values = inputs @ self.weights + self.biases
        self.calculated = values
        return values, self.activation.forward(values)
    
    def backward(self, accumulated_gradient: Tensor) -> Tensor:
        dc_da = accumulated_gradient
        da_dz = self.activation.derivative(self.calculated)

        batch_size = self.inputs.shape[0]

        dc_dz = dc_da * da_dz

        self.grad_weights = self.inputs.T @ dc_dz
        self.grad_biases = dc_dz.sum(axis=0)

        return dc_dz @ self.weights.T

    def step(self, adjust_w: Tensor, adjust_b: Tensor) -> None:
        self.weights += adjust_w
        self.biases += adjust_b

    def register(self, prev: int) -> None:
        match self.w_init.lower():
            case "xavier_uniform" | "glorot_uniform" | "xavier" | "glorot":
                limit = np.sqrt(6. / (prev + self.neurons))
                self.weights = Tensor(np.random.uniform(-limit, limit, (prev, self.neurons)))
            case "glorot_normal" | "xavier_normal":
                std_dev = np.sqrt(2. / (prev + self.neurons))
                self.weights = Tensor(np.random.randn(prev, self.neurons) * std_dev)
            case "lecun_uniform" | "lecun":
                limit = np.sqrt(3. / prev)
                self.weights = Tensor(np.random.uniform(-limit, limit, (prev, self.neurons)))
            case "lecun_normal":
                std_dev = np.sqrt(1. / prev)
                self.weights = Tensor(np.random.randn(prev, self.neurons) * std_dev)
            case "he_normal" | "he" | "default":
                std_dev = np.sqrt(2. / prev)
                self.weights = Tensor(np.random.randn(prev, self.neurons) * std_dev)
            case "he_uniform":
                limit = np.sqrt(6. / prev)
                self.weights = Tensor(np.random.uniform(-limit, limit, (prev, self.neurons)))
            case "uniform":
                limit = np.sqrt(6. / prev)
                self.weights = Tensor(np.random.uniform(-limit, limit, (prev, self.neurons)))
            case "normal":
                std_dev = np.sqrt(2. / prev)
                self.weights = Tensor(np.random.randn(prev, self.neurons) * std_dev)
            case "random":
                self.weights = Tensor(np.random.randn(prev, self.neurons))
            case "zero":
                self.weights = Tensor(np.zeros((prev, self.neurons)))
            case "ones":
                self.weights = Tensor(np.ones((prev, self.neurons)))
            case _:
                raise ValueError(f"Unknown weight initialization method: {self.w_init}")
        
        match self.b_init.lower():
            case "zeros" | "default":
                self.biases = Tensor(np.zeros((1, self.neurons)))
            case "ones":
                self.biases = Tensor(np.ones((1, self.neurons)))
            case "random":
                self.biases = Tensor(np.random.randn(1, self.neurons))
            case _:
                raise ValueError(f"Unknown bias initialization method: {self.b_init}")

    def __repr__(self) -> str:
        return f"TensorNN.{self.__class__.__name__}(neurons={self.neurons}, activation={self.activation})"


class Input(Layer):
    """
    Input layer. This is a dummy layer that just passes the input to the next layer.
    This layer should be the first layer in every network to describe the input shape.
    It is possible to give this layer an activation function if you want an activation
    applied to the inputs every time the network is run. It could be useful in situations
    where you want to normalize the inputs before passing them to the next layer.
    """

    def __init__(self, num_inputs: int, activation: Activation = NoActivation()) -> None:
        """
        Initialize input layer.

        :param num_inputs: the number of inputs to this layer/number of outputs of this layer
        :param activation: the activation function applied before the layer output is calculated
        """
        super().__init__(num_inputs, activation)

    def forward(self, inputs: Tensor) -> Tuple[Tensor, Tensor]:
        return inputs, self.activation.forward(inputs)
    
    def register(self, prev: int) -> None:
        pass

    def backward(self, accumulated_gradient: Tensor) -> Tensor:
        return accumulated_gradient
    
    def step(self, adjust_w: Tensor, adjust_b: Tensor) -> None:
        pass