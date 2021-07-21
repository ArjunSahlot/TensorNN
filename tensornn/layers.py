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

from typing import Optional
import numpy as np

from .tensor import Tensor
from .activation import Activation
from .errors import NotImplemented


__all__ = [
    "Dense",
]


class Layer:
    """
    Base layer class. All layer classes should inherit from this.
    """

    biases: Tensor

    def __init__(
        self,
        num_inputs: Tensor,
        num_neurons: Tensor,
        activation: Optional[Activation] = None,
        zero_biases: bool = True
    ) -> None:
        """
        Initialize the layer.

        :param num_inputs: the number of neurons/outputs in the previous layer
        :param num_neurons: the number of neurons in this layer/number of outputs of this layer
        :param activation: the activation function applied before the layer output is calculated
        :param zero_biases: whether or not the biases should be initialized to 0, if your network dies try setting this to False
        """

        self.weights: Tensor = np.random.randn(num_inputs, num_neurons)
        if zero_biases:
            self.biases: Tensor = np.zeros((1, num_neurons))
        else:
            self.biases: Tensor = np.random.randn(1, num_neurons)

        self.activation: Optional[Activation] = activation

    def forward(self, inputs: Tensor) -> Tensor:
        """
        Calculate a forwards pass of this layer with activation.

        :param inputs: outputs from the previous layer
        :returns: the output calculated after this layer and activation
        """

        if self.activation is not None:
            activated = self.activation.forward(inputs)
        else:
            activated = inputs

        return self._forward(activated)

    def _forward(self, inputs: Tensor) -> Tensor:
        """
        Calculate a forwards pass of this layer, without activation.

        :param inputs: outputs from the previous layer
        :returns: the output calculated after this layer
        """

        raise NotImplemented("layers", self)


class Dense(Layer):
    """
    A dense layer. Each neuron is connected to all neurons in the previous layer.
    Output is calculated by: (output of previous layer * weights) + biases.
    """

    def _forward(self, inputs):
        # @ is __matmul__ from python3.5+, https://www.python.org/dev/peps/pep-0465/
        return inputs @ self.weights + self.biases
