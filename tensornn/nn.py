"""
This file contains the neural network class which is essentially
a collections of layers.
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

from typing import List, Optional, Sequence
import numpy as np

from .layers import Layer
from .tensor import Tensor
from .optimizers import Optimizer, Adam
from .loss import Loss, CategoricalCrossEntropy


__all__ = [
    "NeuralNetwork",
]


class NeuralNetwork:
    """
    Create your neural network with this class.
    """

    def __init__(
        self,
        layers: Optional[Sequence[Layer]] = [],
        loss: Optional[Loss] = CategoricalCrossEntropy(),
        optimizer: Optional[Optimizer] = Adam()
    ) -> None:
        """
        Initialize the network

        :param layers: list of layers that make up network
        :param loss: type of loss this network uses to calculate loss
        """

        self.layers: List[Layer] = list(layers)
        self.loss = loss
        self.optimizer = optimizer

    def add(self, layer: Layer) -> None:
        """
        Add another layer to the network. This is the same as initializing
        the network with this layer.

        :param layer: the layer to be added
        :returns: nothing
        """

        self.layers.append(layer)

    def forward(self, inputs: Tensor) -> Tensor:
        """
        Get the output of the neural network.

        :param inputs: inputs to the network, list of batches which contain a list of inputs
        :returns: output of network after being passed through all layers
        """

        for layer in self.layers:
            inputs = layer.forward(inputs)

        return inputs
