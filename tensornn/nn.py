"""
This file contains the neural network class.
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

from typing import List, Optional, Sequence, Union
import numpy as np

from .layers import Layer
from .tensor import Tensor
from .optimizers import Optimizer
from .loss import Loss
from .errors import RegisteredError, TooFewLayersError


__all__ = [
    "NeuralNetwork",
]


class NeuralNetwork:
    """
    Create your neural network with this class.
    """

    def __init__(self, layers: Optional[Sequence[Layer]] = ()) -> None:
        """
        Initialize the network.

        :param layers: list of layers that make up network
        """

        self.layers: List[Layer] = list(layers)
        self.loss: Optional[Loss] = None
        self.optimizer: Optional[Optimizer] = None

    def add(self, layers: Union[Layer, Sequence[Layer]]) -> None:
        """
        Add another layer to the network. This is the same as initializing
        the network with this layer.

        :param layer: the layer to be added
        :returns: nothing
        """

        if isinstance(layers, Sequence):
            self.layers.extend(layers)
        else:
            self.layers.append(layers)

    def predict(self, inputs: Tensor) -> Tensor:
        """
        Get the output of the neural network. This method should be used on a trained neural
        network otherwise it will produce random useless data.

        :param inputs: inputs to the network, list of batches which contain a list of inputs
        :returns: output of network after being passed through all layers
        """

        for layer in self.layers:
            inputs = layer.forward(inputs)

        return inputs

    def register(self, loss: Loss, optimizer: Optimizer) -> None:
        """
        Register the neural network. This method initializes the network with loss and optimizer
        and also finishes up any last touches to its layers.

        :param loss: type of loss this network uses to calculate loss
        :param optimizer: type of optimizer this network uses
        """

        self.loss = loss
        self.optimizer = optimizer

        if not self.layers:
            raise TooFewLayersError("NeuralNetwork needs at least 1 layer")

        # register all layers
        curr_neurons = self.layers[0].neurons
        for layer in self.layers[1:]:
            layer.register(curr_neurons)
            curr_neurons = layer.neurons

    def train(self, inputs: Tensor, desired_outputs: Tensor) -> None:
        """
        Train the neural network. What training essentially does is adjust the weights and
        biases of the neural network for the inputs to match the desired outputs as close
        as possible.

        :param inputs: training data which is inputted to the network
        :param desired_outputs: these values is what you want the network to output for respective inputs
        :returns: nothing
        """

        if None in (self.loss, self.optimizer):
            raise RegisteredError("NeuralNetwork is not registered")
