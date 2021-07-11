"""
This file contains the activation functions of TensorNN. Activation
functions modify their input to create non-linearity in the network.
This allows your network to handle more complex problems. They are
very similar to a layer.
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

import numpy as np
from .tensor import Tensor
from .errors import NotImplemented


__all__ = [
    "ReLU",
]


class Activation:
    """
    Base activation class. All activation classes should inherit from this.
    """

    def forward(self, inputs: Tensor) -> Tensor:
        """
        Calculate a forwards pass of this activation function.

        :param inputs: the outputs from the previous layer
        :returns: the inputs after they are passed through the activation function
        """

        raise NotImplemented("activation", self)


class ReLU(Activation):
    """
    The rectified linear unit activation function is one of the simplest activation function.

    If ReLU is given a negative value(<0) it will convert it into a 0, otherwise it will keep it the same.
    """

    def forward(self, inputs):
        return np.maximum(0, inputs)


class Softmax(Activation):
    """
    The softmax activation function is most commonly used in the output layer.
    The goal of softmax is to convert the predicted values of the network into percentages that add up to 1.
    Ex. it converts [-1.42, 3.312, 0.192] to [0.00835, 0.94970, 0.41935] which is much easier to understand.

    The way it works is ...
    """

    def forward(self, inputs):
        exp = np.exp(inputs-np.max(inputs, axis=1, keepdims=True))
        return exp/np.sum(exp, axis=1, keepdims=True)
