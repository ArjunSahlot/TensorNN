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

import numpy as np
from .tensor import Tensor


__all__ = [
    "Dense",
]


class Layer:
    def __init__(self, inputs: Tensor, neurons: Tensor, zero_biases: bool = True) -> None:
        self.weights: Tensor = np.random.randn(inputs, neurons)
        if zero_biases:
            self.biases: Tensor = np.zeros((1, neurons))
        else:
            self.biases: Tensor = np.random.randn(1, neurons)

    def forward(self, inputs: Tensor):
        raise NotImplementedError(
            f"TensorNN.{self.__class__.__name__} is not currently implemented")


class Dense(Layer):
    def forward(self, inputs):
        return inputs @ self.weights + self.biases
