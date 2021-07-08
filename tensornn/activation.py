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
    def forward(self, inputs: Tensor) -> Tensor:
        raise NotImplemented("activation", self)


class ReLU(Activation):
    def forward(self, inputs):
        return np.maximum(0, inputs)


class Softmax(Activation):
    def forward(self, inputs):
        exp = np.exp(inputs-np.max(inputs, axis=1, keepdims=True))
        return exp/np.sum(exp, axis=1, keepdims=True)
