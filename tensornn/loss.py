"""
This file contains the loss functions used in TensorNN. Loss functions are
ways your neural network calculates how off its calculations are. Then this
information is used to improve/train it.
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
    "CategoricalCrossEntropy"
]


class Loss:
    def get(self, pred: Tensor, actual: Tensor) -> Tensor:
        return np.mean(self.calculate(pred, actual))

    def calculate(self, pred: Tensor, actual: Tensor) -> Tensor:
        raise NotImplemented("loss", self)


class CategoricalCrossEntropy(Loss):
    def calculate(self, pred, actual):
        clipped = np.clip(pred, 1e-10, 1-1e-10)

        if len(actual.shape) == 2:
            actual = np.argmax(actual, axis=1)

        true_pred = clipped[range(len(pred)), actual]
        return -np.log(true_pred)
