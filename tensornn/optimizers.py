"""
This file contains optimizers that help tune your neural network.
Optimizers enable us to improve our neural network efficiently.
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

import numpy as np

from .tensor import Tensor
from .utils import TensorNNObject

__all__ = [
    "Optimizer",
    "SGD",
]


class Optimizer(ABC, TensorNNObject):
    """ """


class SGD(Optimizer):
    """
    Stochastic gradient descent optimizer. But, this is actually mini-batch stochastic
    gradient descent. You can make it standard SGD by setting batch_size to 1 in training.
    """

    def __init__(self, learning_rate: float = 0.01) -> None:
        """
        Initialize the optimizer.

        :param learning_rate: the learning rate of the optimizer
        """
        self.learning_rate = learning_rate
