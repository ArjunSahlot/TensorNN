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
    """
    Base loss class. All loss classes should inherit from this.
    """

    def calculate(self, pred: Tensor, desired: Tensor) -> Tensor:
        """
        The mean of all the loss values in this batch.

        :param pred: the prediction of the network
        :param desired: the desired values which the network should have gotten close to
        :returns: the average of calculated loss for one whole pass of the network across all batches
        """

        return np.mean(self._calculate(pred, desired))

    def _calculate(self, pred: Tensor, desired: Tensor) -> Tensor:
        """
        Calculates the loss for one whole pass of the network.

        :param pred: the prediction of the network
        :param desired: the desired values which the network should have gotten close to
        :returns: the calculated loss for one whole pass of the network
        """

        raise NotImplemented("loss", self)


class CategoricalCrossEntropy(Loss):
    """
    Despite its long name, the way the categorical cross entropy loss is calculated is simple.

    Let's say our prediction (after softmax) is [0.7, 0.2, 0.1], and the desired values are
    [1, 0, 0]. We can simply get the prediction number at the index of the 1 in the desired
    values. 1 is at index 0 so we look at index 0 of our prediction which would be 0.7.
    Now we just take the negative log of 0.7 and we are done!

    Note: log in programming is usually log base e or natural log or ln in math
    """

    def _calculate(self, pred, desired):
        clipped = np.clip(pred, 1e-10, 1-1e-10)

        # If desired is an array of one hot encoded vectors
        if len(desired.shape) == 2:
            desired = np.argmax(desired, axis=1)

        true_pred = clipped[range(len(pred)), desired]
        return -np.log(true_pred)
