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

from abc import ABC, abstractmethod

import numpy as np

from .tensor import Tensor


__all__ = [
    "Loss",
    "CategoricalCrossEntropy",
]


class Loss(ABC):
    """
    Base loss class. All loss classes should inherit from this.
    """

    def calculate(self, pred: Tensor, desired: Tensor) -> Tensor:
        """
        The mean of all the loss values in this batch.

        :param pred: the prediction of the network
        :param desired: the desired values which the network should have gotten close to
        :returns: the average of calculated loss for one whole pass of the network
        """

        return self._post(np.mean(self._pre(pred, desired)))

    @abstractmethod
    def _pre(self, pred: Tensor, desired: Tensor) -> Tensor:
        """
        Calculates the loss for one whole pass of the network. `_pre` runs before calculating
        the mean and it is also required to be overridden

        :param pred: the prediction of the network
        :param desired: the desired values which the network should have gotten close to
        :returns: the calculated loss for one whole pass of the network
        """

    def _post(self, mean: Tensor) -> Tensor:
        """
        Calculated after the mean has been calculated. This is required for losses like RMSE.

        :param mean: the mean already calculated
        :returns: loss calculated
        """

        return mean


class CategoricalCrossEntropy(Loss):
    """
    Despite its long name, the way the categorical cross entropy loss is calculated is simple.

    Let's say our prediction (after softmax) is `[0.7, 0.2, 0.1]`, and the desired values are
    `[1, 0, 0]`. We can simply get the prediction number at the index of the `1` in the desired
    values. 1 is at index 0 so we look at index 0 of our prediction which would be `0.7`.
    Now we just take the negative log of `0.7` and we are done!

    Note: log in programming is usually `logₑ` or natural `log` or `ln` in math
    """

    def _pre(self, pred: Tensor, desired: Tensor) -> Tensor:
        clipped = np.clip(pred, 1e-15, 1-1e-15)

        # If desired is an array of one hot encoded vectors
        if len(desired.shape) == 2:
            desired = np.argmax(desired, axis=1)

        true_pred = clipped[range(len(pred)), desired]
        return -np.log(true_pred)


class MSE(Loss):
    """
    Mean squared error is calculated extremely simply.
    1. Find the difference between the prediction vs. the actual results we should have got
    2. Square these values, because negatives are the same as positives, only magnitude matters
    3. Sum up all the values
    4. calculate mean, done in base class (`tnn.loss.Loss`)

    ex: our predictions: `[[[0.1, 0.2, 0.7], [0.2, 0.3, 0.5]]]`, desired: `[[[0, 0, 1], [0, 0, 1]]]`
    1. pred - actual: `[[[0.1, 0.2, -0.3]`, `[0.2, 0.3, -0.5]]]`
    2. squared: `[[0.01, 0.04, 0.09]`, `[0.04, 0.09, 0.25]]`
    3. sums: `[0.14, 0.48]`
    4. mean: `0.26`
    """

    def _pre(self, pred: Tensor, desired: Tensor) -> Tensor:
        return np.sum(np.square(pred - desired), axis=1)


class RMSE(MSE):
    """
    Root mean squared error is just MSE, but it includes a square root after taking the average.
    """

    def _post(self, mean: Tensor) -> Tensor:
        return np.sqrt(mean)
