"""
This file contains the loss functions used in TensorNN. Loss functions are
ways your neural network calculates how off its calculations are. Then this
information is used to improve/train it.
"""

#
#  TensorNN
#  Python machine learning library/framework made from scratch.
#  Copyright Arjun Sahlot 2023
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
    "Loss",
    "CategoricalCrossEntropy",
    "MSE",
    "RMSE",
    "MAE",
    "MSLE",
    "Poisson",
    "SquaredHinge",
    "RSS",
]


class Loss(ABC, TensorNNObject):
    """
    Base loss class. All loss classes should inherit from this.
    """

    @abstractmethod
    def calculate(self, pred: Tensor, desired: Tensor) -> Tensor:
        """
        The loss function is used to calculate how off the predictions of the network are.

        :param pred: the prediction of the network
        :param desired: the desired values which the network should have gotten close to
        :returns: the average of calculated loss for one whole pass of the network
        """

    @abstractmethod
    def derivative(self, pred: Tensor, desired: Tensor) -> Tensor:
        """
        Used in backpropagation which helps calculates how much each neuron impacts the loss.

        :param pred: the prediction of the network
        :param desired: the desired values which the network should have gotten close to
        :returns: the derivative of the loss function wrt the last layer of the network
        """


class CategoricalCrossEntropy(Loss):
    """
    It is recommended to use the Softmax activation function with this loss.
    Despite its long name, the way that categorical cross entropy loss is calculated is simple.

    Let's say our prediction (after softmax) is ``[0.7, 0.2, 0.1]``, and the desired values are
    ``[1, 0, 0]``. We can simply get the prediction number at the index of the ``1`` in the desired
    values. 1 is at index 0 so we look at index 0 of our prediction which would be ``0.7``.
    Now we just take the negative log of ``0.7`` and we are done!

    Note: log in programming is usually ``logₑ`` or ``natural log`` or ``ln`` in math.
    """

    def calculate(self, pred: Tensor, desired: Tensor) -> Tensor:
        pred = pred.clip(1e-15, 1 - 1e-15)  # prevent np.log(0)
        return -np.sum(desired * np.log(pred), axis=1)

    def derivative(self, pred: Tensor, desired: Tensor) -> Tensor:
        return pred - desired

CCE = CategoricalCrossEntropy


class BinaryCrossEntropy(Loss):
    """
    Sigmoid is the only activation function compatible with BinaryCrossEntropy loss.
    This is how it is calculated: ``-(desired*log(pred) + (1-desired)*log(1-pred))``.

    Note: log in programming is usually ``logₑ`` or ``natural log`` or ``ln`` in math
    """

    def calculate(self, pred: Tensor, desired: Tensor) -> Tensor:
        pred = pred.clip(1e-15, 1 - 1e-15)  # prevent np.log(0)
        return -(desired * np.log(pred) + (1 - desired) * np.log(1 - pred))

BCE = BinaryCrossEntropy


class MeanSquaredError(Loss):
    """
    Mean squared error is calculated extremely simply.
    1. Find the difference between the prediction vs. the actual results we should have got
    2. Square these values, because negatives are the same as positives, only magnitude matters
    3. calculate mean

    ex: our predictions: ``[0.1, 0.2, 0.7]``, desired: ``[0, 0, 1]``
    1. pred - actual: ``[0.1, 0.2, -0.3]``
    2. squared: ``[0.01, 0.04, 0.09]``
    3. mean: ``0.04666667``
    """

    def calculate(self, pred: Tensor, desired: Tensor) -> Tensor:
        return np.mean(np.square(pred - desired), axis=1)

    def derivative(self, pred: Tensor, desired: Tensor) -> Tensor:
        return 2 * (pred - desired)

MSE = MeanSquaredError


class RootMeanSquaredError(Loss):
    """
    Root mean squared error is just MSE, but it includes a square root after taking the average.
    """

    def calculate(self, pred: Tensor, desired: Tensor) -> Tensor:
        return np.sqrt(np.mean(np.square(pred - desired), axis=1))

RMSE = RootMeanSquaredError


class MeanAbsoluteError(Loss):
    """
    Mean absolute error is MSE but instead of squaring the values, you absolute value them.
    """

    def calculate(self, pred: Tensor, desired: Tensor) -> Tensor:
        return np.mean(np.abs(pred - desired), axis=1)

MAE = MeanAbsoluteError


class MeanSquaredLogarithmicError(Loss):
    """
    Mean squared logarithmic error is MSE but taking the log of our values before subtraction.
    """

    def calculate(self, pred: Tensor, desired: Tensor) -> Tensor:
        pred[pred == -1] = 1e-15 - 1
        squared_log_error = np.square(np.log(pred + 1) - np.log(desired + 1))
        return np.mean(squared_log_error, axis=1)

MSLE = MeanSquaredLogarithmicError


class Poisson(Loss):
    """
    Poisson loss is calculated with this formula: average of (pred-desired*logₑ(pred))
    """

    def calculate(self, pred: Tensor, desired: Tensor) -> Tensor:
        pred[pred == 0] = 1e-15
        error = pred - desired * np.log(pred)
        return np.mean(error, axis=1)


class SquaredHinge(Loss):
    """
    Square hinge loss is calculated with this formula: max(0, 1-pred*desired)^2
    """

    def calculate(self, pred: Tensor, desired: Tensor) -> Tensor:
        error = 1 - pred * desired
        return np.sum(np.square(np.maximum(0, error)), axis=1)

SHL = SquaredHinge


class ResidualSumOfSquares(Loss):
    """
    Residual sum of squares loss is MSE but instead of doing the mean, you do the sum.
    """

    def calculate(self, pred: Tensor, desired: Tensor) -> Tensor:
        return np.sum(np.square(pred - desired), axis=1)

RSS = ResidualSumOfSquares