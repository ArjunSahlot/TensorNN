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
from .utils import one_hot


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


class Loss(ABC):
    """
    Base loss class. All loss classes should inherit from this.

    vec_type is the type of format the loss function would prefer getting the input,
    either 1("one-hot") or 2("int").
    For example, MSE would prefer getting one-hot and CategoricalCrossEntropy would
    like int
    """

    vec_type: int

    def calculate(self, pred: Tensor, desired: Tensor) -> Tensor:
        """
        The mean of all the loss values in this batch.

        :param pred: the prediction of the network
        :param desired: the desired values which the network should have gotten close to
        :returns: the average of calculated loss for one whole pass of the network
        """

        # If wanted one-hot and not one-hot format
        if self.vec_type == 1 and desired.ndim == 2:
            desired = one_hot(desired, pred.shape[-1])
        # Otherwise if wanted int and one-hot format
        elif self.vec_type == 2 and desired.ndim == 1:
            desired = np.argmax(desired, axis=int(len(desired.shape) == 2))

        return np.mean(self._pre(pred, desired))

    @abstractmethod
    def _pre(self, pred: Tensor, desired: Tensor) -> Tensor:
        """
        Calculates the loss for one whole pass of the network. ``_pre`` runs before calculating
        the mean and it is also required to be overridden

        :param pred: the prediction of the network
        :param desired: the desired values which the network should have gotten close to, one-hot
        :returns: the calculated loss for one whole pass of the network
        """


class CategoricalCrossEntropy(Loss):
    """
    It is recommended to use the Softmax activation function with this loss.
    Despite its long name, the way the categorical cross entropy loss is calculated is simple.

    Let's say our prediction (after softmax) is ``[0.7, 0.2, 0.1]``, and the desired values are
    ``[1, 0, 0]``. We can simply get the prediction number at the index of the ``1`` in the desired
    values. 1 is at index 0 so we look at index 0 of our prediction which would be ``0.7``.
    Now we just take the negative log of ``0.7`` and we are done!

    Note: log in programming is usually ``logₑ`` or ``natural log`` or ``ln`` in math.
    """

    vec_type = 2

    def _pre(self, pred: Tensor, desired: Tensor) -> Tensor:
        pred = pred.clip(1e-15, 1 - 1e-15)  # prevent np.log(0)
        true_pred = pred[..., desired]
        return -np.log(true_pred)


class BinaryCrossEntropy(Loss):
    """
    Sigmoid is the only activation function compatible with BinaryCrossEntropy loss.
    This is how it is calculated: ``-(desired*log(pred) + (1-desired)*log(1-pred))``.

    Note: log in programming is usually ``logₑ`` or ``natural log`` or ``ln`` in math
    """

    vec_type = 1

    def _pre(self, pred: Tensor, desired: Tensor) -> Tensor:
        pred = pred.clip(1e-15, 1 - 1e-15)  # prevent np.log(0)
        return -(desired * np.log(pred) + (1 - desired) * np.log(1 - pred))


class MSE(Loss):
    """
    Mean squared error is calculated extremely simply.
    1. Find the difference between the prediction vs. the actual results we should have got
    2. Square these values, because negatives are the same as positives, only magnitude matters
    3. calculate means

    ex: our predictions: ``[[0.1, 0.2, 0.7], [0.2, 0.3, 0.5]]``, desired: ``[[0, 0, 1], [0, 0, 1]]``
    1. pred - actual: ``[[0.1, 0.2, -0.3], [0.2, 0.3, -0.5]]``
    2. squared: ``[[0.01, 0.04, 0.09], [0.04, 0.09, 0.25]]``
    3. means: ``[0.04666667, 0.12666667]``
    """

    vec_type = 1

    def _pre(self, pred: Tensor, desired: Tensor) -> Tensor:
        squared_error = np.square(pred - desired)
        return np.mean(squared_error, axis=int(len(squared_error.shape) != 1))


class RMSE(Loss):
    """
    Root mean squared error is just MSE, but it includes a square root after taking the average.
    """

    vec_type = 1

    def _pre(self, pred: Tensor, desired: Tensor) -> Tensor:
        squared_error = np.square(pred - desired)
        return np.sqrt(np.mean(squared_error, axis=int(len(squared_error.shape) != 1)))


class MAE(Loss):
    """
    Mean absolute error is MSE but instead of squaring the values, you absolute value them.
    """

    vec_type = 1

    def _pre(self, pred: Tensor, desired: Tensor) -> Tensor:
        abs_error = np.abs(pred - desired)
        return np.mean(abs_error, axis=int(len(abs_error.shape) != 1))


class MSLE(Loss):
    """
    Mean squared logarithmic error is MSE but taking the log of our values before subtraction.
    """

    vec_type = 1

    def _pre(self, pred: Tensor, desired: Tensor) -> Tensor:
        pred[pred == -1] = 1e-15 - 1
        squared_log_error = np.square(np.log(pred + 1) - np.log(desired + 1))
        return np.mean(squared_log_error, axis=int(len(squared_log_error.shape) != 1))


class Poisson(Loss):
    """
    Possion loss is calculated with this formula: average of (pred-desired*logₑpred)
    """

    vec_type = 1

    def _pre(self, pred: Tensor, desired: Tensor) -> Tensor:
        pred[pred == 0] = 1e-15
        error = pred - desired * np.log(pred)
        return np.mean(error, axis=int(len(error.shape) != 1))


class SquaredHinge(Loss):
    """
    Square hinge loss is calculated with this formula: max(0, 1-pred*desired)^2
    """

    vec_type = 1

    def _pre(self, pred: Tensor, desired: Tensor) -> Tensor:
        error = 1 - pred * desired
        return np.sum(np.square(np.maximum(0, error)), axis=int(len(error.shape) != 1))


class RSS(Loss):
    """
    Residual sum of squares loss is MSE but instead of doing the mean, you do the sum.
    """

    vec_type = 1

    def _pre(self, pred: Tensor, desired: Tensor) -> Tensor:
        squared_error = np.square(pred - desired)
        return np.sum(squared_error, axis=int(len(squared_error.shape) != 1))
