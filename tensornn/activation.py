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
    "Softmax",
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
    Ex: [12.319, -91.3, 0.132] -> [12.319, 0, 0.132]
    """

    def forward(self, inputs):
        return np.maximum(0, inputs)


class Softmax(Activation):
    """
    The softmax activation function is most commonly used in the output layer.
    The goal of softmax is to convert the predicted values of the network into percentages that add up to 1.
    Ex. it converts [-1.42, 3.312, 0.192] to [0.00835, 0.94970, 0.41935] which is much easier to understand.

    When coming up with a way to write this, a big problem is negative numbers since we can't have negative
    numbers in our final output. So how do we get rid of them? Do we clip them to 0? Do we square them? Do
    we use absolute value? Though all these methods seem nice, they take away from the value of negative
    numbers. If we clip to 0 then negative numbers are no more than just 0, and squaring or using absolute
    value will just result in the opposite of what we want (large negative number turns into large positive
    number). So the most effective way is to use exponentiation. Through exponentiation, negative numbers
    will be small while positive numbers will be large.

    But exponentiation raises a new problem, super large numbers which can cause overflow. Fortunately there
    is a simple solution, we can convert all the values into non positive values prior to exponentiation. We
    can do this by subtracting each value by the maximum value of our output. This way our values before
    exponentiation will range between -inf to 0 and our values after exponentiation will range between 0
    (e^-inf) to 1 (e^0).

    Finally, to come up with all the percentages we can just figure out how much each value contributes to the
    final sum, what fraction of the sum does the value make. So we can do each value divided by the total sum.

    All steps/TLDR:
    Starting values (from previous example): [-1.42, 3.312, 0.192]
    Subtract largest value to make all negative: 3.312 is max so subtract from all values, [-4.732, 0, -3.120]
    Exponentiation, raise each value to e (e^x): [0.0080884, 1, 0.04415717]
    Come up with percentages, divide each number by the sum: sum is 1.05224557 so we divide each value by it,
    [0.00836574, 0.94969828, 0.04193599]

    Although it may look long numpy simplifies it for us a bunch.
    """

    def forward(self, inputs):
        exp = np.exp(inputs-np.max(inputs, axis=1, keepdims=True))
        return exp/np.sum(exp, axis=1, keepdims=True)
