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

from abc import ABC, abstractmethod
import warnings

import numpy as np

from .tensor import Tensor
from .utils import TensorNNObject

__all__ = [
    "Activation",
    "NoActivation",
    "ReLU",
    "Softmax",
    "LeakyReLU",
    "ELU",
    "Sigmoid",
    "Swish",
    "NewtonsSerpentine",
    "Tanh",
]


# TODO: add .plot to all activations to show the graph of the function easily
class Activation(ABC, TensorNNObject):
    """
    Base activation class. All activation classes should inherit from this.
    """

    @abstractmethod
    def forward(self, inputs: Tensor) -> Tensor:
        """
        Calculate a forwards pass of this activation function.

        :param inputs: the outputs from the previous layer
        :returns: the inputs after they are passed through the activation function
        """

    @abstractmethod
    def derivative(self, inputs: Tensor) -> Tensor:
        """
        The derivative of the function. Used for backpropagation.

        :param inputs: get the derivative of the function at this input
        :returns: the derivative of the function at the given input
        """


class NoActivation(Activation):
    """
    Linear activation function, doesn't change anything. Use this if you don't want an activation.
    """

    def forward(self, inputs: Tensor) -> Tensor:
        return inputs
    
    def derivative(self, inputs: Tensor) -> Tensor:
        return np.ones(inputs.shape)


class ReLU(Activation):
    """
    The rectified linear unit activation function is one of the simplest activation function. It is
    a piecewise function.
    Formula: ``if x>=0, x; if x<0, 0``

    Ex: ``[12.319, -91.3, 0.132] -> [12.319, 0, 0.132]``
    """

    def forward(self, inputs: Tensor) -> Tensor:
        return Tensor(np.where(inputs > 0, inputs, 0))

    def derivative(self, inputs: Tensor) -> Tensor:
        return Tensor(np.where(inputs > 0, 1, 0))


class LeakyReLU(Activation):
    """
    Leaky ReLU is extremely similar to ReLU. ReLU is LeakyReLU if A was 1.
    Formula: ``if x>=0, x; if x<0, Ax`` | constants: A(leak)

    Ex, A=0.1: ``[12.319, -91.3, 0.132] -> [12.319, -9.13, 0.132]``
    """

    def __init__(self, a: float = 0.1) -> None:
        """
        Initialize LeakyReLU.

        :param a: multiplier used in formula, checkout help(tnn.activation.LeakyReLU), defaults to 1
        """
        # TODO: enforce a should be positive

        self.a = a

    def forward(self, inputs: Tensor) -> Tensor:
        return Tensor(np.where(inputs > 0, inputs, inputs * self.a))

    def derivative(self, inputs: Tensor) -> Tensor:
        return Tensor(np.where(inputs > 0, 1, self.a))

    def __repr__(self) -> str:
        return f"TensorNN.LeakyReLU(a={self.a})"


class ELU(Activation):
    """
    Exponential linear unit is similar to ReLU, but it is not piecewise.
    Formula: ``A*((e^x)-1)`` | constants: A, e(Euler's number, 2.718...)

    Ex, A=1: ``[12.319, -91.3, 0.132] -> [12.319, -1, 0.132]``
    """

    def __init__(self, a: float = 1) -> None:
        """
        Initialize ELU.

        :param a: multiplier used in formula, checkout help(tnn.activation.ELU), defaults to 1
        """

        self.a = a

    def forward(self, inputs: Tensor) -> Tensor:
        return Tensor(np.where(inputs > 0, inputs, self.a * (np.exp(inputs) - 1)))

    def derivative(self, inputs: Tensor) -> Tensor:
        return Tensor(np.where(inputs > 0, 1, self.a * np.exp(inputs)))

    def __repr__(self) -> str:
        return f"TensorNN.ELU(a={self.a})"


class Softmax(Activation):
    """
    The softmax activation function is most commonly used in the output layer. If you are using this activation
    function, you should be using tnn.CategoricalCrossEntropy as your loss function. This is because the softmax
    function always generates a probability distribution with all values between 0 and 1, and for these types
    of values, tnn.CategoricalCrossEntropy is the best loss function to use.

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
    final sum, what fraction of the sum does each value make. So we can do each value divided by the total sum.

    All steps/TLDR:
    Starting values (from previous example): [-1.42, 3.312, 0.192]
    Subtract largest value to make all negative: 3.312 is max so subtract from all values, [-4.732, 0, -3.120]
    Exponentiation, raise each value to e (e^x): [0.0080884, 1, 0.04415717]
    Come up with percentages, divide each number by the sum: sum is 1.05224557 so we divide each value by it,
    [0.00836574, 0.94969828, 0.04193599]
    """

    def forward(self, inputs: Tensor) -> Tensor:
        exp = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        return Tensor(exp / np.sum(exp, axis=1, keepdims=True))

    def derivative(self, inputs: Tensor) -> Tensor:
        raise NotImplementedError
        # TODO: implement
        return


class Sigmoid(Activation):
    """
    The sigmoid function's output is always between -1 and 1
    Formula: ``1 / (1+e^(-x))`` | constants: e(Euler's number, 2.718...)

    Ex: ``[12.319, -91.3, 0.132] -> [9.99995534e-01, 2.23312895e-40, 5.32952167e-01]``
    """

    def forward(self, inputs: Tensor) -> Tensor:
        return Tensor(1 / (1 + np.exp(-inputs)))

    def derivative(self, inputs: Tensor) -> Tensor:
        sigmoid = self.forward(inputs)
        return Tensor(sigmoid * (1 - sigmoid))


class Swish(Activation):
    """
    The swish activation function is the output of the sigmoid function multiplied by x.
    Formula: ``x / (1+e^(-x))`` | constants: e(Euler's number, 2.718...)

    Ex: ``[12.319, -91.3, 0.132] -> [1.23189450e+01, -2.03884673e-38, 7.03496861e-02]``
    """

    def forward(self, inputs: Tensor) -> Tensor:
        return Tensor(inputs / (1 + np.exp(-inputs)))

    def derivative(self, inputs: Tensor) -> Tensor:
        swish = self.forward(inputs)
        sigmoid = swish / inputs
        return Tensor(swish + sigmoid * (1 - swish))


class NewtonsSerpentine(Activation):
    """
    Haven't seen it anywhere so I am not sure if this is good but seemed like a good candidate.
    NOTE: THIS IS NOT A GOOD CANDIDATE. Larger numbers result in a lower value, which means being
    large doesn't give importance. Do not use unless you want to have some fun ;)

    Formula: ``(A*B*x)/(x^2+A^2)`` | A, B constants

    Ex, A=1,B=1: ``[12.319, -91.3, 0.132] -> [0.08064402, -0.01095159, 0.12973942]``

    https://mathworld.wolfram.com/SerpentineCurve.html
    """

    def __init__(self, a: float = 1, b: float = 1) -> None:
        """
        Initialize Newton's Serpentine. Checkout the formula by using:
        help(tensornn.activation.NewtonsSerpentine)

        :param a: constant in equation, defaults to 1
        :param b: constant in equation, defaults to 1
        """

        # If a is 0
        if not a:
            warnings.warn("Parameter a is 0. This could result in a dead network.")

        # If b is 0
        if not b:
            warnings.warn("Parameter b is 0. This could result in a dead network.")

        self.a = a
        self.b = b

    def forward(self, inputs: Tensor) -> Tensor:
        return Tensor((self.a * self.b * inputs) / (np.square(inputs) + np.square(self.a)))

    def derivative(self, inputs: Tensor) -> Tensor:
        sq_x = np.square(inputs)
        sq_a = np.square(self.a)
        return Tensor((self.a * self.b * (sq_a - sq_x)) / np.square(sq_x + sq_a))

    def __repr__(self) -> str:
        return f"TensorNN.NewtonsSerpentine(a={self.a}, b={self.b})"


class Tanh(Activation):
    """
    TODO: add description
    """

    def forward(self, inputs: Tensor) -> Tensor:
        return Tensor(np.tanh(inputs))

    def derivative(self, inputs: Tensor) -> Tensor:
        tanh = self.forward(inputs)
        return Tensor(1 - np.square(tanh))
