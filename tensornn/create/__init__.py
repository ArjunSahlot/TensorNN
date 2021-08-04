"""
This submodule includes base classes with guidelines for anyone who would like to create their own
layer/activation/loss/optimizer. It could be used to learn how to create parts of a working library,
as well as add any missing parts that you would like to have as part of this library.
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

from ..layers import BaseLayer
from ..loss import BaseLoss
from ..activation import BaseActivation
from ..optimizers import BaseOptimizer


__all__ = [
    "Layer",
    "Loss",
    "Activation",
    "Optimizer"
]


class Layer(BaseLayer, ABC):
    """
    Abstract base layer class. All custom layer classes should inherit from this.

    A neural network is composed of layers. A set of inputs are moved from one layer to another.
    Each layer has its own way of calculating the output its own inputs(outputs of previous layer).
    Each layer must also need a few tweakable parameters, tweaking this parameters will allow the
    network to learn and adapt the inputs to the correct outputs.

    Checkout the current __init__ by using tnn.source(tnn.create.Layer.__init__). Also checkout
    the guidelines of some of the methods to see how you should modify them, help(tnn.create.Layer.forward).
    """

    def forward(self, inputs):
        """
        This method adds the activation's output(if it exists) then passes that to the _forward method.
        """

    @abstractmethod
    def _forward(self, inputs):
        """
        This method receives the output of the activation(if it exists). The output of this method should
        be the output of this layer from the given inputs. For example, a dense layer would connect all
        inputs to outputs using weights and biases, but maybe your layer does something else.
        """


class Loss(BaseLoss, ABC):
    """
    Base loss class. All custom loss classes should inherit from this.
    """

    def calculate(self, pred, desired):
        """
        """

    @abstractmethod
    def _calculate(self, pred, desired):
        """
        """


class Activation(BaseActivation, ABC):
    """
    Base activation class. All custom activation classes should inherit from this.
    """

    @abstractmethod
    def forward(self, inputs):
        """
        """


class Optimizer(BaseOptimizer, ABC):
    """
    Base optimizer class. All custom optimizer classes should inherit from this.
    """
