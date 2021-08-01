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

from typing import Optional
import numpy as np

from .utils import guideline_method
from ..layers import BaseLayer
from ..loss import BaseLoss
from ..activation import BaseActivation
from ..optimizers import BaseOptimizer
from ..tensor import Tensor
from ..errors import NotImplemented


__all__ = [
    "Layer",
    "Loss",
    "Activation",
    "Optimizer"
]


class Layer(BaseLayer):
    """
    Base layer class. All custom layer classes should inherit from this.
    """

    recommended_methods_to_change = [
        "forward",
        "_forward"
    ]
    guideline = """
A neural network is composed of layers. A set of inputs are moved from one layer to another.
Each layer has its own way of calculating the output its own inputs(outputs of previous layer).
Each layer must also need a few tweakable parameters, tweaking this parameters will allow the
network to learn and adapt the inputs to the correct outputs.

Checkout the current __init__ by using tnn.source(tnn.create.Layer.__init__). Also checkout
the recommended methods for this class(Layer.recommended_methods_to_change) as well as their
respective guidelines(tnn.create.Layer.forward.guideline).
"""

    def __init__(
        self,
        num_inputs: int,
        num_neurons: int,
        activation: Optional[BaseActivation] = None,
        zero_biases: bool = True
    ) -> None:
        """
        Initialize the layer.

        :param num_inputs: the number of neurons/outputs in the previous layer
        :param num_neurons: the number of neurons in this layer/number of outputs of this layer
        :param activation: the activation function applied before the layer output is calculated
        :param zero_biases: whether or not the biases should be initialized to 0, if your network dies try setting this to False
        """

        self.weights: Tensor = Tensor(np.random.randn(num_inputs, num_neurons))
        if zero_biases:
            self.biases: Tensor = Tensor(np.zeros((1, num_neurons)))
        else:
            self.biases: Tensor = Tensor(np.random.randn(1, num_neurons))

        self.activation: Optional[Activation] = activation

    @guideline_method("""""")
    def forward(self, inputs: Tensor) -> Tensor:
        """
        Calculate a forwards pass of this layer with activation.

        :param inputs: outputs from the previous layer
        :returns: the output calculated after this layer and activation
        """

        if self.activation is not None:
            activated = self.activation.forward(inputs)
        else:
            activated = inputs

        return self._forward(activated)

    @guideline_method("""""")
    def _forward(self, inputs: Tensor) -> Tensor:
        """
        Calculate a forwards pass of this layer, without activation.

        :param inputs: outputs from the previous layer
        :returns: the output calculated after this layer
        """

        raise NotImplemented("create", self)


class Loss(BaseLoss):
    """
    Base loss class. All custom loss classes should inherit from this.
    """

    recommended_methods_to_change = [
        "_calculate"
    ]
    guideline = """"""

    def __init__(self) -> None:
        pass

    @guideline_method("""""")
    def calculate(self, pred: Tensor, desired: Tensor) -> Tensor:
        """
        The mean of all the loss values in this batch.

        :param pred: the prediction of the network
        :param desired: the desired values which the network should have gotten close to
        :returns: the average of calculated loss for one whole pass of the network across all batches
        """

        return np.mean(self._calculate(pred, desired))

    @guideline_method("""""")
    def _calculate(self, pred: Tensor, desired: Tensor) -> Tensor:
        """
        Calculates the loss for one whole pass of the network.

        :param pred: the prediction of the network
        :param desired: the desired values which the network should have gotten close to
        :returns: the calculated loss for one whole pass of the network
        """


class Activation(BaseActivation):
    """
    Base activation class. All custom activation classes should inherit from this.
    """

    recommended_methods_to_change = [
        "forward"
    ]
    guideline = """"""

    def __init__(self) -> None:
        pass

    @guideline_method("""""")
    def forward(self, inputs: Tensor) -> Tensor:
        """
        Calculate a forwards pass of this activation function.

        :param inputs: the outputs from the previous layer
        :returns: the inputs after they are passed through the activation function
        """


class Optimizer(BaseOptimizer):
    """
    Base optimizer class. All custom optimizer classes should inherit from this.
    """

    recommended_methods_to_change = []
    guideline = """"""

    def __init__(self) -> None:
        pass
