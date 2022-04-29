"""
This file contains the neural network class.
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

from typing import Iterable, List, Optional, Sequence, Union
import numpy as np
from tqdm import trange

from tensornn.activation import ReLU, Softmax

from .layers import Dense, Flatten, Layer
from .tensor import Tensor
from .optimizers import SGD, Optimizer
from .loss import CategoricalCrossEntropy, Loss
from .errors import NotRegisteredError, TooFewLayersError, InputDimError


__all__ = [
    "NeuralNetwork",
]


# TODO: DO REPRS/STRS FOR ALL OBJECTS


class NeuralNetwork:
    """
    Create your neural network with this class.
    """

    def __init__(self, layers: Iterable[Layer] = ()) -> None:
        """
        Initialize the network.

        :param layers: list of layers that make up network
        """

        self.layers: List[Layer] = list(layers)
        self.loss: Optional[Loss] = None
        self.optimizer: Optional[Optimizer] = None

    @classmethod
    def simple(cls, sizes: Sequence[int]):
        """
        Create a NeuralNetwork from the number of neurons per layer.
        First layer will be Flatten and all hidden layers will be Dense
        with ReLU activation. The last layer will be Dense with the Softmax
        activation. The network will also be registered with
        CategoricalCrossEntropy loss and the SGD optimizer.

        :param sizes: list of numbers of neurons per layer
        """

        if len(sizes) < 2:
            raise TooFewLayersError("NeuralNetwork needs at least 2 layers")

        layers = []
        layers.append(Flatten(sizes[0]))

        for size in sizes[1:-1]:
            layers.append(Dense(size, activation=ReLU()))

        layers.append(Dense(sizes[-1], Softmax()))

        net = cls(layers)
        net.register(CategoricalCrossEntropy(), SGD())
        return net

    def add(self, layers: Union[Layer, Iterable[Layer]]) -> None:
        """
        Add another layer(s) to the network. This is the same as initializing
        the network with this layer.

        :param layer: the layer to be added
        """

        if isinstance(layers, Iterable):
            self.layers.extend(layers)
        else:
            self.layers.append(layers)

    def forward(self, inputs: Tensor) -> Tensor:
        """
        Get the output of the neural network.

        :param inputs: inputs to the network
        :returns: output of network after being passed through all layers
        """

        for layer in self.layers:
            inputs = layer.forward(inputs)

        return inputs

    def predict(self, inputs: Tensor) -> int:
        """
        Get the prediction of the neural network for the given input. This method should only
        be used on a trained network, because otherwise it will produce useless random values.

        :param inputs: inputs to the network
        :returns: the index of the neuron with the highest activation
        """

        return int(np.argmax(self.forward(inputs)))

    def register(self, loss: Loss, optimizer: Optimizer) -> None:
        """
        Register the neural network. This method initializes the network with loss and optimizer
        and also finishes up any last touches to its layers.

        :param loss: type of loss this network uses to calculate loss
        :param optimizer: type of optimizer this network uses
        """

        self.loss = loss
        self.optimizer = optimizer

        if len(self.layers) < 2:
            raise TooFewLayersError("NeuralNetwork needs at least 2 layers")

        # register all layers
        curr_neurons = self.layers[0].neurons
        for layer in self.layers[1:]:
            layer.register(curr_neurons)
            curr_neurons = layer.neurons

    def train(
        self,
        inputs: Tensor,
        desired_outputs: Tensor,
        epochs: int,
        verbose: bool = True,
        debug: bool = False,
    ) -> None:
        """
        Train the neural network. What training essentially does is adjust the weights and
        biases of the neural network for the inputs to match the desired outputs as close
        as possible.

        :param inputs: training data which is inputted to the network
        :param desired_outputs: these values is what you want the network to output for respective inputs
        :param epochs: how many iterations will your network will run to learn
        :param verbose: print current progress of the neural network, defaults to True
        :param debug: whether or not to output some important variables, defaults to False
        :raises NotRegisteredError: network not registered
        :raises InputDimError: inputs not at least 2d
        """

        # network is only registered when both of these values are assigned
        if None in (self.loss, self.optimizer):
            raise NotRegisteredError("NeuralNetwork is not registered")

        if inputs.ndim < 2:
            raise InputDimError(
                "Received inputs less than 2D. Need to be at least 2D. "
                "Use numpy.atleast_2d to make them 2D. "
                "tnn.atleast_2d also works if you would prefer not to import numpy."
            )

        # number of training inputs have matching desired outputs
        training_inp_size = min(len(inputs), len(desired_outputs))
        if debug:
            print(
                f"Training data length: {training_inp_size}, "
                f"inps: {len(inputs)},  desired_outs: {len(desired_outputs)}"
            )

        for epoch in range(epochs):
            for i in trange(training_inp_size, desc=f"Epoch {epoch}", unit="inputs"):
                inp = inputs[i]
                desired = desired_outputs[i]
                self._train_single(inp, desired)

    def _train_single(self, inp: Tensor, desired: Tensor) -> None:
        """
        Not meant for external use. Only used within the NeuralNetwork class.

        :param inp: the input to the network
        :param desired: the output you want associated with the given input
        """

        pass


# stuff
class NeuralNetwork(object):
    def __init__(self, layers=[2, 10, 1], activations=["sigmoid", "sigmoid"]):
        assert len(layers) == len(activations) + 1
        self.layers = layers
        self.activations = activations
        self.weights = []
        self.biases = []
        for i in range(len(layers) - 1):
            self.weights.append(np.random.randn(layers[i + 1], layers[i]))
            self.biases.append(np.random.randn(layers[i + 1], 1))

    def feedforward(self, x):
        # return the feedforward value for x
        a = np.copy(x)
        z_s = []
        a_s = [a]
        for i in range(len(self.weights)):
            activation_function = self.getActivationFunction(self.activations[i])
            z_s.append(self.weights[i].dot(a) + self.biases[i])
            a = activation_function(z_s[-1])
            a_s.append(a)
        return (z_s, a_s)

    def backpropagation(self, y, z_s, a_s):
        dw = []  # dC/dW
        db = []  # dC/dB
        deltas = [None] * len(
            self.weights
        )  # delta = dC/dZ  known as error for each layer
        # insert the last layer error
        deltas[-1] = (y - a_s[-1]) * (
            self.getDerivitiveActivationFunction(self.activations[-1])
        )(z_s[-1])
        # Perform BackPropagation
        for i in reversed(range(len(deltas) - 1)):
            deltas[i] = self.weights[i + 1].T.dot(deltas[i + 1]) * (
                self.getDerivitiveActivationFunction(self.activations[i])(z_s[i])
            )
        # a= [print(d.shape) for d in deltas]
        batch_size = y.shape[1]
        db = [d.dot(np.ones((batch_size, 1))) / float(batch_size) for d in deltas]
        dw = [d.dot(a_s[i].T) / float(batch_size) for i, d in enumerate(deltas)]
        # return the derivitives respect to weight matrix and biases
        return dw, db

    def train(self, x, y, batch_size=10, epochs=100, lr=0.01):
        # update weights and biases based on the output
        for e in range(epochs):
            i = 0
            while i < len(y):
                x_batch = x[i : i + batch_size]
                y_batch = y[i : i + batch_size]
                i = i + batch_size
                z_s, a_s = self.feedforward(x_batch)
                dw, db = self.backpropagation(y_batch, z_s, a_s)
                self.weights = [
                    w + lr * dweight for w, dweight in zip(self.weights, dw)
                ]
                self.biases = [w + lr * dbias for w, dbias in zip(self.biases, db)]
                print("loss = {}".format(np.linalg.norm(a_s[-1] - y_batch)))

    @staticmethod
    def getActivationFunction(name):
        if name == "sigmoid":
            return lambda x: np.exp(x) / (1 + np.exp(x))
        elif name == "linear":
            return lambda x: x
        elif name == "relu":

            def relu(x):
                y = np.copy(x)
                y[y < 0] = 0
                return y

            return relu
        else:
            print("Unknown activation function. linear is used")
            return lambda x: x

    @staticmethod
    def getDerivitiveActivationFunction(name):
        if name == "sigmoid":
            sig = lambda x: np.exp(x) / (1 + np.exp(x))
            return lambda x: sig(x) * (1 - sig(x))
        elif name == "linear":
            return lambda x: 1
        elif name == "relu":

            def relu_diff(x):
                y = np.copy(x)
                y[y >= 0] = 1
                y[y < 0] = 0
                return y

            return relu_diff
        else:
            print("Unknown activation function. linear is used")
            return lambda x: 1


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    nn = NeuralNetwork([1, 100, 1], activations=["sigmoid", "sigmoid"])
    X = 2 * np.pi * np.random.rand(1000).reshape(1, -1)
    y = np.sin(X)

    nn.train(X, y, epochs=10000, batch_size=64, lr=0.1)
    _, a_s = nn.feedforward(X)
    # print(y, X)
    plt.scatter(X.flatten(), y.flatten())
    plt.scatter(X.flatten(), a_s[-1].flatten())
    plt.show()
