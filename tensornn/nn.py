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

from typing import Iterable, List, Optional, Sequence, Union, Tuple
import numpy as np
from tqdm import trange

from tensornn.activation import ReLU, Softmax

from .layers import Dense, Layer, flatten
from .tensor import Tensor
from .optimizers import SGD, Optimizer
from .loss import CategoricalCrossEntropy, Loss
from .errors import NotRegisteredError, InputDimError, InitializationError


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
        First layer will be considered the input layer. All layers will be the
        Dense layer with the ReLU activation. The last layer will be Dense with
        the Softmax activation. The network will also be registered with
        CategoricalCrossEntropy loss and the SGD optimizer.

        :param sizes: list of numbers of neurons per layer
        """
        
        if len(sizes) < 3:
            raise InitializationError("NeuralNetwork needs at least 2 layers, excluding input layer")

        layers = []
        layers.append(Dense(sizes[1], num_inputs=sizes[0], activation=ReLU()))
        for size in sizes[2:-1]:
            layers.append(Dense(size, activation=ReLU()))
        layers.append(Dense(sizes[-1], activation=Softmax()))

        net = cls(layers)
        net.register(CategoricalCrossEntropy(), SGD())
        return net

    def add(self, layers: Union[Layer, Iterable[Layer]]) -> None:
        """
        Add another layer(s) to the network. This is the same as initializing
        the network with this layer included.

        :param layer: the layer to be added
        """

        if isinstance(layers, Iterable):
            self.layers.extend(layers)
        else:
            self.layers.append(layers)

    def forward(self, inputs: Tensor) -> Tensor:
        """
        Get the output of the neural network. Automatically flattens the inputs.

        :param inputs: inputs to the network
        :returns: the output of every layer in the network
        """
        inputs = flatten(inputs)
        raw_outputs = []
        activated_outputs = [inputs]

        if len(inputs) != self.layers[0].inputs:
            raise InputDimError(
                "Input shape does not match the number of neurons in the first layer. "
                f"Number of inputs: {len(inputs)}, first layer neurons: {self.layers[0].inputs}"
            )

        for layer in self.layers:
            z, a = layer.forward(activated_outputs[-1])
            raw_outputs.append(z)
            activated_outputs.append(a)

        return raw_outputs, activated_outputs

    def get_loss(self, inputs: Tensor, desired_outputs: Tensor) -> Tensor:
        """
        Calculate the loss for the given data.

        :param inputs: input to the network
        :param desired_outputs: desired output of the network for the given inputs
        :returns: the loss of the network for the given parameters
        """

        if None in (self.loss, self.optimizer):
            raise NotRegisteredError("Network needs to be registered in order to use Network.get_loss")

        pred = self.forward(inputs)[1][-1]
        return self.loss.calculate(pred, desired_outputs)

    def predict(self, inputs: Tensor) -> int:
        """
        Get the prediction of the neural network for the given input. This method should only
        be used on a trained network, because otherwise it will produce useless random values.

        :param inputs: inputs to the network
        :returns: an array which contains the value of each neuron in the last layer
        """
        return self.forward(inputs)[1][-1]

    def register(self, loss: Loss, optimizer: Optimizer) -> None:
        """
        Register the neural network. This method initializes the network with loss and optimizer
        and also finishes up any last touches to its layers.

        :param loss: type of loss this network uses to calculate loss
        :param optimizer: type of optimizer this network uses
        :raises InitializationError: num_inputs not specified to first layer
        """

        self.loss = loss
        self.optimizer = optimizer

        # register all layers
        if self.layers[0].inputs is None:
            raise InitializationError("First layer must be specified with num_inputs")

        curr_neurons = self.layers[0].neurons
        for layer in self.layers[1:]:
            layer.register(curr_neurons)
            curr_neurons = layer.neurons

    def train(
        self,
        inputs: Tensor,
        desired_outputs: Tensor,
        batch_size: int = 32,
        epochs: int = 5,
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
    
    def backpropagation(self):
        """
        Find out how much each weight/bias contributes to the loss. Not meant for external use.
        """

        outputs = self.forward(self.inputs)
    
    def __repr__(self):
        layers = "\n\t" + ",\n\t".join([str(l) for l in self.layers]) + "\n"
        return f"tnn.NeuralNetwork[inputs={self.layers[0].inputs}, layers={len(self.layers)}]({layers})"


"""
# stuff
class NeuralNetwork(object):
    def __init__(self, layers):
        self.layers = layers
        self.weights = []
        self.biases = []
        for i in range(len(layers) - 1):
            self.weights.append(np.random.randn(layers[i + 1], layers[i]))
            self.biases.append(np.random.randn(layers[i + 1], 1))

        # weights:
        # [
        #    [first layer neurons -> l2n1 weights],
        #    [first layer neurons -> l2n2 weights],
        #    ...8 times total
        # ]
        # ex:
        # [
        #    [a1, b1],
        #    [a2, b2],
        #    ...
        #    [a8, b8]
        # ]
        #
        # current:
        # [
        #    [l1n1 -> second layer neurons weights],
        #    [l1n2 -> second layer neurons weights]
        # ]
        # ex:
        # [
        #    [a1, a2, a3, a4, a5, a6, a7, a8],
        #    [b1, b2, b3, b4, b5, b6, b7, b8]
        # ]

        # biases:
        # [
        #    [each second layer neuron has unique bias],
        #    [l2n2 has unique bias],
        #    ...8 times total
        # ]
        # ex
        # [
        #    [b1],
        #    [b2],
        #    ...
        #    [b8]
        # ]
        #
        # current:
        # [
        #    [all biases of second layer]
        # ]
        # ex:
        # [
        #    [b1, b2, b3, b4, b5, b6, b7, b8]
        # ]

    def feedforward(self, x):
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
"""