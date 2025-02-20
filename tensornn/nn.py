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
from tqdm import tqdm, trange
import numpy as np

from .layers import Dense, Layer
from .tensor import Tensor
from .optimizers import SGD, Optimizer
from .loss import MSE, Loss, CategoricalCrossEntropy
from .errors import NotRegisteredError, InputDimError, InitializationError
from .utils import TensorNNObject
from .activation import ReLU, NoActivation, Softmax, Sigmoid
import tensornn.debug as debug


__all__ = [
    "NeuralNetwork",
]


# TODO: DO REPRS/STRS FOR ALL OBJECTS


class NeuralNetwork(TensorNNObject):
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
    def simple(cls, sizes: Sequence[int], learning_rate: float = 0.001):
        """
        Create a NeuralNetwork from the number of neurons per layer.
        First layer will be considered the input layer. All layers will be the
        Dense layer with the ReLU activation. The last layer will be Dense with
        the Softmax activation. The network will also be registered with
        MSE loss and the SGD optimizer.

        :param sizes: list of numbers of neurons per layer
        :param learning_rate: the learning rate of the network
        """
        
        if len(sizes) < 3:
            raise InitializationError("NeuralNetwork needs at least 2 layers")

        layers = []
        for size in sizes[0:-1]:
            layers.append(Dense(size, activation=ReLU()))
        layers.append(Dense(sizes[-1], activation=Sigmoid()))

        net = cls(layers)
        net.register(MSE(), SGD(learning_rate))
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
        Propagate the inputs through the network and return the last layer's output.

        :param inputs: inputs to the network
        :returns: the output of the last layer in the network
        """

        if None in (self.loss, self.optimizer):
            raise NotRegisteredError("NeuralNetwork is not registered")

        if inputs.ndim != 2:
            flag = "run tnn.atleast_2d on" if inputs.ndim < 2 else "tnn.flatten"
            raise InputDimError(
                f"Input shape must be 2 dimensional. Perhaps you need to {flag} the input?\n"
                f"Input shape dimensions: {inputs.ndim}, expected: 2"
            )

        if inputs.shape[1] != self.layers[0].neurons:
            raise InputDimError(
                "Input shape does not match the number of neurons in the first layer.\n"
                f"Number of inputs: {inputs.shape[1]}, input layer neurons: {self.layers[0].neurons}"
            )

        for layer in self.layers:
            _, inputs = layer.forward(inputs)

        return inputs

    def backward(self, loss_deriv: Tensor) -> None:
        """
        Find out how much each weight/bias contributes to the loss and gets stored in each layer.
        Not meant for external use.
        
        :param loss_deriv: the derivative of the loss wrt the output of the last layer
        :returns: nothing
        """
        for layer in reversed(self.layers):
            loss_deriv = layer.backward(loss_deriv)
    
    def reset_gradients(self) -> None:
        """
        Reset the gradients of all the layers in the network.
        """
        for layer in self.layers:
            layer.reset_gradients()

    def train(
        self,
        inputs: Tensor,
        desired_outputs: Tensor,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        epochs: int = 5,
    ) -> None:
        """
        Train the neural network. What training essentially does is adjust the weights and
        biases of the neural network for the inputs to match the desired outputs as close
        as possible.

        :param inputs: training data which is inputted to the network
        :param desired_outputs: these values is what you want the network to output for respective inputs
        :param epochs: how many iterations will your network will run to learn
        :param verbose: the level of verbosity of the program (1-3), defaults to 1
        :raises NotRegisteredError: network not registered
        :raises InputDimError: inputs not at least 2d
        """

        # network is only registered when both of these values are assigned
        if None in (self.loss, self.optimizer):
            raise NotRegisteredError("NeuralNetwork is not registered")

        if inputs.ndim < 2:
            raise InputDimError(
                "Received inputs less than 2D. Need to be at least 2D. "
                "Perhaps use tnn.atleast_2d to make them 2D?"
            )
        
        if len(inputs) != len(desired_outputs):
            raise InputDimError(
                "Length of inputs and desired_outputs do not match, each input doesn't have a matching desired output. "
                f"Inputs length: {len(inputs)}, desired_outputs length: {len(desired_outputs)}. "
            )

        # number of training inputs have matching desired outputs
        training_inp_size = len(inputs)
        batches = np.ceil(training_inp_size / batch_size)
        minibatches = list(zip(np.array_split(inputs, batches), np.array_split(desired_outputs, batches)))
        if debug.VERBOSE_LEVEL >= debug.Levels.SUMMARY:
            print(
                f"Training data length: {training_inp_size}, "
                f"inps: {len(inputs)},  desired_outs: {len(desired_outputs)}\n"
                f"Total {len(minibatches)} minibatches, each with {len(minibatches[0][0])} inputs"
            )
        
        for epoch in range(epochs):
            if debug.VERBOSE_LEVEL >= debug.Levels.INFO:
                pbar = trange(len(minibatches), desc=f"Epoch {epoch+1}/{epochs}", unit="batches")
            else:
                pbar = range(len(minibatches))
            for i in pbar:
                in_batch, out_batch = minibatches[i]

                # self.reset_gradients()

                # sets up all the inputs for the layers
                pred = self.forward(in_batch)

                # display loss
                loss = self.loss.calculate(pred, out_batch)
                
                if debug.VERBOSE_LEVEL >= debug.Levels.INFO:
                    pbar.set_postfix({"Loss": np.mean(loss)})

                loss_deriv = self.loss.derivative(pred, out_batch)
                self.backward(loss_deriv)

                output = ""
                for layer in self.layers[1:]:
                    layer.step(learning_rate)

                    if debug.VERBOSE_LEVEL >= debug.Levels.DEBUG:
                        layer_out = str(layer).ljust(70) + "\t"
                        weights = f"weights mean: {np.round(np.mean(layer.weights), 5)}".ljust(30) + "\t"
                        biases = f"biases mean: {np.round(np.mean(layer.biases), 5)}".ljust(30) + "\t"
                        grad_weights = f"grad_weights mean: {np.round(np.mean(layer.grad_weights), 5)}".ljust(30) + "\t"
                        grad_biases = f"grad_biases mean: {np.round(np.mean(layer.grad_biases), 5)}".ljust(30)

                        output += layer_out + weights + biases + grad_weights + grad_biases + "\n"

                if debug.VERBOSE_LEVEL >= debug.Levels.DEBUG:
                    if not debug.DEBUG_FILE.exists():
                        debug.DEBUG_FILE.touch()
                    print(output, file=open(debug.DEBUG_FILE, "a"))
                    output = ""


    def get_loss(self, inputs: Tensor, desired_outputs: Tensor) -> Tensor:
        """
        Calculate the loss for the given data.

        :param inputs: input to the network
        :param desired_outputs: desired output of the network for the given inputs
        :returns: the loss of the network for the given parameters
        """

        if None in (self.loss, self.optimizer):
            raise NotRegisteredError("Network needs to be registered in order to use Network.get_loss")

        pred = self.forward(inputs)
        return self.loss.calculate(pred, desired_outputs)

    def predict(self, inputs: Tensor) -> int:
        """
        Get the prediction of the neural network. This will return the index of the most highly
        activated neuron. This method should only be used on a trained network, because otherwise
        it will produce useless random values.

        :param inputs: inputs to the network
        :returns: an array which contains the value of each neuron in the last layer
        """
        return np.argmax(self.forward(inputs), axis=1)

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
        curr_neurons = None
        for layer in self.layers:
            layer.register(curr_neurons)
            curr_neurons = layer.neurons

    def __repr__(self):
        return f"TensorNN.{self.__class__.__name__}(\n" + \
                f"\tlayers=[\n\t\t" + \
                f",\n\t\t".join(map(str, self.layers)) + \
                f"\n\t],\n" + \
                f"\tloss={self.loss},\n" + \
                f"\toptimizer={self.optimizer}\n" + \
                f")"

    def __add__(self, other: Union[Layer, Iterable[Layer]]) -> "NeuralNetwork":
        """
        Add another layer(s) to the network. This is the same as initializing
        the network with this layer included.

        :param other: the layer(s) to be added
        :returns: the network with the layer(s) added
        """
        self.add(other)
        return self
