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
from pathlib import Path
from itertools import count
import pickle

from tqdm import tqdm, trange
import numpy as np

from .layers import Dense, Layer, Input
from .tensor import Tensor
from .optimizers import SGD, Optimizer
from .loss import MSE, Loss, CategoricalCrossEntropy
from .errors import NotRegisteredError, InputDimError, InitializationError
from .utils import TensorNNObject
from .activation import ReLU, NoActivation, Softmax, Sigmoid
from .debug import debug


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
        
        if len(sizes) < 2:
            raise InitializationError("NeuralNetwork needs at least 2 layers")

        layers = [Input(sizes[0])]
        for size in sizes[1:-1]:
            layers.append(Dense(size, activation=ReLU()))
        layers.append(Dense(sizes[-1]))

        net = cls(layers)
        net.register(MSE(), SGD(learning_rate))
        return net
    
    @classmethod
    def load(cls, path: str) -> 'NeuralNetwork':
        """
        Load a network from a file using the pickle module.

        :param path: path to load the network from
        :returns: the loaded network
        """
        with open(path, "rb") as f:
            return pickle.load(f)

    def save(self, path: str) -> None:
        """
        Save the network to a file using the pickle module.

        :param path: path to save the network to
        """
        with open(path, "wb") as f:
            pickle.dump(self, f)

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
        
        self.optimizer.reset()
    
    def get_minibatches(
        self,
        inputs: Tensor,
        desired_outputs: Tensor,
        batch_size: int,
        shuffle: bool = False
    ) -> List[Tuple[Tensor, Tensor]]:
        """
        Get the minibatches of the inputs and desired outputs.
        This is used to split the inputs and desired outputs into smaller batches for training.
        
        :param inputs: the inputs to the network
        :param desired_outputs: the desired outputs of the network
        :param batch_size: the size of each batch
        :param shuffle: whether to shuffle the inputs and desired outputs before splitting into batches
        :returns: a list of tuples containing the inputs and desired outputs for each batch
        """
        if shuffle:
            indices = np.arange(len(inputs))
            np.random.shuffle(indices)
            inputs = inputs[indices]
            desired_outputs = desired_outputs[indices]

        minibatches = []
        for i in range(0, len(inputs), batch_size):
            minibatches.append((inputs[i:i + batch_size], desired_outputs[i:i + batch_size]))

        return minibatches

    def train(
        self,
        inputs: Tensor,
        desired_outputs: Tensor,
        epochs: int = 5,
        batch_size: int = 32,
        learning_rate: Optional[float] = None,
        **kwargs
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
        
        if learning_rate is not None:
            self.optimizer.set_lr(learning_rate)

        # number of training inputs have matching desired outputs
        if debug.info.summary:
            print(
                f"Training data length: {len(inputs)}\n"
                f"Total {np.ceil(len(inputs)/batch_size)} minibatches, each with {batch_size} inputs"
            )

        if debug.file:
            file = Path(debug.file.name.get_value())
            if not file.exists():
                file.touch()
            else:
                file.unlink()
                file.touch()
        
        plot_data = {
            "loss": [],
        }

        epoch_range = count(0) if epochs == float("inf") else range(epochs)
        
        for epoch in epoch_range:
            try:
                minibatches = self.get_minibatches(inputs, desired_outputs, batch_size, shuffle=True)
                update_iter = (epoch+1) % kwargs.get("progress_update_every", 1) == 0
                progress_bar = debug.info.progress and update_iter
                if progress_bar:
                    description = f"Epoch {epoch+1}/{epochs}" if epochs != float("inf") else f"Epoch {epoch+1}"
                    pbar = trange(len(minibatches), desc=description, unit="batches")
                    # print(f"\n{self.forward(inputs)[:4]}\n{desired_outputs[:4]}")
                    # print(self.layers[1].weights)
                    # print(self.layers[1].biases)
                    # print(self.layers[2].weights)
                    # print(self.layers[2].biases)
                    # print(self.loss.calculate(self.forward(inputs), desired_outputs))
                else:
                    pbar = range(len(minibatches))

                num_output_features = self.layers[-1].neurons
                acc_outs = np.empty((0, num_output_features))
                acc_preds = np.empty((0, num_output_features))
                for i in pbar:
                    in_batch, out_batch = minibatches[i]

                    # sets up all the inputs for the layers
                    pred = self.forward(in_batch)

                    acc_outs = np.concatenate((acc_outs, out_batch), axis=0)
                    acc_preds = np.concatenate((acc_preds, pred), axis=0)

                    # display loss
                    if progress_bar:
                        loss = self.loss.calculate(acc_preds, acc_outs)
                        pbar.set_postfix({"Loss": np.mean(loss)})

                    loss_deriv = self.loss.derivative(pred, out_batch)
                    self.backward(loss_deriv)

                    self.optimizer.step()

                    if debug.file and debug.file.update.get_value() == "batch" and update_iter:
                        self.update_log(epoch)
                    
                    if kwargs.get("plot_data", False) and kwargs["update"] == "batch" and update_iter:
                        plot_data["loss"].append(loss)

                if debug.file and debug.file.update.get_value() == "epoch" and update_iter:
                    self.update_log(epoch)
                
                if kwargs.get("plot_data", False) and kwargs["update"] == "epoch" and update_iter:
                    plot_data["loss"].append(loss)

            except KeyboardInterrupt:
                print("Training interrupted by user")
                if epochs == float("inf"):
                    break
                else:
                    raise KeyboardInterrupt

        return plot_data if kwargs.get("plot_data", False) else None

    def update_log(self, epoch) -> None:
        """
        Update the log file with the current weights, biases, and gradients of the network.
        """

        output = f"Epoch {epoch+1}:\n"

        for layer in self.layers[1:]:
            layer_out = str(layer).ljust(70) + "\t"
            weights = f"weights mean: {np.round(np.mean(layer.weights), 5)}".ljust(30) + "\t"
            biases = f"biases mean: {np.round(np.mean(layer.biases), 5)}".ljust(30) + "\t"
            grad_weights = f"grad_weights mean: {np.round(np.mean(layer.grad_weights), 5)}".ljust(30) + "\t"
            grad_biases = f"grad_biases mean: {np.round(np.mean(layer.grad_biases), 5)}".ljust(30)

            output += layer_out + \
                        (weights if debug.file.weights else "") + \
                        (biases if debug.file.biases else "") + \
                        (grad_weights if debug.file.weights and debug.file.gradients else "") + \
                        (grad_biases if debug.file.biases and debug.file.gradients else "") + "\n"

        with Path(debug.file.name.get_value()).open("a") as f:
            f.write(output + "\n")

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

        if len(self.layers) < 2:
            raise InitializationError("NeuralNetwork needs at least 2 layers")

        self.loss = loss
        if isinstance(loss, CategoricalCrossEntropy) != isinstance(self.layers[-1].activation, Softmax):
            raise InitializationError(
                "Both CategoricalCrossEntropy and Softmax need to be used together. " +
                (
                    "Use Softmax as the activation function of the last layer "
                    if not isinstance(self.layers[-1].activation, Softmax) else
                    "Use CategoricalCrossEntropy as the loss function "
                ) +
                "to resolve this issue."
            )

        self.optimizer = optimizer
        self.optimizer.register(self)

        # register all layers
        curr_neurons = 0
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

    def __add__(self, other: Union[Layer, Iterable[Layer]]) -> 'NeuralNetwork':
        """
        Add another layer(s) to the network. This is the same as initializing
        the network with this layer included.

        :param other: the layer(s) to be added
        :returns: the network with the layer(s) added
        """
        self.add(other)
        return self
