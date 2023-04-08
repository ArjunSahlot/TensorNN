"""
This file contains useful variables that are used in TensorNN.
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

import sys
import inspect
from typing import Any, Iterable, Optional, TextIO, Union

import numpy as np

from .tensor import Tensor


__all__ = ["source", "one_hot", "normalize", "atleast_2d"]


def source(obj: Any, output: Optional[TextIO] = sys.stdout) -> str:
    """
    Get the source code of a TensorNN object.

    :param obj: the tensornn object, ex: tnn.nn.NeuralNetwork
    :param output: file to output to
    :returns: the source code of the given object
    """

    try:
        rv = f"In file: {inspect.getsourcefile(obj)}\n\n{inspect.getsource(obj)}"
    except TypeError:
        rv = f"No source available for this object"

    if output is not None:
        print(rv, file=output)

    return rv


def one_hot(values: Union[int, Iterable[int]], classes: int) -> Tensor:
    """
    Get the one-hot representation of an integer. One-hot representation is like
    the opposite of np.argmax. Let's we want our network's output to be
    [0, 1](second neuron lit up, other neuron not), that would be the 'one-hot vector'.
    If you were to run np.argmax([0, 1]), you would get the index of the 1(which is also
    the index of the max value).

    :param values: to be converted to one-hot, ex one_hot(3, 5) -> [0, 0, 0, 1, 0]
    :param classes: number of different places for the 1, len of one-hot
    :returns: one-hot vector from the given params
    """

    if isinstance(values, Iterable):
        return Tensor([one_hot(i, classes) for i in values])
    return Tensor([1 if i == values else 0 for i in range(classes)])


def normalize(data):
    """
    Normalize the training data. This will never hurt your data, it will always help it, make sure
    to use this every time. This function will make it so that the largest value
    in the data is 1.

    :param data: training data to the network
    :returns: normalized data, max is 1
    """

    return data / np.max(data)


# def partial_derivative(func, *args):
    
atleast_2d = np.atleast_2d
