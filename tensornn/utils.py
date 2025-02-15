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
from functools import wraps

import numpy as np

from .tensor import Tensor


__all__ = ["source", "one_hot", "normalize", "atleast_2d", "set_seed", "flatten"]


def source(obj: Any, output: Optional[TextIO] = sys.stdout) -> str:
    """
    Get the source code of a TensorNN object.

    :param obj: the tensornn object, ex: tnn.nn.NeuralNetwork
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
    [0, 1](first neuron on, second off), that would be the 'one-hot vector'.
    If you were to run np.argmax([0, 1]), you would get the index of the 1(which is also
    the index of the max value).

    :param values: to be converted to one-hot (max 1D), ex: one_hot(3, 5) -> [0, 0, 0, 1, 0]
    :param classes: number of different places for the 1, len of one-hot
    :returns: one-hot vector from the given params
    """

    arr = []

    if isinstance(values, int):
        arr = np.zeros(classes, dtype=int)
        arr[values] = 1
    else:
        if len(np.array(values).shape) > 1:
            raise ValueError("Values for one-hot exceed 2 dimensions")
        arr = np.zeros((len(values), classes), dtype=int)
        arr[np.arange(len(values)), values] = 1
    return arr


def takes_one_hot(pos: int = 2):
    """
    Apply this decorator to a function that takes in a one-hot vector. Used for loss functions.

    :param pos: position of the argument to convert to one-hot vector, default 2 for loss functions
    :returns: decorator
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if len(args[pos].shape) == 1:
                args = list(args)
                args[pos] = one_hot(args[pos], args[pos-1].shape[1])
            return func(*args, **kwargs)

        return wrapper
    return decorator


# I don't like this. I initially did it this way to provide flexibility to the user.
# But, I think this is instead just overcomplicated and not helpful. Instead, let's make
# it simple so the user understands what's going on and can easily adjust the data themselves.
# TODO: improve this
def takes_single_value(pos: int = 1):
    """
    Apply this decorator to a function that takes in a single value. Used for loss functions.

    :param pos: position of the argument to convert to single value, default 1 for loss functions
    :returns: decorator
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if len(args[pos].shape) > 1:
                args = list(args)
                args[pos] = np.argmax(args[pos], axis=1)
            return func(*args, **kwargs)

        return wrapper
    return decorator


def normalize(data):
    """
    Normalize the training data. This will never hurt your data, it will always help it, make sure
    to use this every time. This function will make it so that the largest value
    in the data is 1.

    :param data: training data to the network
    :returns: normalized data, max is 1
    """

    return data / np.max(data)


def set_seed(seed):
    """
    Set the seed for TensorNN's randomness. Useful for reproducibility while testing and debugging.

    :param seed: the seed to set
    """

    np.random.seed(seed)


def flatten(inputs: Tensor) -> Tensor:
    """
    Flatten the inputs array. For example, a neural network cannot take in an image
    as input since it is 2D, so we can flatten it to make it 1D.
    """
    # return Tensor(inputs.reshape(inputs.shape[0], np.prod(inputs.shape[1:])))
    return Tensor(inputs.reshape(np.prod(inputs.shape)))


atleast_2d = np.atleast_2d


class TensorNNObject:
    """
    Base class for all TensorNN objects. This class is used to provide a common interface for all
    TensorNN objects and to provide a base __repr__ method for all TensorNN objects.
    """

    def source(self, output: Optional[TextIO] = sys.stdout) -> str:
        return source(self, output)

    def __repr__(self):
        return f"TensorNN.{self.__class__.__name__}"