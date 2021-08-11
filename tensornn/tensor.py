"""
This file contains the Tensor class for TensorNN. Essentially, it is
just a class which does matrix operations for N-dimensional arrays.
Currently we extend off of numpy's ndarray class since it is efficient
and has a bunch of useful operations. If needed, we can add additional
functionality to our extended class such as new methods.
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


__all__ = [
    "Tensor",
]


class Tensor(np.ndarray):
    """
    The tensor class. Currently functions like numpy.ndarray but with a custom print.
    """

    def __new__(cls, input_array=()):
        return np.asarray(input_array).view(cls)


# Add TensorNN.Tensor() around the output of np.ndarray
Tensor.tmp_str = Tensor.__str__
Tensor.__str__ = lambda self: "TensorNN.Tensor(" + ("\n    " if len(self.shape) > 1 else "") + ("    ".join(
    self.tmp_str().splitlines(True)) if len(self.shape) > 1 else self.tmp_str()) + ("\n" if len(self.shape) > 1 else "") + ")"
