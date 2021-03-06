"""
This file contains errors the TensorNN might raise.
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


__all__ = ["NotRegisteredError", "TooFewLayersError", "InputDimError"]


class NotRegisteredError(Exception):
    """
    Raised when you try to train your NeuralNetwork before registering it.
    """


class TooFewLayersError(Exception):
    """
    Raised when your NeuralNetwork has less than 1 layer.
    """


class InputDimError(Exception):
    """
    Raised when number of dimensions of inputs is not at least 2.
    """
