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

__all__ = [
    "source",
]


def source(obj, output=None):
    """
    Get the source code of a TensorNN object.
    """
    import inspect

    try:
        rv = f"In file: {inspect.getsourcefile(obj)}\n\n{inspect.getsource(obj)}"
    except Exception:
        rv = f"No source available for {obj.__name__}"
    finally:
        if output is None:
            return rv

        excep = False
        try:
            print(rv, file=output)
        except Exception:
            excep = True

        if excep:
            raise FileNotFoundError(f"{output} is not a valid output file.")
