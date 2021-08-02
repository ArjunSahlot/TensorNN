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

from .create.utils import GuidelineMethod


__all__ = [
    "source",
]


def source(obj, output=sys.stdout):
    """
    Get the source code of a TensorNN object.
    """
    import inspect

    try:
        if isinstance(obj, GuidelineMethod):
            rv = f"In file: {obj.srcfile}\n\n{obj.src}"
        else:
            rv = f"In file: {inspect.getsourcefile(obj)}\n\n{inspect.getsource(obj)}"
    except Exception:
        rv = f"No source available for this object"

    if output is not None:
        print(rv, file=output)

    return rv
