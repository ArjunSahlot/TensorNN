"""
This file contains helpful debugging utilities for TensorNN.
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

from enum import IntEnum
from pathlib import Path

__all__ = [
    "VERBOSE_LEVEL",
    "Levels",
    "set_debug_file",
]


class Levels(IntEnum):
    # Level 0:
    # Only errors
    ERROR = 0

    # Level 1:
    # Warning messages, such as dead networks, deprecated functions, etc.
    WARNING = 1

    # Level 2:
    # Additional information on top of warnings, think of it as QOL
    # messages such as loading updates, etc.
    INFO = 2

    # Level 3:
    # Think of this level as extra info that you don't need but is convenient and
    # interesting to have around. Stuff like summaries
    SUMMARY = 3

    # Level 4:
    # Even more debugging information. Outputs things like mean of weights/gradients
    # while training and redirects it to tnn.debug.DEBUG_FILE.
    DEBUG = 4


VERBOSE_LEVEL = Levels.INFO

DEBUG_FILE = Path("tnn_debug.log")

def set_debug_file(file: str) -> None:
    global DEBUG_FILE
    DEBUG_FILE = Path(file)
