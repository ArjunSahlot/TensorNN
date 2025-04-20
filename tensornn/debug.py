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

from pathlib import Path
from typing import Literal

__all__ = [
    "debug",
]


class Property:
    def __init__(self, value: bool = True, parent: 'Property' = None, name: str = "root"):
        self._value = value
        self._properties: dict[str, Property] = {}
        self._parent = parent
        self._special_value = None
        self._is_special = False
        self._name = name
    
    def enable(self) -> 'Property':
        self._value = True
        self._is_special = False
        return self
    
    def disable(self) -> 'Property':
        self._value = False
        self._is_special = False
        return self
    
    def __getattr__(self, name: str) -> 'Property':
        if name.startswith("_"):
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
        
        if name not in self._properties:
            self._properties[name] = Property(self._value, parent=self, name=name)
        
        return self._properties[name]
    
    def __setattr__(self, name: str, value: any) -> None:
        if name.startswith("_"):
            super().__setattr__(name, value)
            return
        
        if hasattr(self, "_properties"):
            if name not in self._properties:
                self._properties[name] = Property(True, parent=self, name=name)
                
            if isinstance(value, bool):
                self._properties[name]._value = value
                self._properties[name]._is_special = False
            else:
                self._properties[name]._special_value = value
                self._properties[name]._is_special = True
        else:
            super().__setattr__(name, value)
    
    def __bool__(self) -> bool:
        if self._parent is not None and not bool(self._parent):
            return False
        return self._value
    
    def __repr__(self) -> str:
        return self._format_tree()
    
    def _format_tree(self, indent: int = 0, is_last: bool = True, prefix: str = "") -> str:
        result = []
        marker = "└── " if is_last else "├── "
        
        if self._name != "root":
            value_str = self._get_value_str()
            result.append(f"{prefix}{marker}{self._name}: {value_str}")

            if is_last:
                new_prefix = prefix + "    "
            else:
                new_prefix = prefix + "│   "
        else:
            new_prefix = ""
        
        sorted_props = sorted(self._properties.items())
        
        for i, (key, prop) in enumerate(sorted_props):
            is_last_child = (i == len(sorted_props) - 1)
            result.append(prop._format_tree(indent + 4, is_last_child, new_prefix))
        
        return "\n".join(result)
    
    def _get_value_str(self) -> str:
        if self._parent is not None and not bool(self._parent):
            if self._is_special:
                return f"{self._special_value} (inactive - parent disabled)"
            else:
                return f"{self._value} (inactive - parent disabled)"
        
        if self._is_special:
            return f"{self._special_value}"
        return f"{self._value}"
    
    def get_value(self):
        if self._parent is not None and not bool(self._parent):
            return False
            
        if self._is_special:
            return self._special_value
        return self._value
    
    def from_dict(self, config: dict[str, any]) -> None:
        for key, value in config.items():
            if isinstance(value, dict):
                if key not in self._properties:
                    self._properties[key] = Property(True, parent=self, name=key)
                self._properties[key].from_dict(value)
            else:
                setattr(self, key, value)


class DebugInfoHints:
    progress: bool
    """
    Display progress bars through tqdm when applicable.
    default: True
    """

    tips: bool
    """
    Display helpful tips and hints for debugging and improving the network.
    default: True
    """

    summary: bool
    """
    Display a summaries of long tasks
    default: True
    """

class DebugWarningHints:
    deprecated: bool
    """
    Display warnings for deprecated features and functions.
    default: True
    """

    network: bool
    """
    Display warnings for network related issues.
    default: True
    """


class DebugFileHints:
    weights: bool
    """
    Display weights in the debug file.
    default: True
    """

    biases: bool
    """
    Display biases in the debug file.
    default: True
    """

    gradients: bool
    """
    Display gradients in the debug file.
    default: True
    """

    reduction: Literal["mean", "sum"]
    """
    Outputting all the data to the file can be a lot of data. This parameter
    specifies how to reduce the data. If 'mean', it will output the mean of the data.
    If 'sum', it will output the sum of the data.
    default: "mean"
    """

    update: Literal["epoch", "batch"]
    """
    When to update the debug file. If 'epoch', it will update the file at the end of each epoch.
    If 'batch', it will update the file at the end of each batch.
    default: "epoch"
    """

    name: str
    """
    The name of the debug file.
    default: "tnn_debug.log"
    """

class DebugHints:
    info: DebugInfoHints
    """
    Output helpful information for debugging and improving the network.
    default: ON
    """

    warning: DebugWarningHints
    """
    Output warnings for deprecated features and functions, and network related issues.
    default: ON
    """

    file: DebugFileHints
    """
    Output data to a debug file.
    default: OFF
    """
    
    def from_dict(self, config): ...


debug: DebugHints = Property()
debug.from_dict({
    "info": {
        "progress": True,
        "tips": True,
        "summary": True,
    },
    "warning": {
        "deprecated": True,
        "network": True,
    },
    "file": {
        "name": "tnn_debug.log",
        "weights": True,
        "biases": True,
        "gradients": True,
        "reduction": "mean",
        "update": "epoch",
    }
})

debug.file.disable()
