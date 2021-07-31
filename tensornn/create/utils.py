"""
This file contains the guideline objects required in the TensorNN create submodule.
The way you use these is -

Create the base class and add guidelines to certain methods:

class Animal:
    def __init__(self, home, age):
        self.home, self.age = home, age

    @guideline_method("Move via changing your home point.")
    def move(self, new_home):
        self.home = new_home



While the user is creating a subclass of your class, they can use the guidelines:

class Dog(Human):
    def move(new_home):
        # Hmmm what should I put here, let me check the guideline:
        print(Animal.move.guideline)
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

import inspect


class GuidelineMethod:
    def __init__(self, guideline, method):
        self.__guideline = guideline
        self.method = method
        self.srcfile = inspect.getsourcefile(method)
        self.src = inspect.getsource(method)

    @property
    def guideline(self):
        return self.__guideline

    def __call__(self, *args, **kwargs):
        return self.method(*args, **kwargs)


def guideline_method(guideline):
    def wrapper(method):
        return GuidelineMethod(guideline, method)

    return wrapper
