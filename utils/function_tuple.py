from __future__ import annotations
from typing import TYPE_CHECKING, Set, Dict
if TYPE_CHECKING:
    from assembly_calculus.brain import Brain, Area


class FunctionTuple:

    def __init__(self, read_methods=None):
        self.functions = read_methods

    # TODO: Maybe change to args, kwargs?
    def __call__(self, area: Area, *, brain: Brain):
        return [function(area, brain=brain) for function in self.functions]

    def append(self, functions):
        self.functions += functions

    def register_function(self, function):
        self.functions.append(function)
        return function

    def index(self, function):
        return self.functions.find(function)
