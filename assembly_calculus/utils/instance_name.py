from traceback import walk_stack
from functools import cached_property

# TODO2: document that this behaviour is only appropriate for use in interpreter
# Response: This can be used in regular executions as well for debugging
from types import FrameType
from typing import Tuple


class RememberInitStack:
    """
    Stores stack on time of init
    """

    def __init__(self):
        self.__init_stack: Tuple[FrameType, ...] = tuple(frame for frame, _ in walk_stack(None))

    @property
    def _init_stack(self) -> Tuple[FrameType, ...]:
        """Stack at time instance init was called"""
        return self.__init_stack


class RememberInitName(RememberInitStack):
    """
    Stores the name of the variable this object was first stored in
    (assuming such existed at time of init)
    """

    @cached_property
    def instance_name(self) -> str:
        """First variable name object was stored in"""
        for frame in reversed(self._init_stack[1:]):
            frame_locals = frame.f_locals

            for var_name, var_value in frame_locals.items():
                if id(var_value) == id(self):
                    return var_name

        return "???"
