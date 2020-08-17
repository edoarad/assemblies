from traceback import walk_stack
from functools import cached_property


class RememberInitStack:
    def __init__(self):
        self.__init_stack = tuple(frame for frame, _ in walk_stack(None))

    @property
    def _init_stack(self):
        return self.__init_stack


class RememberInitName(RememberInitStack):
    @cached_property
    def instance_name(self):
        for frame in reversed(self._init_stack[1:]):
            frame_locals = frame.f_locals

            for var_name, var_value in frame_locals.items():
                if id(var_value) == id(self):
                    return var_name

        return "???"
