from __future__ import annotations
from abc import abstractmethod, ABC
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .assembly import Assembly
    from ..brain import Brain


class Reader(ABC):

    @staticmethod
    @abstractmethod
    def read(assembly: Assembly, preserve_brain: bool = False, *, brain: Brain):
        pass

    @staticmethod
    def update_hook(assembly: Assembly, *, brain: Brain):
        pass
