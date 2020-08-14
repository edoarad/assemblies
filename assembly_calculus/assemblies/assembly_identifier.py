from __future__ import annotations
from abc import abstractmethod, ABC
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .assembly import Assembly
    from ..brain import Brain


class AssemblyIdentifier(ABC):

    @staticmethod
    @abstractmethod
    def representative_neurons(assembly: Assembly, preserve_brain: bool = False, *, brain: Brain):
        pass
