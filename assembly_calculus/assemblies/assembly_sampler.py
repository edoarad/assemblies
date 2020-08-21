from __future__ import annotations
from abc import abstractmethod, ABC
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from assembly_calculus.assemblies.assembly import Assembly
    from assembly_calculus.brain import Brain


class AssemblySampler(ABC):

    @staticmethod
    @abstractmethod
    def sample_neurons(assembly: Assembly, preserve_brain: bool = False, *, brain: Brain):
        pass
