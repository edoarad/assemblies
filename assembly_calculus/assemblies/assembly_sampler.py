from __future__ import annotations
from abc import abstractmethod, ABC
from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    from assembly_calculus.assemblies.assembly import Assembly
    from assembly_calculus.brain import Brain


class AssemblySampler(ABC):
    """
    An assembly sampler is an object that has the ability to sample neurons of a given assembly
    An assembly sampler is basically the definition of the assembly, it decides which neurons to fire when the assembly
    has to fire.
    """
    @staticmethod
    @abstractmethod
    def sample_neurons(assembly: Assembly, preserve_brain: bool = False, *, brain: Brain) -> Iterable[int]:
        pass
