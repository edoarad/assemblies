from __future__ import annotations
from typing import Optional, Union, TYPE_CHECKING, Dict, Set, Tuple
from assembly_calculus.brain.components_utils import verify_component_params

from assembly_calculus.utils import UniquelyIdentifiable, bindable_brain, overlap

if TYPE_CHECKING:
    from assembly_calculus.brain import Brain
    from assembly_calculus.assemblies.assembly import Assembly


@bindable_brain.cls
class Area(UniquelyIdentifiable):
    """
    Area is a description of some place in the brain.
    The area consists of:
    Attributes:
        n: The number of neurons in it.
        k: The number of firing neurons in each round.
        beta: Plasticity constant which adjust how much each edge will change between rounds.
    The area can be added to a brain, (either by designing a connectome and constructing a brain from it, or
    by directly adding the brain).
    The same area can be added to multiple brains. (Effective for describing multiple brains with similar design)
    Bindable_properties:
        When the area is bounded to some brain, the area can provide the following properties (delegated from the bounded
        brain)
        winners - the winners of the area.
        support - the support in the area.
        active_assembly - Find the active assembly in the current area.
    """

    def __init__(self, n: int, k: Optional[int] = None, beta: float = 0.01):
        super(Area, self).__init__()

        verify_component_params(n,k,beta)
        self.beta: float = beta
        self.n: int = n
        self.k: int = k or int(n ** 0.5)

    @bindable_brain.property
    def winners(self, *, brain: Brain) -> Set[int]:
        return set(brain.winners[self])

    @bindable_brain.property
    def support(self, *, brain: Brain):
        return brain.support[self]

    @bindable_brain.method
    def read(self, *, preserve_brain: bool = True, brain: Brain) -> Optional[Assembly]:
        """
        Returns the most activated assembly in the area.
        Note: only available when bounded to some brain.
        """
        THRESHOLD = 0.20  # If there is no assembly that overlaps with winners in at least TRESHOLD percent, return None
        assemblies: Set[Assembly] = brain.recipe.area_assembly_mapping[self]
        overlaps: Dict[Assembly, float] = {}
        for assembly in assemblies:
            overlaps[assembly] = overlap(brain.winners[self],
                                         assembly.sample_neurons(preserve_brain=preserve_brain, brain=brain))

        maximal_assembly = max(overlaps.keys(), key=lambda x: overlaps[x])
        return maximal_assembly if overlaps[maximal_assembly] > THRESHOLD else None

    @bindable_brain.property
    def active_assembly(self, *, brain: Brain) -> Optional[Assembly]:
        return self.read(preserve_brain=True, brain=brain)

    def __repr__(self):
        return f"Area(n={self.n}, k={self.k}, beta={self.beta})"


class Stimulus(UniquelyIdentifiable):
    """
    Stimulus is a description of receptive neurons for a brain which alert the brain on some input. (For example some
    smell is currently present).
    Same as the area the stimulus can be added to multiple brain parts.
    Attributes:
        n: The number of neurons in the stimulus (during a firing all of them fire).
        beta: Plasticity constant which adjust how much each edge will change between rounds.
    """
    def __init__(self, n: int, beta: float = 0.05):
        super(Stimulus, self).__init__()
        self.n = n
        self.beta = beta

    def __repr__(self):
        return f"Stimulus(n={self.n}, beta={self.beta})"


class OutputArea(Area):
    """
    Special case of area with n=2, k=1.
    This meant to implement an output bit.
    Where the first neuron represent 0 and the second 1.
    Note: Since k=1 only one of them can fire in each point.
    """
    def __init__(self, beta: float):
        super(OutputArea, self).__init__(n=2, k=1, beta=beta)

    def __repr__(self):
        return f"OutputArea(n={self.n}, beta={self.beta})"


BrainPart = Union[Area, Stimulus]


class Connection:
    def __init__(self, source: BrainPart, dest: BrainPart, synapses=None):
        """
        Generic representation of a connection between two brain parts.
        :param source: The source brain part.
        :param dest: The destination brain part.
        :param synapses: Some dict-like representation of the synapses.
        """
        self.source: BrainPart = source
        self.dest: BrainPart = dest
        self.synapses = synapses if synapses is not None else {}

    @property
    def beta(self):
        """The beta of the connection"""
        return self.dest.beta

    def __getitem__(self, key: Tuple[int, int]):
        return self.synapses[key]

    def __setitem__(self, key: Tuple[int, int], value: float):
        self.synapses[key] = value

    def __repr__(self):
        return f"Connection(synapses={self.synapses!r})"
