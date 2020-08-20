"""
This file hold algorithmic/complicated methods concerning assemblies.
This helps assembly.py be less bloated
"""

from __future__ import annotations
from random import sample
from typing import List, TYPE_CHECKING, Dict, Union, Iterable
from ..brain import Area

if TYPE_CHECKING:
    from ..brain import Brain
    from .assembly import Assembly, AssemblySet, Projectable


def common_value(*values):
    values = set(values)
    return values.pop() if len(values) == 1 else None


# TODO: change random sampling to performance sampling
# TODO2: sampling should take multiplicity into account
def activate(projectables: Iterable[Projectable], *, brain: Brain):
    """to prevent code duplication, this function does the common thing
    of taking a list of assemblies and creating a dictionary from area to neurons (of the
    assemblies) to set as winners"""

    """
    IMPORTANT (discuss with edo): we try to implement this in a new way:
    because the winners set must be of constant size, we choose randomly "k" winners
    out of the pool that is the union of all neurons of assemblies in the list.
    """
    from .assembly import Assembly
    assemblies = tuple(projectable for projectable in projectables if isinstance(projectable, Assembly))

    # TODO: List[float] or List[int]?
    # create a mapping from the areas to the neurons we want to fire
    area_neuron_mapping: Dict[Area, List[float]] = {ass.area: [] for ass in assemblies}

    for ass in assemblies:
        area_neuron_mapping[ass.area] += list(ass.sample_neurons(brain=brain))

    # update winners for relevant areas in the connectome
    for source in area_neuron_mapping.keys():
        brain.winners[source] = sample(area_neuron_mapping[source], k=source.k)  # choose randomly out of winners


def union(obj1: Union[Assembly, AssemblySet], obj2: Union[Assembly, AssemblySet]):
    """
    this method is set as __or__ of both assembly classes and returns an
    AssemblyTuple object which holds their union.
    """
    from .assembly import Assembly, AssemblySet

    if obj2 is Ellipsis:
        return AssemblySet(obj1)
    tuple1 = AssemblySet(obj1) if isinstance(obj1, Assembly) else obj1
    tuple2 = AssemblySet(obj2) if isinstance(obj2, Assembly) else obj2
    # We still support the '+' syntax for assembly tuples.
    return AssemblySet(*tuple1, *tuple2)
