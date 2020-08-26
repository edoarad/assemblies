"""
This file hold algorithmic/complicated methods concerning assemblies.
This helps assembly.py be less bloated
"""

from __future__ import annotations
from typing import List, TYPE_CHECKING, Dict, Union, Iterable, Optional, Tuple
from assembly_calculus.brain import Area
import numpy as np

if TYPE_CHECKING:
    from assembly_calculus.brain import Brain
    from assembly_calculus.assemblies.assembly import Assembly, AssemblySet, Projectable


def common_value(*values) -> Optional:
    values = set(values)
    return values.pop() if len(values) == 1 else None


# TODO: change random sampling to performance sampling (To performance, if possible)
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

    # create a mapping from the areas to the neurons we want to fire, and weight of each neuron in selection
    area_neuron_mapping: Dict[Area, Tuple[List[int], List[float]]] = {ass.area: [[], []] for ass in assemblies}

    for ass in assemblies:
        area_neuron_mapping[ass.area][0] += list(ass.sample_neurons(brain=brain))

    for source in area_neuron_mapping.keys():
        no_repeat_neurons = list(set(area_neuron_mapping[source][0]))
        n = len(no_repeat_neurons)
        area_neuron_mapping[source][1] = [area_neuron_mapping[source][0].count(x)/n for x in no_repeat_neurons]
        area_neuron_mapping[source][0] = no_repeat_neurons

    # update winners for relevant areas in the connectome
    for source in area_neuron_mapping.keys():
        # choose randomly out of winners, according to weights
        brain.winners[source] = np.random.choice(area_neuron_mapping[source][0],
                                                 source.k, replace=False, p=area_neuron_mapping[source][1])


def union(obj1: Union[Assembly, AssemblySet], obj2: Union[Assembly, AssemblySet]) -> AssemblySet:
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
