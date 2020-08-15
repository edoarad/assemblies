"""
This file hold algorithmic/complicated methods concerning assemblies.
This helps assembly.py be less bloated
"""

from __future__ import annotations
from itertools import product
from random import sample
from typing import List, Tuple, TYPE_CHECKING, Dict, Union
from ..brain import Area

if TYPE_CHECKING:
    from ..brain import Brain
    from .assembly import Assembly, AssemblyTuple


# TODO: change random sampling to performance sampling
# TODO: get k from area
def activate_assemblies(assemblies: Tuple[Assembly, ...], *, brain: Brain):
    """to prevent code duplication, this function does the common thing
    of taking a list of assemblies and creating a dictionary from area to neurons (of the
    assemblies) to set as winners"""

    """
    IMPORTANT (discuss with edo): we try to implement this in a new way:
    because the winners set must be of constant size, we choose randomly "k" winners
    out of the pool that is the union of all neurons of assemblies in the list.
    """

    if len(assemblies) == 0:
        raise IndexError("tried to activate empty list of assemblies")

    # create a mapping from the areas to the neurons we want to fire
    area_neuron_mapping: Dict[Area, List[float]] = {ass.area: [] for ass in assemblies}

    # we save the amount of winners brain expects
    read_result = list(assemblies[0].sample_neurons(brain=brain))
    k = len(read_result)
    for ass in assemblies:
        area_neuron_mapping[ass.area] += list(ass.sample_neurons(brain=brain))

    # update winners for relevant areas in the connectome
    for source in area_neuron_mapping.keys():
        brain.winners[source] = sample(area_neuron_mapping[source], k=k)  # choose randomly out of winners


def util_associate(a: Tuple[Assembly, ...], b: Tuple[Assembly, ...], *, brain: Brain) -> None:
    """
    Associates two lists of assemblies, by strengthening each bond in the
    corresponding bipartite graph.
    for simple binary operation use Assembly.associate([a],[b]).
    for each x in A, y in B, associate (x,y).
    A1 z-z B1
    A2 -X- B2
    A3 z-z B3

    :param brain: brain context to work in, passed from AssemblyTuple which is bound to a brain.
    :param a: first list
    :param b: second list
    """
    if len(a) == 0 or len(b) == 0:
        raise IndexError("one side of associate is Empty!")
    pairs = product(a, b)
    for x, y in pairs:
        activate_assemblies((x, y), brain=brain)
        brain.next_round(subconnectome={x.area: [x.area]}, replace=True, iterations=brain.repeat)


def util_merge(assemblies: Tuple[Assembly, ...], area: Area, *, brain: Brain = None):
    """
    Creates a new assembly with all input assemblies as parents.
    Practically creates a new assembly with one-directional links from parents
    (this function should not be called by the user, and is used internally
    by AssemblyTuple. user should use the >> syntax documented in merge).

    :param brain: the brain in which the merge occurs (again, passed from AssemblyTuple)
    :param assemblies: the parents of the new merged assembly
    :param area: the area into which we merge
    :returns: resulting merged assembly
    """
    from .assembly import Assembly  # we cannot import this for the rest of the file, circular import

    # Response: Added area checks
    if not isinstance(area, Area):
        raise TypeError("Project target must be an Area in the brain")

    merged_assembly: Assembly = Assembly(assemblies, area,
                                         initial_recipes=set.intersection(*[x.appears_in for x in assemblies]))
    # TODO: this is actually a way to check if we're in "binded" or "non binded" state.
    # TODO: can you think of a nicer way to do that?
    # TODO: otherwise it seems like a big block of code inside the function that sometimes happens and sometimes not. it is error-prone
    # Response: This does not serve as a check to see if we are bound or not,
    #           this serves as a way to perform syntactic assemblies operations in order to define new assemblies
    #           without performing the operations themselves.
    #           This is simply to support the recipe ecosystem.
    #
    #           In my opinion, this class should be designed as if binding does not exist, and binding is
    #           purely a syntactic sugar that makes the usage easier
    if brain is not None:
        activate_assemblies(assemblies, brain=brain)

        # TODO: Is this OK? (To Edo)
        brain.winners[area] = list()
        brain.next_round(subconnectome={**{ass.area: [area] for ass in assemblies}, area: [area]}, replace=True,
                         iterations=brain.repeat)
    merged_assembly.bind_like(*assemblies)
    return merged_assembly


def union(obj1: Union[Assembly, AssemblyTuple], obj2: Union[Assembly, AssemblyTuple]):
    from .assembly import Assembly, AssemblyTuple
    """
    this method is set as __or__ of both assembly classes and returns an
    AssemblyTuple object which holds their union.
    """
    if obj2 is ellipsis:
        return AssemblyTuple(obj1)
    tuple1 = AssemblyTuple(obj1) if isinstance(obj1, Assembly) else obj1
    tuple2 = AssemblyTuple(obj2) if isinstance(obj2, Assembly) else obj2
    # We still support the '+' syntax for assembly tuples.
    return tuple1 + tuple2
