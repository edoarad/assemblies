from __future__ import annotations

from itertools import chain
from typing import Iterable, Dict, List, TYPE_CHECKING

from assembly_calculus.brain import Brain, Area, Stimulus

if TYPE_CHECKING:
    from assembly_calculus.assemblies import Projectable


def fire_many(brain: Brain, projectables: Iterable[Projectable], area: Area):
    """
    This function works by creating a "Parent tree", (Which is actually a directed acyclic graph) first,
    and then by going from the top layer, which will consist of stimuli, and traversing down the tree
    while firing each layer to areas its relevant descendant inhibit.
    For example, "firing" an assembly, we will first climb up its parent tree (Assemblies of course may have
    multiple parents, such as results of merge. Then we iterate over the resulting list in reverse, while firing
    each layer to the relevant areas, which are saved in a dictionary format:
    The thing we will project: areas to project it to

    :param brain: the brain in which the firing happens
    :param projectables: a list of projectable objects to be projected
    :param area: the area into which the objects are projected
    """
    # construct the firing hierarchy
    layers = _construct_firing_order(projectables, area)
    # now, fire each layer:
    _fire_layered_areas(brain, layers)


def _construct_firing_order(projectables: Iterable[Projectable], area: Area) -> List[Dict[Projectable, List[Area]]]:
    """
    Construct the "firing hierarchy", by going up the "parenthood" tree
    :param projectables: a list of projectable objects to be projected
    :param area: the area into which the objects are projected
    :return: Layered firing tree
    """
    from ..assemblies import Projectable, Assembly
    # initialize layers with the lowest level in the tree
    layers: List[Dict[Projectable, List[Area]]] = [{projectable: [area] for projectable in projectables}]
    # climb upwards until the current layers' parents are all stimuli (so there's no more climbing)
    while any(isinstance(projectable, Assembly) for projectable in layers[-1]):
        prev_layer: Iterable[Assembly] = (ass for ass in layers[-1].keys() if not isinstance(ass, Stimulus))
        current_layer: Dict[Projectable, List[Area]] = {}
        for ass in prev_layer:
            for parent in ass.parents:
                # map parent to all areas into which this parent needs to be fired
                current_layer[parent] = current_layer.get(ass, []) + [ass.area]

        layers.append(current_layer)

    # reverse the layers list to fire all parents the top to the original assemblies we've entered
    layers = layers[::-1]
    return layers


def _fire_layered_areas(brain: Brain, firing_order: List[Dict[Projectable, List[Area]]]):
    """
    Fire layers of projectables one by one
    :param brain: Brain in which to to fire
    :param firing_order: Layers of projectables, and corresponding areas (to fire into)
    """
    from ..assemblies import Assembly
    for layer in firing_order:
        stimuli_mappings: Dict[Stimulus, List[Area]] = {stim: areas
                                                        for stim, areas in
                                                        layer.items() if isinstance(stim, Stimulus)}
        assembly_mapping: Dict[Area, List[Area]] = {}
        for ass, areas in filter(lambda t: (lambda assembly, _: isinstance(assembly, Assembly))(*t), layer.items()):
            # map area to all areas into which this area needs to be fired
            assembly_mapping[ass.area] = assembly_mapping.get(ass.area, []) + areas

        mapping = {**stimuli_mappings, **assembly_mapping}

        targets = chain(*mapping.values())
        for target in targets:
            mapping[target] = mapping.get(target, []) + [target]

        brain.next_round(subconnectome=mapping)   # fire this layer of objects
