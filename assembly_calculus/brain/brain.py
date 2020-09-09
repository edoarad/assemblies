from __future__ import annotations

from typing import Dict, Set, TYPE_CHECKING, List, Optional, Union, Type
from contextlib import contextmanager

from assembly_calculus.brain.brain_recipe import BrainRecipe
from assembly_calculus.brain.components import BrainPart, Stimulus, Area
from assembly_calculus.brain.connectome.abstract_connectome import AbstractConnectome
from assembly_calculus.utils import UniquelyIdentifiable

if TYPE_CHECKING:
    from assembly_calculus.assemblies import Assembly


class Brain(UniquelyIdentifiable):
    """
    Represents a simulated brain, with it's connectome which holds the areas, stimuli, and all the synapse weights.
    The brain updates by selecting a subgraph of stimuli and areas, and activating only those connections.
    The brain object works with a general connectome, which export an unified api for how the connections between the
    parts of the brain should be used.
    In case of need, one should extend the connectome API as he would like to make
    the implementation of the brain easier/better.
    Note that the brain implementation shouldn't depends on the
    underlying implementation of the connectome.

    Attributes:
        connectome: the brain's connectome object, holding the areas, stimuli and the synapse weights.
        recipe: a BrainRecipe object describing the recipe to build the current brain.
        repeat: number of times to perform fire (only assembly use it)

    Properties:
        winners: The winners of each area in the current state.
        support: The past-winners of each area until now.


    """

    def __init__(self, connectome: AbstractConnectome, recipe: BrainRecipe = None, repeat: int = 1):
        super(Brain, self).__init__()
        self.repeat = repeat
        self.recipe = recipe or BrainRecipe()
        self.connectome: AbstractConnectome = connectome
        self.ctx_stack: List[Dict[Union[BrainPart, Assembly], Optional[Brain]]] = []

        for area in self.recipe.areas:
            self.add_area(area)

        for stimulus in self.recipe.stimuli:
            self.add_stimulus(stimulus)

    def fire(self, subconnectome: Dict[BrainPart, Union[List[BrainPart], Set[BrainPart]]],
             iterations: int = 1, override_winners: Dict[Area, List[int]] = None):
        """
        Make the brain run a discrete round of firing while activating only some specific connections in the brain.
        :param subconnectome: A dictionary of connections to use in the projection
        :param iterations: number of fire iterations
        :param override_winners: if passed, will override the winners in the Area with the value
        """
        for _ in range(iterations):
            self.connectome.fire(subconnectome, override_winners=override_winners)

    def add_area(self, area: Area):
        """
        Add a new area to the brain
        :param area: New area object
        """
        self.recipe.append(area)
        self.connectome.add_area(area)

    def add_stimulus(self, stimulus: Stimulus):
        """
        Add a new stimulus to the brain
        :param stimulus: New area object
        """
        self.recipe.append(stimulus)
        self.connectome.add_stimulus(stimulus)

    @property
    def winners(self):
        return self.connectome.winners

    @property
    def support(self):
        return self.connectome.support

    @contextmanager
    def temporary_plasticity(self, mode: bool):
        """
        During the duration of this context manager the plasticity would be enabled if the mode is True.
        Usage:
        >>> with Brain().temporary_plasticity(False) as brain:
        >>>     brain.fire(...)  # plasticity is off now
        """
        original_plasticity: bool = self.connectome.plasticity
        self.connectome.plasticity = mode
        yield self
        self.connectome.plasticity = original_plasticity

    @contextmanager
    def freeze(self, freeze: bool = True):
        """
        During the duration of this context manager, there would be no changes in the connectome (winners or
        Usage:
        >>> brain = Brain()
        >>> with brain.freeze():
        >>>     brain.fire(...)  # plasticity is off now
        >>> # brain has the same winners as before
        """
        if not freeze:
            yield self
            return

        original_winners = {area: self.winners[area].copy() for area in self.recipe.areas}
        with self.temporary_plasticity(mode=False):
            yield self
        for area in self.recipe.areas:
            self.winners[area] = original_winners[area]

    def __enter__(self):
        current_ctx_stack: Dict[Union[BrainPart, Assembly], Optional[Brain]] = {}

        for area in self.recipe.areas:
            if 'brain' in area.bound_params:
                current_ctx_stack[area] = area.bound_params['brain']
            area.bind(brain=self)

        for assembly in self.recipe.assemblies:
            if 'brain' in assembly.bound_params:
                current_ctx_stack[assembly] = assembly.bound_params['brain']
            assembly.bind(brain=self)

        self.ctx_stack.append(current_ctx_stack)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        current_ctx_stack: Dict[Union[BrainPart, Assembly], Optional[Brain]] = self.ctx_stack.pop()

        for area in self.recipe.areas:
            area.unbind('brain')
            if area in current_ctx_stack:
                area.bind(brain=current_ctx_stack[area])

        for assembly in self.recipe.assemblies:
            assembly.unbind('brain')
            if assembly in current_ctx_stack:
                assembly.bind(brain=current_ctx_stack[assembly])


def bake(recipe: BrainRecipe, p: float, connectome_cls: Type[AbstractConnectome],
         train_repeat: int = 10, effective_repeat: int = 3) -> Brain:
    """Bakes a brain from a recipe, adds all relevant brain parts and performs the initialization sequence"""
    brain = Brain(connectome_cls(p), recipe=recipe, repeat=train_repeat)
    recipe.initialize_brain(brain)
    brain.repeat = effective_repeat
    return brain
