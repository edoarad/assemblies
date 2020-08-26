from __future__ import annotations
from typing import List, Union, TYPE_CHECKING, Dict, Optional, Set

from assembly_calculus.brain.components import Area, Stimulus, BrainPart
from assembly_calculus.utils import Recording

# TODO: (To API Team), if you don't like the name, please change it
BrainObjects = None

if TYPE_CHECKING:
    from assembly_calculus.brain import Brain
    from assembly_calculus.assemblies import Assembly
    BrainObjects = Union[BrainPart, Assembly]


class BrainRecipe:
    """
    Stores a description of a brain, that can then be initialized (baked)

    Stores:
        - Areas
        - Stimuli,
        - Assemblies
        - Initialization sequence
    """
    def __init__(self, *brain_objects: BrainObjects):
        # Areas in brain
        self.areas: Set[Area] = set()
        # Assemblies in brain
        self.assemblies: Set[Assembly] = set()
        # Assemblies in each area
        self.area_assembly_mapping: Dict[Area, Set[Assembly]] = {}
        # Stimuli in brain
        self.stimuli: Set[Stimulus] = set()

        # Actions to perform on initialization
        self.initialization: Recording = Recording()

        # Stack for Context Manager, used to store bound values
        self.ctx_stack: List[Dict[Assembly, Recording]] = []

        # Add parts to brain
        self.extend(*brain_objects)

    def _add_area(self, area: Area):
        self.areas.add(area)
        if area not in self.area_assembly_mapping:
            self.area_assembly_mapping[area] = set()

    def _add_stimulus(self, stimulus: Stimulus):
        self.stimuli.add(stimulus)

    def _add_assembly(self, assembly: Assembly):
        self._add_area(assembly.area)
        self.assemblies.add(assembly)
        self.area_assembly_mapping[assembly.area].add(assembly)
        if self not in assembly.appears_in:
            assembly.appears_in.add(self)

    def append(self, brain_object: BrainObjects):
        from ..assemblies import Assembly

        if isinstance(brain_object, Area):
            self._add_area(brain_object)
        elif isinstance(brain_object, Stimulus):
            self._add_stimulus(brain_object)
        elif isinstance(brain_object, Assembly):
            self._add_assembly(brain_object)
        else:
            raise ValueError("Invalid part")

    def extend(self, *brain_objects: BrainObjects):
        for brain_object in brain_objects:
            self.append(brain_object)

    def initialize_brain(self, brain: Brain):
        """Run initialization sequence stored in the recipe"""
        self.initialization.play(brain=brain)

    def __enter__(self):
        current_ctx_stack: Dict[Assembly, Optional[Recording]] = {}

        for assembly in self.assemblies:
            if 'recording' in assembly.bound_params:
                current_ctx_stack[assembly] = assembly.bound_params['recording']
            assembly.bind(recording=self.initialization)

        self.ctx_stack.append(current_ctx_stack)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        current_ctx_stack: Dict[Assembly, Optional[Recording]] = self.ctx_stack.pop()

        for assembly in self.assemblies:
            assembly.unbind('recording')
            if assembly in current_ctx_stack:
                assembly.bind(recording=current_ctx_stack[assembly])
