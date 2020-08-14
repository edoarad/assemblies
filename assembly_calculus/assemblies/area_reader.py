from __future__ import annotations
from typing import TYPE_CHECKING, Set, Dict
if TYPE_CHECKING:
    from .assembly import Assembly
    from ..brain import Brain, Area


# TODO: Generalize to function tuple
class AreaReader:

    def __init__(self, read_methods=None):
        self.read_methods = [AreaReader.default_read] + read_methods

    # TODO: Replace with __call__?
    def read(self, area: Area, *, brain: Brain):
        return [read(area, brain=brain) for read in self.read_methods]

    # TODO: Maybe rename to append/extend?
    def add_read_methods(self, read_methods):
        self.read_methods += read_methods

    # TODO: Change to register_function?
    def read_method_wrapper(self, read_method):
        self.read_methods.append(read_method)
        return read_method

    # TODO: Maybe just call this index?
    def get_read_method_index(self, read_method):
        return self.read_methods.find(read_method)

    @staticmethod
    def default_read(area: Area, *, brain: Brain):
        assemblies: Set[Assembly] = brain.recipe.area_assembly_mapping[area]
        overlap: Dict[Assembly, float] = {}
        for assembly in assemblies:
            # TODO: extract calculation to function with indicative name
            overlap[assembly] = len(
                set(brain.winners[area]) & set(
                    assembly.representative_neurons(preserve_brain=True, brain=brain))) / area.k
        return max(overlap.keys(), key=lambda x: overlap[x])  # TODO: return None below some threshold
