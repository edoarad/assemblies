from __future__ import annotations
# Allows forward declarations and such :)
from collections import defaultdict
from itertools import product, chain
from typing import Iterable, Union, Tuple, TYPE_CHECKING, Set, Type, Optional

from assembly_calculus.assemblies.assembly_sampler import AssemblySampler
from assembly_calculus.assemblies.assembly_samplers.recursive_sampler import RecursiveSampler
from assembly_calculus.utils import ImplicitResolution, Bindable, UniquelyIdentifiable, set_hash, attach_recording, record_method, \
    bindable_brain
from assembly_calculus.brain import Stimulus, Area
from assembly_calculus.assemblies.utils import union, activate, common_value

if TYPE_CHECKING:
    from assembly_calculus.brain import Brain
    from assembly_calculus.brain import BrainRecipe


"""
Standard python 3.8 typing
Projectable is an umbrella type for regular assemblies 
and top level assemblies with no parents (i.e stimuli)
"""
Projectable = Union['Assembly', Stimulus]


"""
Magic constants
"""
PROJECT_REPEAT = 1
MERGE_REPEAT = 20
RECIPROCAL_PROJECT_REPEAT = MERGE_REPEAT
ASSOCIATE_REPEAT = 20


class AssemblySet(UniquelyIdentifiable):
    """
    Assembly tuple is used as an intermediate structure to support syntax such as
    assemblies merge (a1 | a2 | .. | a_n) >> area and other assemblies operations.
    """

    def __new__(cls, *assemblies):
        # We now allow AssemblyTuples to be unique by the hash of the tuple of the
        # sorted hashes of their content assemblies
        return UniquelyIdentifiable.__new__(cls, uid=set_hash(assemblies))

    def __init__(self, *assemblies: Assembly):
        """
        :param assemblies: The set of assemblies in the tuple
        """
        # Asserting tuple not empty, and that all object are projectable.
        if len(assemblies) == 0:
            raise IndexError("Assembly tuple is empty")

        if not all([isinstance(x, Assembly) or isinstance(x, Stimulus) for x in assemblies]):
            raise TypeError("Tried to initialize Assembly tuple with invalid object")

        UniquelyIdentifiable.__init__(self)
        # Remove duplicates
        self.assemblies: Tuple[Assembly, ...] = tuple(set(assemblies))

    def _merge_recording_resolver(self: AssemblySet, area: Area, **kwargs):
        return Bindable.bound_value('recording', *self)

    @record_method(_merge_recording_resolver, execute_anyway=True)
    @ImplicitResolution(brain=lambda self, area, **_: Bindable.bound_value('brain', *self))
    def merge(self, area: Area, *, brain: Brain = None) -> Assembly:
        """
        Creates strong bi-directional links between assemblies via an intermediate assembly located at area

        Can be used by user with >> or directly by:
        (ass1 | ass2).merge( ... ) or AssemblyTuple(list of assemblies).merge( ... )
        """
        if not isinstance(area, Area):
            raise ValueError("Project target must be an Area in the brain")

        merged_assembly = Assembly(self.assemblies, area,
                                   initial_recipes=set.intersection(*[x.appears_in for x in self]))
        if brain is not None:
            brain.winners[area] = list()
            all_parents = list(chain(*(assembly.parents for assembly in self)))
            activate(all_parents, brain=brain)

            subconnectome = defaultdict(lambda: set())
            for assembly in self:
                for parent in assembly.parents:
                    parent_area = parent if isinstance(parent, Stimulus) else parent.area
                    subconnectome[parent_area].add(assembly.area)

                subconnectome[assembly.area].add(area)
                subconnectome[area].add(assembly.area)

            brain.fire(subconnectome=subconnectome, iterations=brain.repeat * MERGE_REPEAT)

        merged_assembly.bind_like(*self)
        return merged_assembly

    def _associate_recording_resolver(self: AssemblySet, other: AssemblySet, **kwargs):
        return Bindable.bound_value('recording', *self, *other)

    @record_method(_associate_recording_resolver, execute_anyway=False)
    @ImplicitResolution(brain=lambda self, other, **_: Bindable.bound_value('brain', *self, *other))
    def associate(self, other: Optional[AssemblySet], *, brain: Brain):
        """
        Associates (causes overlap of representative_neurons to increase) assemblies in self with assemblies in other,
        can be thought of as performing associations (by activating all parents simultaneously)
         on all edges in the bipartite graph induced by self <-> other one-by-one.
        If other is None, performs association of self <-> self.

        As of now has no syntactic sugar, so use by:
        (... Left Side Assemblies ...).associate(... Right Side Assemblies ...)
        """
        if (area := common_value(*(assembly.area for assembly in (self | other)))) is None:
            raise ValueError("All assemblies must be in the same area for association")

        brain.winners[area] = list()
        pairs: Iterable[Tuple[Assembly, Assembly]] = product(self, other) if other is not None else product(self, self)
        for x, y in pairs:
            if x == y:
                # Skip associating assembly with self
                continue

            activate(x.parents + y.parents, brain=brain)

            parent_areas = {ass.area for ass in (x.parents + y.parents)}
            subconnectome = {parent_area: [area] for parent_area in parent_areas}
            # We compensate with more rounds, in order to avoid natural mixing (by removing self edge area: [area])
            brain.fire(subconnectome=subconnectome, iterations=brain.repeat * ASSOCIATE_REPEAT)

    def __rshift__(self, target_area: Area) -> Assembly:
        """
        In the context of assemblies, >> symbolizes merge.
        Example: (within a brain context) (a1 | a2 | a3)>>area

        :param target_area: the area we merge into
        :return: the new merged assembly
        """
        if not isinstance(target_area, Area):
            raise Exception("Assemblies must be merged onto an area")
        return self.merge(target_area)

    def __iter__(self):
        return iter(self.assemblies)

    __or__ = union  # we now support the | operator for both Assembly and AssemblyTuple objects.


@attach_recording
@bindable_brain.cls
class Assembly(UniquelyIdentifiable):
    """
    A representation of an assembly of neurons that can be binded to a specific brain
    in which it appears. An assembly is defined primarily by its parents - the assemblies
    and/or stimuli that were fired to create it.
    This class implements basic operations on assemblies (project, reciprocal_project,
    merge and associate) by using a AssemblySampler object, which interacts with the brain directly.
    """
    _default_sampler: Type[AssemblySampler] = RecursiveSampler

    def __new__(cls, parents: Iterable[Projectable], area: Area, initial_recipes: Iterable[BrainRecipe] = None,
                sampler: AssemblySampler = None):
        return UniquelyIdentifiable.__new__(cls, uid=hash((area, set_hash(parents))))

    def __init__(self, parents: Iterable[Projectable], area: Area,
                 initial_recipes: Iterable[BrainRecipe] = None, sampler: Type[AssemblySampler] = None):
        """
        :param parents: the Assemblies and/or Stimuli that were used to create the assembly
        :param area: an Area where the Assembly "lives"
        :param initial_recipes: an iterable containing every BrainRecipe in which the assembly appears
        :param sampler: a subclass of AssemblySampler that can sample what neurons should be fired in the next project
                        operation (which are the assemblies' representative neurons, used to perform read operations)
        """
        # We hash an assembly using its parents (sorted by id) and area
        # this way equivalent assemblies have the same id.
        UniquelyIdentifiable.__init__(self)

        self.parents: Tuple[Projectable, ...] = tuple(parents)
        self.area: Area = area
        self._sampler = sampler
        self.appears_in: Set[BrainRecipe] = set(initial_recipes or [])
        for recipe in self.appears_in:
            recipe.append(self)

    @property
    def sampler(self) -> Type[AssemblySampler]:
        # property decorator means we can access this as assembly.sampler
        return self._sampler or Assembly._default_sampler

    @staticmethod
    def set_default_sampler(sampler: Type[AssemblySampler]):
        """
        :param sampler: New default assembly sampler
        """
        Assembly._default_sampler = sampler

    @bindable_brain.method
    def sample_neurons(self, preserve_brain=False, *, brain: Brain) -> Set[int]:
        """
        :param preserve_brain: Boolean flag determining whether or not to have side effects on the brain
        :param brain: Brain from which to sample assembly
        :return: The set of neurons representing the assembly (at the current point of time)
        """
        return set(self.sampler.sample_neurons(self, preserve_brain=preserve_brain, brain=brain))

    @bindable_brain.property
    def representative_neurons(self, *, brain: Brain) -> Set[int]:
        return self.sample_neurons(preserve_brain=True, brain=brain)

    @record_method(execute_anyway=True)
    @bindable_brain.method
    def project(self, area: Area, *, brain: Brain = None) -> Assembly:
        """
        Projects an assembly into an area.
        :param brain: Should be supplied automatically by context (with brain / with recipe)
        :param area: the area in which the new assembly is going to be created
        :returns: resulting projected assembly
        """
        if not isinstance(area, Area):
            raise TypeError("Projection target must be an Area in the Brain")

        projected_assembly = Assembly([self], area, initial_recipes=self.appears_in)

        if brain is not None:
            activate([self], brain=brain)
            brain.winners[area] = list()
            brain.fire(subconnectome={self.area: [area], area: [area]},
                             iterations=brain.repeat * PROJECT_REPEAT)

        projected_assembly.bind_like(self)
        return projected_assembly

    def __rshift__(self, target: Area) -> Assembly:
        """
        In the context of assemblies, >> represents project.
        Example: assembly >> Area
        :param target: the area into which we project
        :returns: the new assembly that was created
        """
        return self.project(target)

    @record_method(execute_anyway=False)
    @bindable_brain.method
    def reciprocal_project(self, area: Area, *, brain: Brain = None) -> Assembly:
        """
        Reciprocally projects an assembly into an area,
        creating a projected assembly with strong bi-directional links to the current one.
        example usage:
        b = a.reciprocal_project(someArea)
        (now b.area = someArea, and b and a are strongly linked)
        :param area: the area into which we project
        :param brain: Should be supplied automatically by context (with brain / with recipe)
        :returns: Resulting projected assembly
        """
        if not isinstance(area, Area):
            raise TypeError("Project target must be an Area in the brain")

        # To improve convergence rate we decided to first project (and stabilize assembly)
        # and only then begin doing a reciprocal project
        projected_assembly = self.project(area, brain=brain)
        if brain is not None:
            activate(self.parents, brain=brain)

            subconnectome = {
                **{(parent.area if isinstance(parent, Assembly) else parent): [self.area] for parent in self.parents},
                self.area: [area], area: [self.area]
            }
            brain.fire(subconnectome=subconnectome, iterations=brain.repeat * RECIPROCAL_PROJECT_REPEAT)

        projected_assembly.bind_like(self)
        return projected_assembly

    def __lt__(self, other: Assembly):
        """
        Checks that other is a child assembly of self.
        :param other: the assembly we compare against
        """
        return isinstance(other, Assembly) and other in self.parents

    def __gt__(self, other: Assembly):
        """
        Checks if self is a child assembly of other.
        :param other: the assembly we compare against
        """
        return isinstance(other, Assembly) and self in other.parents

    def __repr__(self):
        return f"Assembly(parents=[%s], area=%s, sampler=%s)" %\
               (", ".join(parent.instance_name for parent in self.parents), self.area.instance_name,
                self.sampler.__name__)

    __or__ = union
