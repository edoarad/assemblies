from __future__ import annotations
# Allows forward declarations and such :)

from typing import Iterable, Union, Tuple, TYPE_CHECKING, Set

from .assembly_sampler import AssemblySampler
from .assembly_samplers.recursive_sampler import RecursiveSampler
from ..utils import ImplicitResolution, Bindable, UniquelyIdentifiable, set_hash, attach_recording, record_method, \
    bindable_brain
from ..brain import Stimulus, Area
from .utils import util_merge, util_associate, union

if TYPE_CHECKING:  # TODO: this is not needed. It's better to always import them.
    # Response: Sadly we need to do it to avoid cyclic imports,
    #           So I use TYPE_CHECKING for typing only imports
    from ..brain import Brain
    from ..brain import BrainRecipe

"""
Standard python 3.8 typing
Projectable is an umbrella type for regular assemblies 
and top level assemblies with no parents (i.e stimuli)
"""
Projectable = Union['Assembly', Stimulus]


class AssemblyTuple(UniquelyIdentifiable):
    """
    Assembly tuple is used as an intermediate structure to support syntax such as
    group merge ( a1 | a2 | .. | a_n >> area) and other group operations.
    """

    def __new__(cls, *assemblies):
        # We now allow AssemblyTuples to be unique by the hash of the tuple of the
        # sorted hashes of their content assemblies
        return UniquelyIdentifiable.__new__(cls, uid=set_hash(assemblies))

    def __init__(self, *assemblies: Assembly):
        """
        :param assemblies: the set of assemblies in the tuple
        """
        # Asserting tuple not empty, and that all object are projectable.
        if len(assemblies) == 0:
            raise IndexError("Assembly tuple is empty")

        if not all([isinstance(x, Assembly) or isinstance(x, Stimulus) for x in assemblies]):
            raise TypeError("Tried to initialize Assembly tuple with invalid object")

        UniquelyIdentifiable.__init__(self)
        # TODO: Convert to set so no duplicates?
        self.assemblies: Tuple[Assembly, ...] = assemblies

    def __add__(self, other: AssemblyTuple) -> AssemblyTuple:
        """
        In the context of AssemblyTuples, + creates a new AssemblyTuple containing the members
        of both parts.

        :param other: the other AssemblyTuple we add
        :returns: the new AssemblyTuple
        """
        if not isinstance(other, AssemblyTuple):
            raise TypeError("Assemblies can be concatenated only to assemblies")
        return AssemblyTuple(*(self.assemblies + other.assemblies))

    @record_method(lambda self, area, **_: Bindable.bound_value('recording', *self), execute_anyway=True)
    @ImplicitResolution(brain=lambda self, area, **_: Bindable.bound_value('brain', *self))
    def merge(self, area: Area, *, brain: Brain = None):
        """
        can be used by user with >> or directly by:
        (ass1 | ass2).merge( ... ) or AssemblyTuple(list of assemblies).merge( ... )
        """
        return util_merge(self.assemblies, area, brain=brain)

    @record_method(lambda self, other, **_: Bindable.bound_value('recording', *self, *other), execute_anyway=False)
    @ImplicitResolution(brain=lambda self, other, **_: Bindable.bound_value('brain', *self, *other))
    def associate(self, other: AssemblyTuple, *, brain: Brain):
        """
        as of now has no syntactic sugar, so use by:
        (ass1 | ass2).associate( *another AssemblyTuple ) within a recipe context.
        """
        return util_associate(self.assemblies, other.assemblies, brain=brain)

    def __rshift__(self, target_area: Area):
        """
        In the context of assemblies, >> symbolizes merge.
        Example: (within a brain context) (a1|a2|a3)>>area

        :param target_area: the area we merge into
        :return: the new merged assembly
        """
        if not isinstance(target_area, Area):
            raise Exception("Assemblies must be merged onto an area")
        return self.merge(target_area)

    def __iter__(self):
        return iter(self.assemblies)

    __or__ = union  # we now support the | operator for both Assembly and AssemblyTuple objects.


# TODO: Better documentation for user-functions, add example usages w\ and w\o bindable
@attach_recording
@bindable_brain.cls
class Assembly(UniquelyIdentifiable):
    # Response: An assembly is in particular a tuple of assemblies of length 1, they share many logical operations.
    # They share many properties, and in particular a singular assembly supports more operations.
    """
    A representation of an assembly of neurons that can be binded to a specific brain
    in which it appears. An assembly is defined primarily by its parents - the assemblies
    and/or stimuli that were fired to create it.
    This class implements basic operations on assemblies (project, reciprocal_project,
    merge and associate) by using a AssemblySampler object, which interacts with the brain directly.
    """
    _default_sampler: AssemblySampler = RecursiveSampler

    def __new__(cls, parents: Iterable[Projectable], area: Area, initial_recipes: Iterable[BrainRecipe] = None,
                sampler: AssemblySampler = None):
        return UniquelyIdentifiable.__new__(cls, uid=hash((area, set_hash(parents))))

    def __init__(self, parents: Iterable[Projectable], area: Area,
                 initial_recipes: Iterable[BrainRecipe] = None, sampler: AssemblySampler = None):
        """
        :param parents: the Assemblies and/or Stimuli that were used to create the assembly
        :param area: an Area where the Assembly "lives"
        :param initial_recipes: an iterable containing every BrainRecipe in which the assembly appears
        :param sampler: a subclass of AssemblySampler that can sample what neurons should be fired in the next project
                        operation
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
    def sampler(self) -> AssemblySampler:
        # property decorator means we can access this as assembly.sampler
        return self._sampler or Assembly._default_sampler

    @staticmethod
    def set_default_sampler(sampler):
        Assembly._default_sampler = sampler

    @bindable_brain.method
    def sample_neurons(self, preserve_brain=False, *, brain: Brain) -> Set[int, ...]:
        return set(self.sampler.sample_neurons(self, preserve_brain=preserve_brain, brain=brain))

    @bindable_brain.property
    def representative_neurons(self, *, brain: Brain) -> Set[int, ...]:
        return self.sample_neurons(preserve_brain=True, brain=brain)

    @record_method(execute_anyway=True)
    @bindable_brain.method
    def project(self, area: Area, *, brain: Brain = None) -> Assembly:
        """
        Projects an assembly into an area.

        :param brain: the brain in which the projection happens
        :param area: the area in which the new assembly is going to be created
        :returns: resulting projected assembly
        """
        if not isinstance(area, Area):
            raise TypeError("Projection target must be an Area in the Brain")
        return util_merge((self,), area, brain=brain)  # project was actually just this line

    def __rshift__(self, target: Area):
        """
        In the context of assemblies, >> represents project.
        Example: a >> A (a is an assembly, A is an area)

        :param target: the area into which we project
        :returns: the new assembly that was created
        """
        # TODO: No need for error since it already appears in project
        if not isinstance(target, Area):
            raise TypeError("Assembly must be projected onto an area")
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
        :param brain: should be supplied by the context of usage, NOT manually by user
        :returns: Resulting projected assembly
        """
        projected_assembly: Assembly = self.project(area, brain=brain)
        projected_assembly.project(self.area, brain=brain)
        return projected_assembly

    # TODO: lt and gt logic can be implemented using a common method
    # Response: True, but I think it is a tad more readable this way
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

    __or__ = union
