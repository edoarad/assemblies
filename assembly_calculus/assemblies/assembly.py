from __future__ import annotations
# Allows forward declarations and such :)

from typing import Iterable, Union, Tuple, TYPE_CHECKING, Set, Optional, Dict

from .reader import Reader
from .assembly_readers.read_recursive import ReadRecursive
from ..utils import Recordable, ImplicitResolution, Bindable, UniquelyIdentifiable
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


# TODO: add uniqelyidentifyable
@Recordable(('merge', True), 'associate',
            resolution=ImplicitResolution(
                lambda instance, name: Bindable.implicitly_resolve_many(instance.assemblies, name, False), 'recording'))
class AssemblyTuple(object):
    """
    Assembly tuple is used as an intermediate structure to support syntax such as
    group merge ( a1 | a2 | .. | a_n >> area) and other group operations.
    """

    def __init__(self, *assemblies: Assembly):
        """
        :param assemblies: the set of assemblies in the tuple
        """

        # asserting tuple not empty, and that all object are projectable.
        if len(assemblies) == 0:
            raise IndexError("Assembly tuple is empty")
        if not all([isinstance(x, Assembly) or isinstance(x, Stimulus) for x in assemblies]):
            raise TypeError("Tried to initialize Assembly tuple with invalid object")

        self.assemblies: Tuple[Assembly, ...] = assemblies

    # TODO: This is confusing, because I expect Assembly + Assembly = Assembly.
    #       There are other solutions. Even just AssemblyTuple(ass1, ass2) >> area is
    #       better, but I'm sure you can do better than that.
    # RESPONSE: we think the syntax ass1 + ass2 >> area is cool and conveys the meaning of the action
    # and the syntax you recommended (AssemblyTuple(ass1, ass2) >> area) is still usable in this implementation
    # if someone prefers.
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

    def merge(self, area: Area, *, brain: Brain = None):
        """
        can be used by user with >> or directly by:
        (ass1 | ass2).merge( ... ) or AssemblyTuple(list of assemblies).merge( ... )
        """
        brain = brain or Bindable[Assembly].implicitly_resolve_many(self.assemblies, 'brain', False)[1]
        return util_merge(self.assemblies, area, brain=brain)

    def associate(self, other: AssemblyTuple, *, brain: Brain = None):
        """
        as of now has no syntactic sugar, so use by:
        (ass1 | ass2).associate( *another AssemblyTuple ) within a recipe context.
        """
        brain = brain or Bindable[Assembly].implicitly_resolve_many(self.assemblies + other.assemblies,
                                                                    'brain', False)[1]
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
@Recordable(('project', True), ('reciprocal_project', True))
@Bindable('brain')
class Assembly(UniquelyIdentifiable):
    # Response: An assembly is in particular a tuple of assemblies of length 1, they share many logical operations.
    # They share many properties, and in particular a singular assembly supports more operations.
    """
    A representation of an assembly of neurons that can be binded to a specific brain
    in which it appears. An assembly is defined primarily by its parents - the assemblies
    and/or stimuli that were fired to create it.
    This class implements basic operations on assemblies (project, reciprocal_project,
    merge and associate) by using a reader object, which interacts with the brain directly.
    """
    _default_reader: Reader = ReadRecursive

    @staticmethod
    def assembly_hash(area, parents):
        # we sort the list so that the order in the list of parents doesnt matter
        return hash((area, *sorted(parents, key=hash)))

    def __new__(cls, parents: Iterable[Projectable], area: Area, initial_recipes: Iterable[BrainRecipe] = None,
                reader: str = 'default'):
        return UniquelyIdentifiable.__new__(cls, uid=Assembly.assembly_hash(area, parents))

    def __init__(self, parents: Iterable[Projectable], area: Area,
                 initial_recipes: Iterable[BrainRecipe] = None, reader: Reader = None):
        """
        :param parents: the Assemblies and/or Stimuli that were used to create the assembly
        :param area: an Area where the Assembly "lives"
        :param initial_recipes: an iterable containing every BrainRecipe in which the assembly appears
        :param reader: name of a read driver pulled from assembly_readers. defaults to 'default'
        """

        # We hash an assembly using its parents (sorted by id) and area
        # this way equivalent assemblies have the same id.
        UniquelyIdentifiable.__init__(self)
        AssemblyTuple.__init__(self, self)

        self.parents: Tuple[Projectable, ...] = tuple(parents)
        self.area: Area = area
        self._reader = reader
        self.appears_in: Set[BrainRecipe] = set(initial_recipes or [])
        for recipe in self.appears_in:
            recipe.append(self)

    @property
    def reader(self) -> Reader:
        # property decorator means we can access this as assembly.reader
        return self._reader or Assembly._default_reader

    @staticmethod
    def set_default_reader(reader):
        Assembly._default_reader = reader

    def representative_neuron(self, preserve_brain=False, *, brain: Brain) -> Set[int, ...]:
        # TODO: Change name of Reader to Identifier???
        return set(self.reader.read(self, preserve_brain=preserve_brain, brain=brain))

    @staticmethod
    def read(area: Area, *, brain: Brain):
        # TODO: Decouple read into different modules
        assemblies: Set[Assembly] = brain.recipe.area_assembly_mapping[area]
        overlap: Dict[Assembly, float] = {}
        for assembly in assemblies:
            # TODO: extract calculation to function with indicative name
            overlap[assembly] = len(
                set(brain.winners[area]) & set(
                    assembly.representative_neuron(preserve_brain=True, brain=brain))) / area.k
        return max(overlap.keys(), key=lambda x: overlap[x])  # TODO: return None below some threshold

    # TODO: Remove this (And in reader class)
    def trigger_reader_update_hook(self, *, brain: Brain):
        """
        some read_drivers may want to be notified on certain changes
        we support this by calling this private function in key places (like project)
        which then triggers the hook in the reader (if it implements it)
        :param brain:
        :return:
        """
        self.reader.update_hook(self, brain=brain)

    def project(self, area: Area, *, brain: Brain = None) -> Assembly:
        """
        Projects an assembly into an area.

        :param brain: the brain in which the projection happens
        :param area: the area in which the new assembly is going to be created
        :returns: resulting projected assembly
        """
        if not isinstance(area, Area) and area in brain.recipe.areas:
            raise TypeError("Projection target must be an Area in the Brain")

        return util_merge((self,), area, brain=brain)  # project was actually just this line

    def __rshift__(self, target: Area):
        """
        In the context of assemblies, >> represents project.
        Example: a >> A (a is an assembly, A is an area)

        :param target: the area into which we project
        :returns: the new assembly that was created
        """
        if not isinstance(target, Area):
            raise TypeError("Assembly must be projected onto an area")
        return self.project(target)

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
        self.trigger_reader_update_hook(brain=brain)

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