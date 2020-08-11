from __future__ import annotations  # TODO: remove this allover, we are using python 3
# Response: It actually allows forward declarations and such :)
from typing import Iterable, Union, Tuple, TYPE_CHECKING, Set, Optional, Dict
from itertools import product

from .read_driver import ReadDriver  # TODO: It shouldn't depend on directory structure.
from ..utils import Recordable, ImplicitResolution, Bindable, UniquelyIdentifiable
from ..brain import Stimulus, Area
from .reader import Reader
from .assembly_readers.read_recursive import ReadRecursive

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


@Recordable(('merge', True), '_associate',
            resolution=ImplicitResolution(
                lambda instance, name: Bindable.implicitly_resolve_many(instance.assemblies, name, False), 'recording'))
class AssemblyTuple(object):
    """
    Assembly tuple is used as an intermediate structure to support syntax such as
    group merge ( a1 + a2 + .. + a_n >> area) and other group operations.
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
        brain = brain or Bindable[Assembly].implicitly_resolve_many(self.assemblies, 'brain', False)[1]
        return Assembly._merge(self.assemblies, area, brain=brain)

    def associate(self, other: AssemblyTuple, *, brain: Brain = None):
        brain = brain or Bindable[Assembly].implicitly_resolve_many(self.assemblies + other.assemblies,
                                                                    'brain', False)[1]
        return Assembly._associate(self.assemblies, other.assemblies, brain=brain)

    def __rshift__(self, target_area: Area):
        """
        In the context of assemblies, >> symbolizes merge.
        Example: (within a brain context) (a1+a2+a3)>>area

        :param target_area: the area we merge into
        :return: the new merged assembly
        """
        if not isinstance(target_area, Area):
            raise Exception("Assemblies must be merged onto an area")
        return self.merge(target_area)

    def __iter__(self):
        return iter(self.assemblies)


@Recordable(('project', True), ('reciprocal_project', True))
@Bindable('brain')
class Assembly(UniquelyIdentifiable, AssemblyTuple):
    # TODO: It makes no logical sense for Assembly to inherit AssemblyTuple.
    # TODO: instead, they can inherit from a mutual `AssemblyOperator` class that defines the operators they both support
    # An assembly is in particular a tuple of assemblies of length 1, they share many logical operations.
    # They share many properties, and in particular a singular assembly supports more operations.
    """
    A representation of an assembly of neurons that can be binded to a specific brain
    in which it appears. An assembly is defined primarily by its parents - the assemblies
    and/or stimuli that were fired to create it.
    This class implements basic operations on assemblies (project, reciprocal_project,
    merge and associate) by using a reader object, which interacts with the brain directly.
    """
    _default_reader: Reader = ReadRecursive

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
        UniquelyIdentifiable.__init__(self, uid=hash((area, *sorted(parents, key=hash))))
        AssemblyTuple.__init__(self, self)

        self.parents: Tuple[Projectable, ...] = tuple(parents)
        self.area: Area = area
        self.reader = reader
        self.appears_in: Set[BrainRecipe] = set(initial_recipes or [])
        for recipe in self.appears_in:
            recipe.append(self)

    @staticmethod
    def set_default_reader(reader):
        Assembly._default_reader = reader

    # TODO: this name is not indicative. Perhaps change to something like to_representative_neuron_subset.
    # Response: We will add this in the documentation, to_representative_neuron_subset is simply too long
    # TODO: reader.read is _very_ confusing with Assembly.read. Rename reader.
    def identify(self, preserve_brain=False, *, brain: Brain) -> Set[int, ...]:
        if self.reader:
            read = self.reader.read
        else:
            read = Assembly._default_reader.read
        return set(read(self, preserve_brain=preserve_brain, brain=brain))

    @staticmethod
    def read(area: Area, *, brain: Brain):
        assemblies: Set[Assembly] = brain.recipe.area_assembly_mapping[area]
        overlap: Dict[Assembly, float] = {}
        for assembly in assemblies:
            # TODO: extract calculation to function with indicative name
            overlap[assembly] = len(
                set(area.winners) & set(assembly.identify(preserve_brain=True, brain=brain))) / area.k
        return max(overlap.keys(), key=lambda x: overlap[x])  # TODO: return None below some threshold

    # TODO 4: there is no existing reader with `update_hook`. either make such reader and test the code using it, or remove all update_hook usages
    # TODO: Yonatan: Let's just remove this
    def trigger_reader_update_hook(self, *, brain: Brain):
        """
        some read_drivers may want to be notified on certain changes
        we support this by calling this private function in key places (like project)
        which then triggers the hook in the reader (if it implements it)
        :param brain:
        :return:
        """
        self.reader.update_hook(self, brain=brain)

    # TODO: throughout bindable classes, users might error and give the brain parameter even if the object is binded.
    #       Is this a problem? can you help the user not make any mistakes?
    # Response: This is a feature by design, and allows regular code to ignore explicit binding
    #           (needed to function properly). An inexperienced user should never pass the brain parameter explicitly,
    #           so this shouldn't happen anyway.
    # TODO: add option to manually change the assemblies' recipes
    # Response: To avoid bugs, this assembly is added to all recipes it can be used in
    def project(self, area: Area, *, brain: Brain = None, iterations: Optional[int] = None) -> Assembly:
        """
        Projects an assembly into an area.

        :param brain: the brain in which the projection happens
        :param area: the area in which the new assembly is going to be created
        :returns: resulting projected assembly
        """
        if not isinstance(area, Area) and area in brain.recipe.areas:
            raise TypeError("Projection target must be an Area in the Brain")

        projected_assembly: Assembly = Assembly([self], area, initial_recipes=self.appears_in)
        if brain is not None:

            # read activate_assemblies for documentation
            Assembly.activate_assemblies([self], area, brain)

            projected_assembly.trigger_reader_update_hook(brain=brain)

        # TODO: calling `bind_like` manually is error-prone because someone can forget it. can you make a decorator or a more automated way to do it?
        # Response: This is the standard path defined in the Bindable API,
        #           And "automation" will be quite weird, anyway this is used only a couple of times, and only in internal API
        projected_assembly.bind_like(self)
        return projected_assembly

    @staticmethod
    def activate_assemblies(assemblies, area: Area,  brain : Brain):
        """
        to prevent code duplication, this function does the common thing
        of taking a list of assemblies and creating a dictionary from area to neurons (of the
        assemblies) to set as winners, and fire in the next round (into area).
        (activating the assemblies in the list).
        """
        # create a mapping from the areas to the neurons we want to fire
        area_neuron_mapping = {ass.area: [] for ass in assemblies}
        for ass in assemblies:
            area_neuron_mapping[ass.area] = list(
                ass.identify(brain=brain))

        # update winners for relevant areas in the connectome
        for source in area_neuron_mapping.keys():
            brain.connectome.winners[source] = area_neuron_mapping[source]

        # creating the correct subconnectome
        subconnectome = {source: [area] for source in area_neuron_mapping}
        # we also want area to be able to fire into itself (area -> area)
        subconnectome[area] = subconnectome.get(area, []) + [area]

        # Replace=True for better performance
        brain.next_round(subconnectome=subconnectome, replace=True,
                         iterations=brain.repeat)

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

        :param brain: the brain in which the projection occurs
        :param area: the area into which we project
        :returns: Resulting projected assembly
        """
        projected_assembly: Assembly = self.project(area, brain=brain)
        projected_assembly.project(self.area, brain=brain)
        self.trigger_reader_update_hook(brain=brain)

        return projected_assembly

    @staticmethod
    def _merge(assemblies: Tuple[Assembly, ...], area: Area, *, brain: Brain = None):
        """
        Creates a new assembly with all input assemblies as parents.
        Practically creates a new assembly with one-directional links from parents
        ONLY CALL AS: Assembly.merge(...), as the function is invariant under input order.

        :param brain: the brain in which the merge occurs
        :param assemblies: the parents of the new merged assembly
        :param area: the area into which we merge
        :returns: resulting merged assembly
        """
        # Response: Added area checks
        if not isinstance(area, Area) and area in brain.recipe.areas:
            raise TypeError("Project target must be an Area in the brain")

        # TODO 2: check documentation of `intersection` - it seems to be an instance method that works here by chance!
        # Response: In the official documentation they actually mention it
        #           - https://docs.python.org/3/library/stdtypes.html#frozenset.intersection
        #           They probably forgot to update this in the source code?
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

            # Read activate_assemblies for documentation
            Assembly.activate_assemblies(assemblies, area, brain)

            merged_assembly.trigger_reader_update_hook(brain=brain)
        merged_assembly.bind_like(*assemblies)
        return merged_assembly

    @staticmethod
    def _associate(a: Tuple[Assembly, ...], b: Tuple[Assembly, ...], *, brain: Brain = None) -> None:
        # TODO: it's not the right logic
        """
        Associates two lists of assemblies, by strengthening each bond in the
        corresponding bipartite graph.
        for simple binary operation use Assembly.associate([a],[b]).
        for each x in A, y in B, associate (x,y).
        A1 z-z B1
        A2 -X- B2
        A3 z-z B3

        :param a: first list
        :param b: second list
        """
        pairs = product(a, b)
        for x, y in pairs:
            x.project(y.area, brain=brain)
            y.project(x.area, brain=brain)

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
