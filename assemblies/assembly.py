from __future__ import annotations  # TODO: remove this allover, we are using python 3
# Response: No, this allows future declaration and simpler typing
from .read_driver import ReadDriver  # TODO: It shouldn't depend on directory structure.
from utils.blueprints.recordable import Recordable
from utils.implicit_resolution import ImplicitResolution
from utils.bindable import Bindable
from brain.components import Stimulus, Area, UniquelyIdentifiable
from typing import Iterable, Union, Tuple, TYPE_CHECKING, Set, Optional, Dict
from itertools import product

if TYPE_CHECKING:  # TODO: this is not needed. It's better to always import them.
    # Response: This is to avoid cyclic imports...
    from brain import Brain
    from brain.brain_recipe import BrainRecipe

"""
standard python 3.8 typing
Projectable is an umbrella type for regular assemblies 
and top level assemblies with no parents (i.e stimuli)
"""
Projectable = Union['Assembly', Stimulus]

bound_assembly_tuple = ImplicitResolution(lambda instance, name:
                                          Bindable.implicitly_resolve_many(instance.assemblies, name, True), 'brain')


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

    @bound_assembly_tuple
    def merge(self, area: Area, *, brain: Brain = None):
        return Assembly._merge(self.assemblies, area, brain=brain)

    def associate(self, other: AssemblyTuple, *, brain: Brain = None):
        # Hack for resolution
        return AssemblyTuple(*(self.assemblies + other.assemblies))._associate(self, other, brain=brain)

    # Yoantan: Please document and rename better
    @bound_assembly_tuple
    def _associate(self, left: AssemblyTuple, right: AssemblyTuple, *, brain: Brain = None):
        assert all(ass in self.assemblies for ass in left) and all(
            ass in self.assemblies for ass in right), "Inner check, never should be called by user"
        return Assembly._associate(left.assemblies, right.assemblies, brain=brain)

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

    def __init__(self, parents: Iterable[Projectable], area: Area,
                 initial_recipes: Iterable[BrainRecipe] = None, reader: str = 'default'):
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
        self.reader = ReadDriver(reader)
        self.appears_in: Set[BrainRecipe] = set(initial_recipes or [])
        for recipe in self.appears_in:
            recipe.append(self)

    # TODO: this name is not indicative. Perhaps change to something like to_representative_neuron_subset.
    # Response: We will add this in the documentation, to_representative_neuron_subset is simply too long
    # TODO: reader.read is _very_ confusing with Assembly.read. Rename reader.
    def identify(self, preserve_brain=False, *, brain: Brain) -> Set[int, ...]:
        return set(self.reader.read(self, brain, preserve_brain=preserve_brain))

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
    # Response: We have no current use case, but we had ideas in the past that involve this so we will keep this,
    #           No need to test as there is no logic...
    def trigger_reader_update_hook(self, *, brain: Brain):
        """
        some read_drivers may want to be notified on certain changes
        we support this by calling this private function in key places (like project)
        which then triggers the hook in the reader (if it implements it)
        :param brain:
        :return:
        """
        self.reader.update_hook(brain, self)

    # TODO: throughout bindable classes, users might error and give the brain parameter even if the object is binded.
    #       Is this a problem? can you help the user not make any mistakes?
    # Response: No. This is a feature by design, and allows regular code to ignore explicit binding
    #           (needed to function properly). Binding is very explicit and an inexperienced user should never
    #           pass the brain parameter explicitly.
    # TODO: add option to manually change the assemblies' recipes
    # Response: This does not make sense. To avoid bugs, this assembly is added to all recipes it can be used in.
    def project(self, area: Area, *, brain: Brain = None, iterations: Optional[int] = None) -> Assembly:
        """
        Projects an assembly into an area.

        :param brain: the brain in which the projection happens
        :param area: the area in which the new assembly is going to be created
        :returns: resulting projected assembly
        """
        # TODO 2: more verification? area is not None, area is inside the brain
        # Response: isinstance(area, Area) => area is not None
        # TODO 3: check any edge cases in the dependency between area and brain
        if not isinstance(area, Area):
            raise TypeError("Project target must be an Area")
        projected_assembly: Assembly = Assembly([self], area, initial_recipes=self.appears_in)
        if brain is not None:
            neurons = self.identify(brain=brain)

            brain.connectome.winners[self.area] = list(neurons)

            # Replace=True for better performance
            # TODO: is it only for better performance? it seems to affect correctness
            # Response: Yes it also affects operation, but you & I (Yonatan) discussed this
            #           And this is the implementation you requested.
            # TODO: *** WRONG LOGIC *** - add mapping area->area
            brain.next_round({self.area: [area]}, replace=True, iterations=iterations or brain.repeat)

            projected_assembly.trigger_reader_update_hook(brain=brain)

        # TODO: calling `bind_like` manually is error-prone because someone can forget it. can you make a decorator or a more automated way to do it?
        # Response: No, this is the standard path defined in the Bindable API.
        #           No user should any be in the situation to call this function manually.
        projected_assembly.bind_like(self)
        return projected_assembly

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

        # TODO 2: check documentation of `intersection` - it seems to be an instance method that works here by chance!
        # Response: Please refer to the official documentation
        #           - https://docs.python.org/3/library/stdtypes.html#frozenset.intersection
        #
        #           They probably forgot to update this in the source code.
        merged_assembly: Assembly = Assembly(assemblies, area,
                                             initial_recipes=set.intersection(*[x.appears_in for x in assemblies]))
        # TODO: this is actually a way to check if we're in "binded" or "non binded" state.
        # TODO: can you think of a nicer way to do that?
        # TODO: otherwise it seems like a big block of code inside the function that sometimes happens and sometimes not. it is error-prone
        # Response: No. This is not a way to check if we are bound or not, this serves as a way to perform syntactic
        #           assemblies operations in order to define new assemblies without performing operations.
        #           This integrates in the recipe-ecosystem.
        if brain is not None:
            # create a mapping from the areas to the neurons we want to fire
            area_neuron_mapping = {ass.area: [] for ass in assemblies}
            for ass in assemblies:
                # TODO: What happens if we merge assemblies that are already in the same area?
                area_neuron_mapping[ass.area] = list(
                    ass.identify(brain=brain))

            # update winners for relevant areas in the connectome
            for source in area_neuron_mapping.keys():
                brain.connectome.winners[source] = area_neuron_mapping[source]

            # Replace=True for better performance
            brain.next_round(subconnectome={source: [area] for source in area_neuron_mapping}, replace=True,
                             iterations=brain.repeat)

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
    # Response: True, but this makes it more readable.
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
