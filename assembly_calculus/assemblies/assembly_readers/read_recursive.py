from __future__ import annotations
from ..reader import Reader
from ...utils.brain_utils import fire_many, revert_changes
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..assembly import Assembly
    from ...brain import Brain


class ReadRecursive(Reader):
    """
    A class representing a reader that obtains information about an assembly using the 'read' method.
    The method works by recursively firing areas from the top of the parent tree of the assembly,
    and examining which neurons were fired.
    Note: This is the default read driver.
    """

    @staticmethod
    def read(assembly: Assembly, preserve_brain: bool = False, *, brain: Brain):
        # TODO: verify this reader makes sense
        """
        Read the winners from given assembly in given brain recursively using fire_many
        and return the result.
        :param assembly: the assembly object
        :param preserve_brain: a boolean representing whether we want to change the brain state or not
        :param brain: the brain object
        :return: the winners as read from the area that we've fired up
        """
        changed_areas = fire_many(brain, assembly.parents, assembly.area, preserve_brain)
        read_value = brain.winners[assembly.area]
        if preserve_brain:
            revert_changes(brain, changed_areas)
        return read_value
