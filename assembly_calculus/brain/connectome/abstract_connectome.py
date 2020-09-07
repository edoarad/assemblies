import numpy as np
from collections import defaultdict
from abc import ABCMeta, abstractmethod
from typing import Dict, List, Tuple, Set

from assembly_calculus.brain.components import BrainPart, Area, Stimulus, Connection


class AbstractConnectome(metaclass=ABCMeta):
    """
    Represent the graph of connections between areas and stimuli of the brain.
    This is a generic abstract class which offer a good infrastructure for building new models of connectome.
    You should implement some of the parts which are left for private case. For example when and how the connectome
    should be initialized, how the connections are represented.

    Attributes:
        areas: List of area objects in the connectome
        stimuli: List of stimulus objects in the connectome
        connections: Dictionary from tuples of BrainPart(Stimulus/Area) and Area to some object which
        represent the connection (e.g. numpy matrix). Each connection is held ObjectProxy which will
        make the connection.
        to be saved by reference. (This makes the get_subconnectome routine much easier to implement)
        winners: The winners of each area in the current state.
        support: The past-winners of each area until now.
    """

    def __init__(self, p, areas=None, stimuli=None):
        self.areas: List[Area] = []
        self.stimuli: List[Stimulus] = []
        self.winners: Dict[Area, np.ndarray] = defaultdict(lambda: [])
        self.support: Dict[Area, Set[int]] = defaultdict(lambda: set())
        self.connections: Dict[Tuple[BrainPart, Area], Connection] = {}
        self.p = p

        if areas:
            self.areas = areas
        if stimuli:
            self.stimuli = stimuli

    def add_area(self, area: Area):
        self.areas.append(area)

    def add_stimulus(self, stimulus: Stimulus):
        self.stimuli.append(stimulus)

    def __repr__(self):
        return f'{self.__class__.__name__} with {len(self.areas)} areas, and {len(self.stimuli)} stimuli'

    def fire(self, connections: Dict[BrainPart, List[Area]], *, override_winners: Dict[Area, List[int]] = None,
             enable_plasticity=True):
        """
        :param connections: The connections on which you want to perform the project
        :param override_winners: if passed, will override the winners in the Area with the value
        :param enable_plasticity: if True, update the connectomes
        """
