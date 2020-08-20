from contextlib import contextmanager
from typing import Union, List

from assembly_calculus import Area, Stimulus
from assembly_calculus.brain import Brain, OutputArea, BrainRecipe
from assembly_calculus.utils import value_or_default


class BrainTestUtils(object):

    def __init__(self, default_p=0.1, area_size=1000, default_winners_size=100, stimulus_size=300,
                 beta=0.1):
        self.P: float = default_p
        self.area_size: int = area_size
        self.winners_size: int = default_winners_size
        self.stimulus_size: int = stimulus_size
        self.beta: float = beta
        self._init_data()

    def __getattr__(self, item):
        if item.startswith('area'):
            area_index = int(item[4:])
            return self.brain.areas[self._areas[area_index]]
        if item.startswith('stim'):
            stimulus_index = int(item[4:])
            return self.brain.stimuli[self._stimuli[stimulus_index]]

    def _init_data(self):
        self.brain: Brain = None
        # Area names, ordered by their creation time
        self._areas: List[Area] = []
        self.output_area: OutputArea = None
        # Stimuli names, order by their creation time
        self._stimuli: List[Stimulus] = []

    def create_brain(self, number_of_areas, p=None, area_size=None, winners_size=None, beta=None,
                     add_output_area=False, number_of_stimuli=0, stimulus_size=None) -> Union[LazyBrain, NonLazyBrain]:
        self._init_data()
        areas = [Area(area_size, winners_size, beta) for _ in range(number_of_areas)]
        stimuli = [Stimulus(stimulus_size) for _ in range(number_of_stimuli)]
        output_area = []
        if add_output_area:
            output_area.append(OutputArea(area_size, beta))

        BrainRecipe(*areas, *stimuli, *output_area)


    def create_and_stimulate_brain(self, number_of_areas, number_of_stimulated_areas=1,
                                   stimulus_size=None, p=None, area_size=None, winners_size=None, beta=None,
                                   add_output_area=False):
        assert number_of_stimulated_areas <= number_of_areas

        self.create_brain(number_of_areas=number_of_areas, p=p, area_size=area_size,
                          winners_size=winners_size, beta=beta, add_output_area=add_output_area)

        stimulus = Stimulus(stimulus_size)
        self._add_stimulus(stimulus)
        areas_to_stimulate = self._areas[:number_of_stimulated_areas]
        self.brain.project(area_to_area={}, stim_to_area={'stimulus': areas_to_stimulate})
        return self.brain

    def _add_area(self, name, n, k, beta):
        self.brain.add_area(name=name,
                            n=value_or_default(n, self.area_size),
                            k=value_or_default(k, self.winners_size),
                            beta=value_or_default(beta, self.beta))
        self._areas.append(name)

    def _add_stimulus(self, name, k):
        self.brain.add_stimulus(name=name,
                                k=value_or_default(k, self.stimulus_size))
        self.brain.stimuli[name].name = name
        self._stimuli.append(name)

    @staticmethod
    @contextmanager
    def change_output_area_settings(n=None, k=None, beta=None):
        original_settings = (OutputArea.n, OutputArea.k, OutputArea.beta)
        if n is not None:
            OutputArea.n = n
        if k is not None:
            OutputArea.k = k
        if beta is not None:
            OutputArea.beta = beta
        yield
        OutputArea.n, OutputArea.k, OutputArea.beta = original_settings
