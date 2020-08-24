from assembly_calculus import Area, Stimulus
from assembly_calculus.brain import Brain, OutputArea, BrainRecipe, bake, Connectome


class BrainTestUtils(object):

	def __init__(self, default_p=0.1, area_size=1000, default_winners_size=100, stimulus_size=300,
	             beta=0.1):
		self.P: float = default_p
		self.area_size: int = area_size
		self.winners_size: int = default_winners_size
		self.stimulus_size: int = stimulus_size
		self.beta: float = beta
		self._init_data()

	def _init_data(self):
		self.brain: Brain = None
		self._recipe: BrainRecipe = None
		self.output_area: OutputArea = None

	def create_brain(self, number_of_areas, p=0.1, area_size=100, winners_size=10, beta=0.05,
	                 add_output_area=False, number_of_stimuli=0, stimulus_size=100) -> Brain:
		self._init_data()
		areas = [Area(area_size, winners_size, beta) for _ in range(number_of_areas)]
		stimuli = [Stimulus(stimulus_size) for _ in range(number_of_stimuli)]
		output_area = []
		if add_output_area:
			output_area.append(OutputArea(beta=beta))
		self._recipe = BrainRecipe(*areas, *stimuli, *output_area)
		self.brain = bake(self._recipe, p, Connectome, 0, 1)
		return self.brain

	def create_and_stimulate_brain(self, number_of_areas, number_of_stimulated_areas=1,
	                               stimulus_size=None, p=None, area_size=None, winners_size=None, beta=None,
	                               add_output_area=False):
		assert number_of_stimulated_areas <= number_of_areas

		self.create_brain(number_of_areas=number_of_areas, p=p, area_size=area_size,
		                  winners_size=winners_size, beta=beta, add_output_area=add_output_area)
		stimulus = Stimulus(stimulus_size)
		self.brain.add_stimulus(stimulus)
		areas_to_stimulate = self.brain.connectome.areas[:number_of_stimulated_areas]
		self.brain.next_round({stimulus: areas_to_stimulate})
		return self.brain
