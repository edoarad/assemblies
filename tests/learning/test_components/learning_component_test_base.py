from unittest import TestCase

from assembly_calculus.learning.components.input import InputStimuli
from assembly_calculus.learning.components.sequence import LearningSequence
from tests.learning.brain_test_utils import BrainTestUtils


class LearningComponentTestBase(TestCase):

	def setUp(self) -> None:
		utils = BrainTestUtils()
		self.brain = utils.create_brain(number_of_areas=3, number_of_stimuli=4,
		                                area_size=100, winners_size=10, add_output_area=True)

		area_a, area_b, area_c = self.brain.connectome.areas[:3]

		self.input_stimuli = InputStimuli(self.brain, 10, area_a, area_b, verbose=False)
		self.sequence = LearningSequence(self.brain, self.input_stimuli)
		self.sequence.add_iteration(input_bits=[0, 1])
		self.sequence.add_iteration(subconnectome={area_a: [area_c], area_b: [area_c]})
		self.sequence.add_iteration(subconnectome={area_c: [utils.output_area]})
