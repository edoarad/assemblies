from unittest import TestCase

from assembly_calculus.brain import Area, OutputArea
from assembly_calculus.learning.input import InputStimuli
from assembly_calculus.learning.components.sequence import LearningSequence
from tests.learning.brain_test_utils import BrainTestUtils


class LearningComponentTestBase(TestCase):

	def setUp(self) -> None:
		utils = BrainTestUtils()
		self.brain = utils.create_brain(
			p=0.1, beta=0.05, number_of_areas=3, number_of_stimuli=4,
			area_size=100, stimulus_size=100,  winners_size=10, add_output_area=True
		)
		areas = [part for part in self.brain.connectome.areas if type(part) == Area]
		output_area = [part for part in self.brain.connectome.areas if type(part) == OutputArea][0]
		self.input_stimuli = InputStimuli(self.brain, 10, areas[0], areas[1], verbose=False)
		self.sequence = LearningSequence(self.brain, self.input_stimuli)
		self.sequence.add_iteration(input_bits=[0, 1])
		self.sequence.add_iteration(subconnectome={areas[0]: {areas[2]}, areas[1]: {areas[2]}})
		self.sequence.add_iteration(subconnectome={areas[2]: {output_area}})
