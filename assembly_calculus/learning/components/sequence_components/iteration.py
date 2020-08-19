from collections import defaultdict
from copy import deepcopy
from typing import List, Dict, Tuple, Set

from ....brain import Area, BrainPart

from ..input import InputStimuli
from ..errors import InputStimuliMisused


class Iteration:
	"""
	A single iteration in a sequence, representing a certain number of
	consecutive projections of the same kind (same sources and targets).
	"""

	def __init__(self, subconnectome: Dict[BrainPart, Set[BrainPart]] = None,
				 input_bits_to_areas: Dict[int, List[Area]] = None,
				 consecutive_runs: int = 1):
		"""
		Create a new iteration.
		:param subconnectome: a mapping between a stimulus/area and the areas/output areas it fires to.
		:param input_bits_to_areas: a mapping between a bit in the input and the areas it's stimuli fire to.
		:param consecutive_runs: the number of consecutive times this iteration is sent (for projection) before moving
			to the next iteration.
		"""
		self.subconnectome = subconnectome or {}
		self.input_bits_to_areas = input_bits_to_areas or {}
		self.consecutive_runs = consecutive_runs

	@staticmethod
	def _to_bits(input_value: int, size: int) -> Tuple[int, ...]:
		return tuple(int(bit) for bit in bin(input_value)[2:].zfill(size))

	@staticmethod
	def _union(list1: list, list2: list):
		"""
		Union two lists and remove duplicates.
		"""
		return sorted(set(list1).union(list2))

	def format(self, input_stimuli: InputStimuli, input_value: int) -> dict:
		"""
		Converting the Iteration object into project parameters, using the input definition
		(the InputStimuli object) and the current input value.
		:param input_stimuli: the InputStimuli object which defines the mapping between input bits and pairs of
		stimuli (one for each possible value of the bit).
		:param input_value: the input value as a base 10 integer (for example, for the input 101, use 5).
		"""
		project_parameters = defaultdict(list, deepcopy(self.subconnectome))

		input_value = self._to_bits(input_value, len(input_stimuli))
		for bit_index, areas in self.input_bits_to_areas.items():
			if sorted(input_stimuli[bit_index].target_areas) != sorted(areas):
				raise InputStimuliMisused(bit_index, input_stimuli[bit_index].target_areas, areas)

			stimulus = input_stimuli[bit_index][input_value[bit_index]]
			project_parameters[stimulus] = self._union(project_parameters[stimulus], areas)

		return project_parameters
