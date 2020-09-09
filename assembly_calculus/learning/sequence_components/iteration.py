from typing import Dict, Tuple, Set

from assembly_calculus import Area, BrainPart

from assembly_calculus.learning.input import InputStimuli


class Iteration:
	"""
	A single iteration in a sequence, representing a certain number of
	consecutive projections of the same kind (same sources and targets).
	"""

	def __init__(self, subconnectome: Dict[BrainPart, Set[BrainPart]] = None,
	             input_bits_to_areas: Dict[int, Set[Area]] = None,
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
	def _union(list1: set, list2: set) -> set:
		"""
		Union two sets and remove duplicates.
		"""
		return set(list1).union(list2)

	def format(self, input_stimuli: InputStimuli, input_value: int) -> dict:
		"""
		Converting the Iteration object into next_round parameters, using the input definition
		(the InputStimuli object) and the current input value.
		:param input_stimuli: the InputStimuli object which defines the mapping between input bits and pairs of
		stimuli (one for each possible value of the bit).
		:param input_value: the input value as a base 10 integer (for example, for the input 101, use 5).
		"""
		subconnectome = {}
		subconnectome.update(self.subconnectome)
		input_value = self._to_bits(input_value, len(input_stimuli))
		for bit_index, areas in self.input_bits_to_areas.items():
			stimulus = input_stimuli[bit_index][input_value[bit_index]]
			if stimulus in subconnectome:
				subconnectome[stimulus] = self._union(subconnectome[stimulus], areas)
			else:
				subconnectome[stimulus] = set(areas)

		return subconnectome
