from typing import List, Dict, Optional, Set

from assembly_calculus import Stimulus
from assembly_calculus.brain import Area, Brain, BrainPart, OutputArea
from assembly_calculus.learning.components.errors import MissingArea, SequenceRunNotInitializedOrInMidRun, \
	SequenceFinalizationError, \
	MissingStimulus
from assembly_calculus.learning.components.input import InputStimuli
from assembly_calculus.learning.components.sequence_components.connections_graph import ConnectionsGraph
from assembly_calculus.learning.components.sequence_components.iteration import Iteration
from assembly_calculus.learning.components.sequence_components.iteration_configuration import IterationConfiguration

NODE_TYPE = {Stimulus: 'stimulus', int: 'input-bit', Area: 'area', OutputArea: 'output'}


class LearningSequence:
	"""
	The learning sequence represent a sequence of projections used for the
	learning process (either at the training or the test stage). it defines the
	order of projections, and the number of their repetitions.

	Note that the sequence is validated to ensure that there is only one
	output, and that each input is eventually connected to the output area.

	Also, after the sequence is created, it allows the user to generate a
	connections graph representing the projections in the sequence.
	"""

	def __init__(self, brain: Brain, input_stimuli: InputStimuli):
		"""
		Create a new empty sequence.

		Once created, you can add iterations to it, and eventually finalize it
		to indicate it is ready and should be validated (and that no new
		iterations will added).

		:param brain: the brain object
		:param input_stimuli: the input stimuli object defining the mapping
		between input bits and their representing pair of stimuli.
		"""
		self._brain = brain
		self._input_stimuli = input_stimuli

		# Representing the given sequence as a graph, for connectivity checking
		self._connections_graph = ConnectionsGraph()

		self._iterations: List[Iteration] = []
		self._configuration: Optional[IterationConfiguration] = None

		self._output_area: Optional[OutputArea] = None
		self._finalized: bool = False

		self._seq = 0
		self._serials = {}

	@property
	def number_of_iterations(self) -> int:
		"""
		Returns the number of iterations currently in the sequence.
		"""
		return len(self._iterations)

	@property
	def output_area(self) -> OutputArea:
		"""
		Returns the output area of the sequence.
		"""
		return self._output_area

	def __iter__(self):
		if self._configuration is None or self._configuration.is_in_mid_run:
			raise SequenceRunNotInitializedOrInMidRun()
		return self

	def __next__(self):
		if self._configuration is None:
			raise SequenceRunNotInitializedOrInMidRun()

		self._configuration.current_run += 1
		if self._configuration.current_run >= self._iterations[self._configuration.current_iter].consecutive_runs:
			# Moving to the next iteration
			self._configuration.current_run = 0
			self._configuration.current_iter += 1

			if self._configuration.current_iter >= self.number_of_iterations:
				# Moving to the next cycle
				self._configuration.current_cycle += 1
				self._configuration.current_iter = 0

				if self._configuration.current_cycle >= self._configuration.number_of_cycles:
					# Number of cycles exceeded
					self._configuration.is_in_mid_run = False
					raise StopIteration()

		current_iteration = self._iterations[self._configuration.current_iter]

		# Indicate that the iteration started, and must exhausted or reset
		# (by initializing a new run) before starting another iteration.
		self._configuration.is_in_mid_run = True

		return current_iteration

	def finalize_sequence(self) -> None:
		"""
		finalizing the sequence before initial running. The sequence cannot be edited after that
		"""
		output_area_seq = self._connections_graph.verify_single_output_area()
		self._output_area = [key for key, value in self._serials.items() if value == output_area_seq][0]
		self._connections_graph.verify_inputs_are_connected_to_output()

	def initialize_run(self, number_of_cycles=float('inf')) -> None:
		"""
		Setting up the running of the sequence iterations
		:param number_of_cycles: the number of full cycles (of all defined iterations) that should be run consecutively
		"""
		if not self._finalized:
			self.finalize_sequence()
		self._configuration = IterationConfiguration(number_of_cycles=number_of_cycles)

	def _serialize_part(self, part: BrainPart):
		"""
		Give each part a serial number for the graph
		"""
		#if not isinstance(part, BrainPart):
		if type(part) in BrainPart.__args__:
			return part

		if part in self._serials:
			return self._serials[part]

		self._serials[part] = self._seq
		self._seq += 1

		return self._seq

	def _verify_stimulus(self, stimulus: Stimulus) -> None:
		"""
		Raising an error if the given stimulus doesn't exist in the brain
		:param stimulus: the stimulus name
		:return: the stimulus object (or exception, on missing)
		"""
		if stimulus not in self._brain.connectome.stimuli:
			raise MissingStimulus(repr(stimulus))

	def _verify_area(self, area: Area) -> None:
		"""
		:param area: the area name
		:return: the area/output area object (or exception, on missing)
		"""
		if area not in self._brain.connectome.areas:
			raise MissingArea(repr(area))

	def _verify_brain_part(self, part: BrainPart) -> None:
		if isinstance(part, Stimulus):
			self._verify_stimulus(part)
		self._verify_area(part)

	def _validate_and_add_connections(self, mapping: Dict,
	                                  consecutive_runs: int):
		"""
		Validate a source-to-targets mapping for the iteration's projections and add the relevant
		connections to the connections graph.
		:param mapping: The source to targets mapping for the iteration projections.
		:param consecutive_runs: Number of consecutive runs of the iteration, to be used as
		the weight of the connections.
		"""

		for source, target_areas in mapping.items():
			#if isinstance(source, BrainPart):
			if type(source) in BrainPart.__args__:
				self._verify_brain_part(source)
			source_node = f'{NODE_TYPE[type(source)]}-{self._serialize_part(source)}'

			for target_area in target_areas:
				#if isinstance(target_area, BrainPart):
				if type(target_area) in BrainPart.__args__:
					self._verify_brain_part(target_area)
				area_node = f'{NODE_TYPE[type(target_area)]}-{self._serialize_part(target_area)}'

				self._connections_graph.add_connection(source_node, area_node, consecutive_runs,
				                                       self.number_of_iterations)

	def _process_input_bits(self, input_bits: List[int]) -> Dict[int, List[Area]]:
		"""
		Generate a mapping of input bits to their connected areas from the given input bits.
		:param input_bits: a list of bits in the input that should fire to their defined areas
		:return: a mapping between input bit and it's areas as defined in the input stimuli mapping.
		"""
		return {input_bit: self._input_stimuli[input_bit].target_areas
		        for input_bit in input_bits}

	def add_iteration(self, subconnectome: Dict[BrainPart, Set[BrainPart]] = {},
	                  input_bits: List[int] = None,
	                  consecutive_runs: int = 1) -> None:
		"""
		Adding an iteration to the learning sequence, consisting of firing stimuli/areas and fired-at areas/output areas
		:param subconnectome: a mapping between a stimulus/area and the areas/output areas it fires to
		:param input_bits: a list of bits in the input that should fire to their defined areas
		:param consecutive_runs: the number of consecutive times this iteration is sent (for projection) before moving
			to the next iteration
		"""
		if self._finalized:
			raise SequenceFinalizationError()

		if subconnectome:
			self._validate_and_add_connections(subconnectome, consecutive_runs)

		if input_bits:
			input_bits_to_areas = self._process_input_bits(input_bits)
			self._validate_and_add_connections(input_bits_to_areas, consecutive_runs)
		else:
			input_bits_to_areas = None

		new_iteration = Iteration(subconnectome=subconnectome,
		                          input_bits_to_areas=input_bits_to_areas,
		                          consecutive_runs=consecutive_runs)
		self._iterations.append(new_iteration)

	def display_connections_graph(self):
		self._connections_graph.display()
