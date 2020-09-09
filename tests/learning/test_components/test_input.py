from unittest import TestCase

from assembly_calculus.brain import BrainPart
from assembly_calculus.learning.errors import MissingArea, MissingStimulus, \
    MaxAttemptsToGenerateStimuliReached
from assembly_calculus.learning.input import InputStimuli, InputBitStimuli, MAX_ATTEMPTS
from assembly_calculus.brain import Brain
from assembly_calculus.brain.connectome import Connectome
from assembly_calculus.brain.components import Area, Stimulus, OutputArea

A: BrainPart
B: BrainPart
C: BrainPart
s0: Stimulus
s1: Stimulus
s2: Stimulus


class InputTests(TestCase):
    def setUp(self) -> None:
        global A, B, C
        global s0, s1, s2

        self.n = 100
        self.k = 10
        self.beta = 0.05
        self.brain = Brain(Connectome(p=0.1))

        A = Area(self.n, self.k, self.beta)
        self.brain.add_area(A)
        B = Area(self.n, self.k, self.beta)
        self.brain.add_area(B)
        C = Area(self.n, self.k, self.beta)
        self.brain.add_area(C)
        
        s0 = Stimulus(self.n, self.beta)
        self.brain.add_stimulus(s0)
        s1 = Stimulus(self.n, self.beta)
        self.brain.add_stimulus(s1)
        s2 = Stimulus(self.n, self.beta)
        self.brain.add_stimulus(s2)

        self.brain.add_area(OutputArea(self.beta))

    def test_input_stimuli_generates_list_of_input_bit_stimuli_objects_from_single_areas(self):
        input_stimuli = InputStimuli(self.brain, self.k, A, B, C)

        self.assertEqual(3, len(input_stimuli))
        for i in range(len(input_stimuli)):
            self.assertIsInstance(input_stimuli[i], InputBitStimuli)

        self.assertListEqual([A], input_stimuli[0].target_areas)
        self.assertListEqual([B], input_stimuli[1].target_areas)
        self.assertListEqual([C], input_stimuli[2].target_areas)

    def test_input_stimuli_generates_list_of_input_bit_stimuli_objects_from_complex_areas(self):
        input_stimuli = InputStimuli(self.brain, self.k, A, B, [A, B], C, [A, C], [A, B, C])

        self.assertEqual(6, len(input_stimuli))
        for i in range(len(input_stimuli)):
            self.assertIsInstance(input_stimuli[i], InputBitStimuli)

        self.assertListEqual([A], input_stimuli[0].target_areas)
        self.assertListEqual([B], input_stimuli[1].target_areas)
        self.assertListEqual([A, B], input_stimuli[2].target_areas)
        self.assertListEqual([C], input_stimuli[3].target_areas)
        self.assertListEqual([A, C], input_stimuli[4].target_areas)
        self.assertListEqual([A, B, C], input_stimuli[5].target_areas)

    def test_input_stimuli_generates_list_of_input_bit_stimuli_objects_from_areas_with_override(self):
        input_stimuli = InputStimuli(self.brain, self.k, A, B, [A, B], override={0: (s0, s1)})

        self.assertEqual(3, len(input_stimuli))
        for i in range(len(input_stimuli)):
            self.assertIsInstance(input_stimuli[i], InputBitStimuli)

        self.assertListEqual([A], input_stimuli[0].target_areas)
        self.assertListEqual([B], input_stimuli[1].target_areas)
        self.assertListEqual([A, B], input_stimuli[2].target_areas)

    def test_input_stimuli_with_non_existent_area_raises(self):
        Non_Existent = Area(self.n, self.k, self.beta)
        self.assertRaises(MissingArea, InputStimuli, self.brain, self.k, A, B, Non_Existent)
        self.assertRaises(MissingArea, InputStimuli, self.brain, self.k, A, [B, Non_Existent])

    def test_input_stimuli_with_non_existent_override_stimulus_raises(self):
        S_Non_Existent = Stimulus(self.n, self.beta)
        self.assertRaises(MissingStimulus, InputStimuli, self.brain, self.k, A, B, override={1: (s0, S_Non_Existent)})

    def test_input_stimuli_with_badly_formatted_override_stimulus_raises(self):
        # Must be a tuple of two, the first value representing a stimulus for 0, the second a stimulus for 1:
        self.assertRaises(ValueError, InputStimuli, self.brain, self.k, A, B, override={1: (s0, s1, s2)})
        self.assertRaises(ValueError, InputStimuli, self.brain, self.k, A, B, override={1: (s0,)})
        self.assertRaises(TypeError, InputStimuli, self.brain, self.k, A, B, override={1: {0: s0, 1: s1}})
        self.assertRaises(TypeError, InputStimuli, self.brain, self.k, A, B, override={1: s0})

    def test_input_stimuli_with_non_area_or_list_of_areas_raises(self):
        self.assertRaises(TypeError, InputStimuli, self.brain, self.k, 'A', 'B')
        self.assertRaises(TypeError, InputStimuli, self.brain, self.k, 0, 1)
        self.assertRaises(TypeError, InputStimuli, self.brain, self.k, (A, B))

        self.assertRaises(TypeError, InputStimuli, self.brain, self.k, ['A', 'B'])
        self.assertRaises(TypeError, InputStimuli, self.brain, self.k, [0, 1])
        self.assertRaises(TypeError, InputStimuli, self.brain, self.k, [A, (B,)])

    def test_input_stimuli_with_different_areas_doesnt_use_the_same_stimuli(self):
        input_stimuli = InputStimuli(self.brain, self.k, A, B, [A, B])

        # input_stimuli[0] is an input bit stimuli, which is of the form: InputBitStimuli(0 = stim0, 1 = stim1)
        # So we want to assure the two input bit stimuli are compiled of different stimuli:
        self.assertNotIn(input_stimuli[0][0], (input_stimuli[1][0], input_stimuli[2][0]))
        self.assertNotIn(input_stimuli[1][0], (input_stimuli[0][0], input_stimuli[2][0]))
        self.assertNotIn(input_stimuli[2][0], (input_stimuli[0][0], input_stimuli[1][0]))

        self.assertNotIn(input_stimuli[0][1], (input_stimuli[1][1], input_stimuli[2][1]))
        self.assertNotIn(input_stimuli[1][1], (input_stimuli[0][1], input_stimuli[2][1]))
        self.assertNotIn(input_stimuli[2][1], (input_stimuli[0][1], input_stimuli[1][1]))

    def test_input_stimuli_with_the_same_area_twice_doesnt_the_use_same_stimuli(self):
        input_stimuli = InputStimuli(self.brain, self.k, A, A)

        # input_stimuli[0] is an input bit stimuli, which is of the form: InputBitStimuli(0 = stim0, 1 = stim1)
        # So we want to assure the two input bit stimuli are compiled of different stimuli:
        self.assertNotEqual(input_stimuli[0][0], input_stimuli[1][0])
        self.assertNotEqual(input_stimuli[0][1], input_stimuli[1][1])

    def test_input_stimuli_with_property_and_get_item_returns_the_same(self):
        input_stimuli = InputStimuli(self.brain, self.k, A, B)

        # input_stimuli[0] is an input bit stimuli, which is of the form: InputBitsStimuli(0 = stim0, 1 = stim1)
        # So we want to assure the two input bit stimuli are compiled of different stimuli:
        self.assertEqual(input_stimuli[0][0], input_stimuli[0].stimulus_for_0)
        self.assertEqual(input_stimuli[0][1], input_stimuli[0].stimulus_for_1)
        self.assertEqual(input_stimuli[1][0], input_stimuli[1].stimulus_for_0)
        self.assertEqual(input_stimuli[1][1], input_stimuli[1].stimulus_for_1)

    def test_cant_create_more_stimuli_than_the_limit(self):
        self.assertIsInstance(InputStimuli(self.brain, self.k, *tuple([A] * MAX_ATTEMPTS)), InputStimuli)
        self.assertRaises(MaxAttemptsToGenerateStimuliReached,
                          InputStimuli, self.brain, self.k, *tuple([A] * (MAX_ATTEMPTS + 1)))
