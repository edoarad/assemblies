This package studies the brain's ability to learn binary functions from a given data set, and checks whether the brain succeeds in it.


## Simulation example

A simulation begins by defining the area and the brain: (assume the parameters were set in advance)
```pycon
>>> from assembly_calculus.brain import Area, OutputArea, Stimulus, Brain
>>> from assembly_calculus.brain.connectome import Connectome
>>> from assembly_calculus.learning.input import InputStimuli
>>> from assembly_calculus.learning.sequence import LearningSequence
>>> from assembly_calculus.learning.data_set.constructors import create_training_set_from_list, create_test_set_from_list
>>> from assembly_calculus.learning.model import LearningModel
>>> n = 1000; k = 31; beta = 0.05; p = 0.1
>>> A = Area(n, k, beta)
>>> B = Area(n, k, beta)
>>> output = OutputArea(beta)
>>> output_values = [0, 1, 0, 1] # binary function to learn
>>> brain = Brain(Connectome(p))
>>> stimuli = Stimulus(n, beta)
>>> brain.add_stimulus(stimuli)
>>> for area in (A, B):
        brain.add_area(area)
>>> brain.add_area(output)
```
Create a mapping between bits of input to actve stimuli using an instance of `InputStimuli`.
```pycon
>>> input_stimuli = InputStimuli(brain, k, A, B)
```
Create sequence of projections for the model, using the possible arguments: `input_bits`, `subconnectome`, `consecutive_runs`.
Use an instance of `LearningSequence`.
```pycon
>>> sequence = LearningSequence(brain, input_stimuli)
>>> sequence.add_iteration(input_bits = [0, 1], subconnectome= {A : [B], B: [B]}, consecutive_runs = 10)
>>> sequence.add_iteration(subconnectome={B: [output]})
```
and display them with:
```pycon
>>> sequence.display_connections_graph()
```
Create a training set:
```pycon
>>> training_set = create_training_set_from_list(data_set_return_values = output_values,    training_set_length = 30, noise_probability = 0)
```
Create the test set to test the inputs:
```pycon
>>> test_set = create_test_set_from_list(output_values)
```
Create the model and train it:
```pycon
>>> model = LearningModel(brain = brain, sequence = sequence, input_stimuli = input_stimuli)
>>> model.train_model(training_set=training_set, number_of_sequence_cycles=1)
>>> test_results = model.test_model(test_set)
```
And print the results:
```pycon
>>> print(
        f"Finished testing the trained model - results:\n"
        f"Accuracy: {test_results.accuracy}\n"
        f"Precision: {test_results.precision}\n"
        f"Recall: {test_results.accuracy}\n")
```