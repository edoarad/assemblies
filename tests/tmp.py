import gc

from assembly_calculus import BrainRecipe, Area, Stimulus, Assembly, Connectome, bake
from assembly_calculus.assemblies.utils import activate
from assembly_calculus.utils import overlap

from utils import protecc_ram


protecc_ram(0.75)

stim_a = Stimulus(100)
stim_b = Stimulus(100)
area_a = Area(1000)
area_b = Area(1000)
area_c = Area(1000)
assembly_a = Assembly([stim_a], area_a)
assembly_b = Assembly([stim_b], area_b)
recipe: BrainRecipe = BrainRecipe(
    stim_a, stim_b, area_a, area_b, assembly_a, assembly_b
)

with recipe:
    assembly_c = assembly_a.reciprocal_project(area_c)

for _ in range(25):
    with bake(recipe, 0.1, Connectome, train_repeat=100, effective_repeat=3) as brain:
        activate((assembly_c, ), brain=brain)
        brain.next_round({area_c: [area_a]}, replace=True)
        coverlap = overlap(assembly_a.representative_neurons, area_a.winners)
        print(f"Overlap: {coverlap * 100}%")

    gc.collect()
