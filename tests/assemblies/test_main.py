import gc

from assembly_calculus import BrainRecipe, Area, Stimulus, Assembly, Connectome, bake
from assembly_calculus.utils.brain_utils import fire_many
from assembly_calculus.assemblies.assembly import AssemblyTuple
from utils import protecc_ram


def test_projection():
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
        assembly_af = assembly_a >> area_c
        assembly_bf = assembly_b >> area_c

    for _ in range(25):
        with bake(recipe, 0.1, Connectome, train_repeat=100, effective_repeat=1) as brain:
            for _ in range(brain.repeat):
                fire_many(brain, [assembly_a], area_c)

            assert area_c.active_assembly == assembly_af, "Separate assemblies have merged :("

        gc.collect()


def test_associate():
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
        assembly_af = assembly_a >> area_c
        assembly_bf = assembly_b >> area_c
        AssemblyTuple(assembly_af).associate(AssemblyTuple(assembly_bf))

    for _ in range(25):
        with bake(recipe, 0.1, Connectome, train_repeat=100, effective_repeat=1) as brain:
            old_winners = assembly_b.representative_neuron(brain=brain)
            for _ in range(brain.repeat):
                fire_many(brain, [assembly_a], area_c)

            # test that the winners of assembly_bf have changed
            new_winners = assembly_b.representative_neuron(brain=brain)
            assert old_winners != new_winners, "the associated assembly is unaffected :("

        gc.collect()
