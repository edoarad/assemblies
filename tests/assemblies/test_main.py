import gc

from assembly_calculus import BrainRecipe, Area, Stimulus, Assembly, Connectome, bake
from assembly_calculus.utils import overlap
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
        assembly_ac = assembly_a >> area_c
        assembly_bc = assembly_b >> area_c

    for _ in range(25):
        with bake(recipe, 0.1, Connectome, train_repeat=100, effective_repeat=3) as brain:
            assembly_a >> area_c

            assert area_c.active_assembly == assembly_ac, "Separate assemblies have merged :("

        gc.collect()


def test_associate():
    def average_overlap(brain_recipe, times=25):
        s = 0
        for _ in range(times):
            with bake(brain_recipe, 0.1, Connectome, train_repeat=100, effective_repeat=3) as brain:
                winners_from_a = set()
                for _ in range(5):
                    assembly_a >> area_c
                    winners_from_a.update(area_c.winners)

                winners_from_b = set()
                for _ in range(5):
                    assembly_b >> area_c
                    winners_from_b.update(area_c.winners)

                s += overlap(winners_from_a, winners_from_b)

            gc.collect()

        return s / times

    # TODO: Avoid code duplication?
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
        assembly_ac = assembly_a >> area_c
        assembly_bc = assembly_b >> area_c

    no_associate = average_overlap(recipe)

    with recipe:
        (assembly_ac | ...).associate(assembly_bc | ...)

    with_associate = average_overlap(recipe)

    print(with_associate, no_associate)
    assert with_associate > no_associate * 1.5, "Associate does not associate"


def test_merge():
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
        (assembly_a | assembly_b) >> area_c

    for _ in range(25):
        with bake(recipe, 0.1, Connectome, train_repeat=100, effective_repeat=3) as brain:
            assembly_a >> area_c
            brain.next_round(subconnectome={area_c: [area_b]}, replace=True, iterations=brain.repeat)

            assert overlap(assembly_b.representative_neurons, area_b.winners) > 0.75, \
                "Assemblies haven't formed bi-directional links"

        gc.collect()
