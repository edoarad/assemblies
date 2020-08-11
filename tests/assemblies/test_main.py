import string
from itertools import product
from typing import Tuple, Dict

from assembly_calculus import BrainRecipe, Area, Stimulus, Assembly, Connectome, bake
from assembly_calculus.utils.brain_utils import fire_many
import pytest

from utils import protecc_ram


@pytest.fixture
def base_recipe() -> Tuple[BrainRecipe, Dict[str, Area], Dict[str, Stimulus], Dict[str, Assembly]]:
    recipe = BrainRecipe()
    stimuli = {f"stim_{c}": Stimulus(100) for c in string.ascii_lowercase[:10]}
    areas = {f"area_{c}": Area(1000, 100) for c in string.ascii_lowercase[:10]}
    recipe.extend(
        *stimuli.values(), *areas.values()
    )
    return recipe, areas, stimuli, {}


@pytest.fixture
def test_recipe(base_recipe) -> Tuple[BrainRecipe, Dict[str, Area], Dict[str, Stimulus], Dict[str, Assembly]]:
    recipe, areas, stimuli, _ = base_recipe
    assemblies = {
        f"assembly_{c}": Assembly((stimuli[f"stim_{c}"],), areas[f"area_{c}"])
        for c in ('a', 'b', 'c', 'd', 'e')
    }
    recipe.extend(*assemblies.values())

    return recipe, areas, stimuli, assemblies


def test_projection(test_recipe):
    protecc_ram(0.75)

    recipe: BrainRecipe
    areas: Dict[str, Area]
    stimuli: Dict[str, Stimulus]
    assemblies: Dict[str, Assembly]
    recipe, areas, stimuli, assemblies = test_recipe

    assembly_a: Assembly
    assembly_b: Assembly
    assembly_a, assembly_b = (assemblies[f"assembly_{c}"] for c in ('a', 'b'))

    area_f: Area
    area_f = areas["area_f"]

    with recipe:
        assembly_af = assembly_a >> area_f
        assembly_bf = assembly_b >> area_f

    # TODO: (Tomer, from Yonatan) check this makes sense
    with bake(recipe, 0.1, Connectome) as brain:
        for _ in range(brain.repeat):
            fire_many(brain, [assembly_a], area_f)

        assert Assembly.read(area_f, brain=brain) == assembly_af
