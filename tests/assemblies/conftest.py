from assembly_calculus import BrainRecipe, Area, Stimulus, Assembly
from pytest import fixture

AREA_SIZE = 1000
STIMULUS_SIZE = int(AREA_SIZE ** 0.5)


@fixture
def stim_a():
    return Stimulus(STIMULUS_SIZE)


@fixture
def stim_b():
    return Stimulus(STIMULUS_SIZE)


@fixture
def area_a():
    return Area(AREA_SIZE)


@fixture
def area_b():
    return Area(AREA_SIZE)


@fixture
def area_c():
    return Area(AREA_SIZE)


@fixture
def assembly_a(stim_a, area_a):
    return Assembly([stim_a], area_a)


@fixture
def assembly_b(stim_b, area_b):
    return Assembly([stim_b], area_b)

@fixture
def assembly_c(assembly_a, assembly_b, area_c):
    return Assembly([assembly_a, assembly_b], area_c)

@fixture
def recipe(stim_a, stim_b, area_a, area_b, area_c, assembly_a, assembly_b):
    return BrainRecipe(
        stim_a, stim_b, area_a, area_b, area_c, assembly_a, assembly_b
    )

