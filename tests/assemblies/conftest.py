from assembly_calculus import BrainRecipe, Area, Stimulus, Assembly
from pytest import fixture


@fixture
def stim_a():
    return Stimulus(31)


@fixture
def stim_b():
    return Stimulus(31)


@fixture
def area_a():
    return Area(1000)


@fixture
def area_b():
    return Area(1000)


@fixture
def area_c():
    return Area(1000)


@fixture
def assembly_a(stim_a, area_a):
    return Assembly([stim_a], area_a)


@fixture
def assembly_b(stim_b, area_b):
    return Assembly([stim_b], area_b)


@fixture
def recipe(stim_a, stim_b, area_a, area_b, area_c, assembly_a, assembly_b):
    return BrainRecipe(
        stim_a, stim_b, area_a, area_b, area_c, assembly_a, assembly_b
    )
