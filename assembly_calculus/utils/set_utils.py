from typing import Hashable, Iterable


# TODO: what happens if there are repetitions in `elements`
def set_hash(elements: Iterable[Hashable]):
    return hash(tuple(sorted(list(elements), key=hash)))


def overlap(A: Iterable, B: Iterable) -> float:
    set_a = set(A)
    set_b = set(B)

    return len(set_a & set_b) / len(set_a | set_b)
