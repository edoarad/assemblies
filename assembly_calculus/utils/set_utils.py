from typing import Hashable, Iterable


def set_hash(elements: Iterable[Hashable]) -> int:
    return hash(tuple(sorted(set(elements), key=hash)))


def overlap(A: Iterable, B: Iterable) -> float:
    set_a = set(A)
    set_b = set(B)

    return len(set_a & set_b) / len(set_a | set_b)
