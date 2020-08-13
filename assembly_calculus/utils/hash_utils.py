from typing import Hashable, Iterable


def set_hash(elements: Iterable[Hashable]):
    return hash(tuple(sorted(list(elements), key=hash)))
