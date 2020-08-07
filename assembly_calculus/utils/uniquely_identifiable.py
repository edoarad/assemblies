from uuid import UUID, uuid4


class UniquelyIdentifiable:
    """
    This class represents objects that are uniquely identifiable, objects that should be identified by instance
    and not by their properties.
    """
    hist = {}

    def __init__(self, uid=None):
        self._uid: UUID = uuid4()
        if uid is not None and uid in UniquelyIdentifiable.hist:
            self._uid = UniquelyIdentifiable.hist[uid]
        elif uid is not None:
            UniquelyIdentifiable.hist[uid] = self._uid

    def __hash__(self):
        return hash(self._uid)

    def __eq__(self, other):
        # TODO: make more readable
        # Removed the type check as it doesn't really matter, hope it is better now
        # TODO 2: avoid edge case in which _uid and getattr are both None
        # Response: I think this can never happen, since we always give it a non-None value in the constructor

        return self._uid == getattr(other, '_uid', None)
