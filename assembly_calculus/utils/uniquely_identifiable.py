from __future__ import annotations
from typing import Dict, Any
from weakref import ref


class NoInitMeta(type):
    def __call__(cls, *args, **kwargs):
        obj = cls.__new__(cls, *args, **kwargs)
        if not getattr(obj, '_done', False):
            obj.__init__(*args, **kwargs)

        return obj


class UniquelyIdentifiable(metaclass=NoInitMeta):
    """
    This class represents objects that are uniquely identifiable, objects that should be identified by instance
    and not by their properties.
    """
    custom_uids: Dict[Any, 'ref[UniquelyIdentifiable]'] = {}

    def __new__(cls, *args, uid: Any = None, **kwargs):
        if uid is not None:
            if uid not in UniquelyIdentifiable.custom_uids or UniquelyIdentifiable.custom_uids[uid]() is None:
                obj = super(UniquelyIdentifiable, cls).__new__(cls)
                UniquelyIdentifiable.custom_uids[uid] = ref(obj)

            obj = UniquelyIdentifiable.custom_uids[uid]()
            setattr(obj, '_uid', uid)
            return obj
        else:
            return super(UniquelyIdentifiable, cls).__new__(cls)

    def __init__(self, *args, **kwargs):
        self._done = True

    def __del__(self):
        uid = getattr(self, '_uid', None)
        if uid is not None:
            del UniquelyIdentifiable.custom_uids[uid]
