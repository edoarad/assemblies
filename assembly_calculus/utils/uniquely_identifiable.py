from __future__ import annotations
from typing import Dict, Any
from weakref import ref

from assembly_calculus.utils.instance_name import RememberInitName


class NoInitMeta(type):
    """
    Metaclass that supports skipping init on creation of instance if _done flag is on
    """
    def __call__(cls, *args, **kwargs):
        obj = cls.__new__(cls, *args, **kwargs)
        if not getattr(obj, '_done', False):
            obj.__init__(*args, **kwargs)

        return obj


# TODO: this class is used in many places, but `uid` is given only in Assembly.
# TODO: what is the purpose of usage in the other places?
# Response: UniquelyIdentifiable implements hash as id, and allows comparison.
# This is the default implementation but it is better to be explicit
# TODO 2: document the intended use or give a good use example
class UniquelyIdentifiable(RememberInitName, metaclass=NoInitMeta):
    """
    This class represents objects that are uniquely identifiable, objects that should be identified by instance
    and not by their properties.
    Which actually means - id(UniquelyIdentifiable(uid=1) == id(UniquelyIdentifiable(uid=1))
    It can give objects with the same uid (the argument for constructor) the same id.
    Note: the uid would probably be different from the real id.
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
        super(UniquelyIdentifiable, self).__init__()
        self._done = True

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __del__(self):
        uid = getattr(self, '_uid', None)
        if uid is not None:
            del UniquelyIdentifiable.custom_uids[uid]

    def __str__(self):
        return "%s(name=%s)" % (self.__class__.__name__, self.instance_name)
