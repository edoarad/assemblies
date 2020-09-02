from inspect import signature as _signature, Signature


def signature(func, use_original=False) -> Signature:
    """Better signature function"""
    return _signature(func) if use_original else (getattr(func, '__signature__', None) or _signature(func))
