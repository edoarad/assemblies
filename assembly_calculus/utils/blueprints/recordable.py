from functools import wraps
from typing import Optional, Callable

from .recording import Recording
from ..bindable import Bindable


recording_bindable = Bindable('recording')


def attach_recording(cls):
    return recording_bindable.cls(cls)


def record_method(recording_resolver: Callable = recording_bindable.resolver('recording'),
                  execute_anyway: bool = False):
    def decorator(function):
        @wraps(function)
        def wrapper(*args, **kwargs):
            recording: Optional[Recording] = recording_resolver(*args, **kwargs)
            if recording is not None:
                # If recording parameter was provided, record function
                recording.append(function, args, kwargs)
                if not execute_anyway:
                    # Check if function should be executed anyway
                    return

            return function(*args, **kwargs)

        return wrapper

    return decorator
