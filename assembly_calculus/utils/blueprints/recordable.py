from functools import wraps
from typing import Optional, Callable

from assembly_calculus.utils.blueprints.recording import Recording
from assembly_calculus.utils.bindable import Bindable

recording_bindable = Bindable('recording')


def attach_recording(cls):
	"""Add a bindable recording value to the class"""
	return recording_bindable.cls(cls)


def record_method(recording_resolver: Callable = recording_bindable.resolver('recording'),
                  execute_anyway: bool = False):
	"""
	Record a method into a recording, so that it can be executed at some other time
	:param recording_resolver: A resolver indicating where to record method, by default the instance's bound recording
	:param execute_anyway: Run function in addition to recording it, returning the function's return value
	:return: Decorated method
	"""

	def decorator(function):
		@wraps(function)
		def wrapper(*args, **kwargs):
			recording: Optional[Recording] = recording_resolver(*args, **kwargs)
			if recording is not None:
				# If recording was found, record function
				recording.append(function, args, kwargs)
				if not execute_anyway:
					# Check if function should be executed anyway
					return

			return function(*args, **kwargs)

		return wrapper

	return decorator
