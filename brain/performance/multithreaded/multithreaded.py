import multiprocessing
import concurrent.futures

from typing import Callable, Any


def __identity__(x):
    return x


class Multithreaded:
    """
    This class is used as a wrapper result for the multithreaded decorator method.
    It takes a function to wrap, and the numbers of threads (defaults to the number of processor cores).
    Additional per-thread parameter pre-processing (via calling .params) can be added to an instance as well
    An on-finish hook can be added for post-processing of the thread data (via calling .after)

    For a usage example see the readme.
    """
    def __init__(self, func, threads=None):
        """
        :param func: The function to be wrapped.
        :param threads: Number of threads. None defaults to the number of processor cores.
        """
        self._function = func
        self._params: Callable[[int, ...], Any] = lambda _, *args, **kwargs: (args, kwargs)
        self._after = __identity__
        self.__name__ = func.__name__

        if hasattr(func, '__docs__'):
            self.__docs__ = func.__docs__
        if hasattr(func, '__signature__'):
            self.__signature__ = func.__signature__

        self._threads = threads or multiprocessing.cpu_count()
        self._executor = concurrent.futures.ThreadPoolExecutor(self._threads)

    def set_params(self, params):
        self._params = params

    def set_after(self, func):
        self._after = func

    def __call__(self, *args, **kwargs):
        futures = {}
        params = self._params(self._threads, *args, **kwargs)
        outs = [None] * self._threads
        for i, value in enumerate(params):
            t_args, t_kwargs = value

            def do_thread(m_i, m_args, m_kwargs):
                outs[m_i] = self._function(*m_args, **m_kwargs)

            futures[self._executor.submit(do_thread, i, t_args, t_kwargs)] = i
        concurrent.futures.wait(futures)
        return self._after(outs)

    def __get__(self, instance, owner):
        if instance:
            mt = Multithreaded(self._function.__get__(instance, owner), self._threads)
            mt.set_after(self._after.__get__(instance, owner))
            # TODO: document usage of __get__ in params
            # TODO 2: "__get__" to const
            # TODO NT: __get__ is a standard method name in python, this is like putting "__main__" in a const,
            # TODO NT: or explaining why you call __main__.py __main__.py
            if hasattr(self._params, '__get__'):
                mt.set_params(getattr(self._params, '__get__')(instance, owner))
            else:
                mt.set_params(self._params)

            return mt

    def __len__(self):
        return self._threads

    def __del__(self):
        self._executor.shutdown(False)


def multithreaded(func=None, *, threads=None):
    """
    :param func: The function to run in a multithreaded fashion
    :param threads: the number of threads to use, defaults to cpu

    Usage example over in the readme.

    """
    return Multithreaded(func, threads) if func else (lambda f: Multithreaded(f, threads))
