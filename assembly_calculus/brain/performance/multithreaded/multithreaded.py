import multiprocessing
import concurrent.futures

from typing import Callable, Any, Iterable, Tuple, Dict, Collection, List


def __identity__(x):
    return x


ThreadNum = int


class _Multithreaded:
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
        # _function is the function each thread will execute.
        self._function = func

        # _params is a function which receives the thread number and parameters to the wrapped function.
        # It should returns an iterable [(args_i, kwargs_i) for i in thread_num] where args_i, kwargs_i are
        # the parameters given to the i'th thread.
        self._params: Callable[[ThreadNum, ...], Iterable[Tuple[Collection, Dict]]] = \
            lambda n, *args, **kwargs: [(args, kwargs) for _ in range(n)]

        # _after is a function which receives the list of outputs of each thread and combines them to get the final
        # return value.
        self._after: Callable[[List], Any] = __identity__

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
        """
        Calling the wrapped function.
        """
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
    return _Multithreaded(func, threads) if func else (lambda f: _Multithreaded(f, threads))
