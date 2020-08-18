from functools import wraps


class FunctionTuple:
    """
    This class allows to compute multiple functions on the same input
    and gather these functions using a wrapper
    """
    def __init__(self, *functions):
        """
        Initial function set
        :param functions: a bunch of functions that expect the same inputs
        """
        self.functions = list(functions)

    def __call__(self, *args, **kwargs):
        """

        :param args:
        :param kwargs:
        :return: return a dictionary of function: output
        """
        return {function: function(*args, **kwargs) for function in self.functions}

    def append(self, functions):
        """
        Add a function to the the function collection
        :param functions: a list of functions
        """
        self.functions += functions

    @wraps
    def register_function(self, function):
        """
        A wrapper that adds the wrapped function to the function collection
        :param function: a function to add to the function collection
        :return: the original function
        """
        self.append([function])
        return function
