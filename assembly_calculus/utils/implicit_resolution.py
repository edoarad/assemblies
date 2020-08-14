from functools import wraps
from inspect import Parameter, Signature
from typing import Tuple, Optional, Callable

from .argument_manipulation import signature


class ImplicitResolution:
    """
    Implicit argument resolution decorator.
    Allows arguments of functions to be resolved on run-time according to some (implicit) resolution function.
    """

    def __init__(self, **resolvers: Callable):
        """
        Creates an ImplicitResolution instance
        :param resolvers: A resolver is a function which is passed the (original) arguments and should resolve
        some additional (implicit) argument
        """
        self.resolvers = resolvers

    @staticmethod
    def wrap_function(function, **resolvers: Callable):
        """Wraps a function to comply with some (implicit) resolvers"""
        # Get the signature of the function we are wrapping
        sig: Signature = signature(function)
        effective_params: Tuple[Parameter, ...] = tuple(param for name, param in sig.parameters.items()
                                                        if name != 'self')
        # Get parameters which can be resolved
        resolvable_params: Tuple[Parameter, ...] = tuple(param for param in effective_params
                                                         if param.name in resolvers)
        resolvable_param_names: Tuple[str, ...] = tuple(param.name for param in resolvable_params)
        # Check for possibly resolvable parameters which are not keyword-only
        problem: Optional[Parameter] = next((param for param in resolvable_params
                                             if param.kind != Parameter.KEYWORD_ONLY), None)
        if problem:
            # Allow possibly resolvable parameters to be keyword-only, to avoid bugs and complications
            raise Exception(f"Cannot allow implicit resolution of parameter [{problem.name}] of [{function.__name__}]"
                            f", must be keyword-only")

        resolvers = {name: resolver for name, resolver in resolvers.items()
                     if name in resolvable_param_names}

        @wraps(function)
        def wrapper(*args, **kwargs):
            for name, resolver in resolvers.items():
                if name not in kwargs and (resolved := resolver(*args, **kwargs)) is not None:
                    kwargs[name] = resolved

            unresolved_params: Tuple[Parameter, ...] = tuple(param for param in resolvable_params
                                                             if param.default == Parameter.empty
                                                             and param.name not in kwargs)
            if len(unresolved_params) > 0:
                raise ValueError("Failed to implicitly resolve arguments [%s] of function %s"
                                 % (", ".join(param.name for param in unresolved_params), function.__name__))

            return function(*args, **kwargs)

        return wrapper

    def __call__(self, function):
        return self.wrap_function(function, **self.resolvers)

    def property(self, function):
        """A property with arguments that should be resolved implicitly"""
        return property(self(function))
