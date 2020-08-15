from functools import wraps
from types import MappingProxyType
from typing import Optional, Any, Tuple, Dict, Set
from inspect import Parameter

from .argument_manipulation import signature
from .implicit_resolution import ImplicitResolution


class Bindable:
    """
    Bindable decorator for classes
    This enables auto-filling parameters into the class functions
    For example, decorating a class Fooable with Bindable('bar'),
     which contains a function foo: (self, bar: int) -> 2 * bar, enables the following behaviour

    fooable = Fooable()
    fooable.bind(bar=5)
    fooable.foo() == 10
    fooable.foo(bar=10) == 20
    """

    def __init__(self, *params: str):
        """
        Creates a bindable decorator
        :param params: Parameters available for instance binding
        """
        self.params: Tuple[str, ...] = params

    @staticmethod
    def bound_value(param_name: str, *instances, graceful_none: bool = True, graceful_discrepancy: bool = False):
        """
        Attempts finding a common bound value in all instances of some parameter
        :param param_name: Parameter to attempt finding bound values for
        :param instances: Instances in which to search for bound value
        :param graceful_none: Return None if the parameter is unbound in all instances
        :param graceful_discrepancy: Return None if there are discrepancies in the bound value
        :return: A common bound value (or None)
        """
        def single_bound_value(name: str, instance) -> Tuple[bool, Optional[Any]]:
            bound_params: Dict[str, Any] = getattr(instance, 'bound_params', {})
            return name in bound_params, bound_params.get(name, None)

        _options: Tuple[Tuple[bool, Optional[Any]], ...] = tuple(single_bound_value(param_name, instance)
                                                                 for instance in instances)
        options: Set[Any] = set(implicit_value for found, implicit_value in _options if found)

        if len(options) == 1 and all(found for found, _ in _options):
            return options.pop()
        elif any(not found for found, _ in _options):
            if not graceful_none:
                raise ValueError("Some instances miss a bound value for %s" % param_name)
        elif len(options) > 1:
            if not graceful_discrepancy:
                raise ValueError("Multiple different bound values for %s in the different instances" % param_name)

        return None

    @staticmethod
    def resolver(argument: str):
        """Returns the resolver corresponding to some (bound) parameter"""
        def resolve(self, *args, **kwargs):
            return Bindable.bound_value(argument, self)

        return resolve

    @property
    def implicit_resolution(self):
        """The implicit resolution object corresponding to this bindable instance"""
        return ImplicitResolution(**{param: Bindable.resolver(param) for param in self.params})

    def cls(self, cls):
        """Wrap a class to support binding the instance parameters"""
        return Bindable.wrap_class(cls, self.params)

    def method(self, function):
        """Wrap a function to support fallback to bound instance values"""
        return self.implicit_resolution(function)

    def property(self, function):
        """Wrap a function to support parameters from bound instance values"""
        return property(self.method(function))

    @staticmethod
    def wrap_class(cls, params: Tuple[str, ...]):
        """
        Wraps a class to comply with the bindable decoration: wraps all methods and adds bind & unbind functionality
        :param cls: Class to decorate
        :param params: Parameters to allow binding for
        :return: Decorated class
        """
        # Update params to include previous bindings
        params: Tuple[str, ...] = tuple(set(getattr(cls, '_bindable_params', ()) + params))
        setattr(cls, '_bindable_params', params)

        original_init = getattr(cls, '__init__', lambda self, *args, **kwargs: None)

        @wraps(original_init)
        def new_init(self, *args, **kwargs):
            # Add _bound_params dictionary to instance
            original_init(self, *args, **kwargs)
            self._bound_params = {}

        setattr(cls, '__init__', new_init)

        # Many possible bound parameters
        def bind(self, **kwargs):
            """Binds (the) parameters {0} to the instance"""
            problem: str = next((kwarg for kwarg in kwargs if kwarg not in params), None)
            if problem:
                raise Exception(f"Cannot bind parameter [{problem}], was not declared bindable")

            for k, v in kwargs.items():
                self._bound_params[k] = v

        def bind_like(self, *others):
            """Binds this instance like another instance"""
            if len(others) == 0:
                self._bound_params = {}
                return

            self._bound_params = getattr(others[0], '_bound_params', {}).copy()
            for other in others[1:]:
                for key, value in getattr(other, 'bound_params', {}).items():
                    if key not in self._bound_params or self._bound_params.get(key) != value:
                        self._bound_params.pop(key)

        def unbind(self, *names: str):
            """Unbinds (the) parameters {0} from the instance, pass no names to unbind all"""
            problem: str = next((name for name in names if name not in params), None)
            if problem:
                raise Exception(f"Cannot unbind parameter [{problem}], was not declared bindable")

            if len(names) == 0:
                names = tuple(self._bound_params.keys())

            for name in names:
                if name in self._bound_params:
                    self._bound_params.pop(name)

        @property
        def bound_params(_self):
            return MappingProxyType(getattr(_self, '_bound_params', {}))

        bind.__doc__ = bind.__doc__.format(params)
        unbind.__doc__ = unbind.__doc__.format(params)

        # Update binding signature to match possible bound parameters
        bind.__signature__ = signature(bind).replace(parameters=
                                                     [Parameter(name='self', kind=Parameter.POSITIONAL_OR_KEYWORD)] +
                                                     [Parameter(name=name, kind=Parameter.KEYWORD_ONLY)
                                                      for name in params])

        # Add bind & unbind to the class
        setattr(cls, 'bind', bind)
        setattr(cls, 'unbind', unbind)
        setattr(cls, 'bind_like', bind_like)
        setattr(cls, 'bound_params', bound_params)

        return cls
