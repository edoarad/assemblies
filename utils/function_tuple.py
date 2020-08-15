class FunctionTuple:

    def __init__(self, *functions):
        self.functions = list(functions)

    def __call__(self, *args, **kwargs):
        return [function(*args, **kwargs) for function in self.functions]

    def append(self, functions):
        self.functions += functions

    def register_function(self, function):
        self.functions.append(function)
        return function

    def index(self, function):
        return self.functions.index(function)
