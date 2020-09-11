
def verify_component_params(n, k=None, beta=None):
    _verify_n(n)
    if k:
        _verify_k(n,k)
    _verify_beta(beta)

def _verify_n(n):
    assert isinstance(n, int)
    assert n > 0
    assert n < 10**5, "The current implementation is constrained to small brains due to high memory usage"

def _verify_k(n,k):
    assert isinstance(k, int)
    assert k > 0
    assert k < n

def _verify_beta(beta):
    assert beta >= 0
    assert beta <= 1