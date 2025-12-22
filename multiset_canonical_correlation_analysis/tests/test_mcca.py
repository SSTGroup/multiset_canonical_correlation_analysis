import numpy as np

from ..mcca import mcca


def test_mcca():
    N = 8
    K = 9
    V = 1000
    X = np.random.randn(N, V, K)
    X -= np.mean(X, axis=1, keepdims=True)
    mcca(X, 'genvar')