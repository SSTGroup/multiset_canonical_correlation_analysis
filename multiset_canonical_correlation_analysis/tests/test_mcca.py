import numpy as np

from ..mcca import mcca


def test_mcca():
    X = np.random.randn(8, 10000, 9)
    X -= np.mean(X, axis=1, keepdims=True)
    mcca(X, 'genvar')
