import numpy as np


def vectorize_datasets(X):
    """
    Concatenate datasets vertically.

    Parameters
    ----------
    X : np.ndarray
        data of dimensions N x T x K

    Returns
    -------
    X_concat : np.ndarray
        stacked datasets of dimensions NK x T

    """

    N, T, K = X.shape
    # stack datasets -> [x^[1]^T, ..., x^[K]^T]^T
    X_concat = np.reshape(np.moveaxis(X, [0, 1, 2], [0, 2, 1]), (N * K, T), 'F')

    return X_concat


def check_zero_mean(X):
    N, T, K = X.shape
    # make sure data is zero-mean
    for k in range(K):
        np.testing.assert_almost_equal(np.mean(X[:, :, k], axis=1), 0)


def make_ccvs_unit_variance(M, Epsilon):
    # make Epsilon unit-variance and write std in W
    M_unit_var = np.zeros_like(M)
    Epsilon_unit_var = np.zeros_like(Epsilon)
    for k in range(Epsilon.shape[2]):
        std = np.std(Epsilon[:, :, k], axis=1, keepdims=True)
        Epsilon_unit_var[:, :, k] = Epsilon[:, :, k] / std
        M_unit_var[:, :, k] = M[:, :, k] / std.T
    return M_unit_var, Epsilon_unit_var


def calculate_ccv_covariance_matrices(Epsilon):
    N, _, K = Epsilon.shape
    ccv_cov = np.zeros((K, K, N))
    for n in range(N):
        ccv_cov[:, :, n] = np.cov(Epsilon[n, :, :].T, ddof=0)  # ddof=0 means dividing by N
    return ccv_cov


def calculate_eigenvalues_from_ccv_covariance_matrices(ccv_cov):
    K, _, N = ccv_cov.shape
    Lambda = np.zeros((N, K))
    for n in range(N):
        Lambda[n, :] = np.linalg.eigh(ccv_cov[:, :, n])[0]
    return Lambda
