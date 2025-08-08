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


def calculate_avg_abs_pearson_correlation_coefficient(S_1, S_2):
    # calculate the Pearson correlation coefficient between the sources in S_1 and S_2 for all permutations
    # return the highest correlation coefficient, which is for the best permutation
    '''

    Parameters
    ----------
    S_1 : np.ndarray
        sources of dimension N x T x K
    S_2 : np.ndarray
        sources of dimension N x T x K

    Returns
    -------
    avg_pearson : float
        average value of the absolute Pearson correlation coefficient
    '''

    N, T, K = S_1.shape

    if S_1.shape != S_2.shape:
        raise AssertionError("'S_1' and 'S_2' must have the same shape.")

    pearsoncoefficient = np.zeros((N, N))
    for col_idx in range(N):
        for m in range(N):
            # avg abs correlation coefficient for nth and mth source component in all K datasets
            pearsoncoefficient[col_idx, m] = 0
            for k in range(K):
                pearsoncoefficient[col_idx, m] += np.abs(pearsonr(S_1[col_idx, :, k], S_2[m, :, k]).statistic)
            pearsoncoefficient[col_idx, m] /= K

    # find the indices of the best permutation
    permutation = np.argmax(pearsoncoefficient, axis=1)

    # if column indices (elements in permutation) are not unique, find other match for duplicates
    if len(np.unique(permutation)) < N:

        # find duplicate column indices
        seen = set()
        dupes = set()
        for col in permutation:
            if col in seen:
                dupes.add(col)
            else:
                seen.add(col)

        # find corresponding row indices
        for col in dupes:
            row = np.where(permutation == col)[0]
            duplicate_pearson = pearsoncoefficient[row, col]
            # find row indices from largest to smallest Pearson correlation coefficient for that column
            sorted_row_idx = np.argsort(-duplicate_pearson)
            # row[sorted_row_idx[0]] has maximum pearson value
            # find new column idx for remaining row indices
            for row_idx in row[sorted_row_idx[1:]]:
                # find col indices from largest to smallest pearson correlation coefficients in that row
                sorted_col_idx = np.argsort(-pearsoncoefficient[row_idx, :])
                # update column idx (element in permutation) for that row
                for col_idx in range(1, len(sorted_col_idx)):
                    # check if second-largest element is not already a match for another row
                    if sorted_col_idx[col_idx] not in permutation:
                        # replace the element in permutation:
                        permutation[row_idx] = sorted_col_idx[col_idx]
                        break
                    else:
                        # wait for the next element
                        pass

    # avg correlation for this permutation
    avg_pearson = np.mean(pearsoncoefficient[np.arange(N), permutation])

    return avg_pearson
