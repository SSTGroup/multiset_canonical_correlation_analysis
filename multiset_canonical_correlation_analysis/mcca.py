import numpy as np
from scipy.linalg import eigh  # can also solve generalized EVD (compared to numpy.linalg.eigh)
from scipy.linalg import sqrtm, block_diag

from .helpers import vectorize_datasets, check_zero_mean, make_ccvs_unit_variance


def mcca(X, algorithm='genvar', max_iter=1000, eps=0.0001, verbose=False):
    """
    Perform mCCA on datasets X^[k].

    Info
    ----
    N: number of components, T: number of samples, K: number of datasets

    Parameters
    ----------
    X : np.ndarray
        Datasets of dimensions N x T x K, where X[:,:,k] corresponds to X^[k]

    algorithm : str, optional
        mCCA algorithm: possible options are: 'sumcor', 'maxvar', 'mivar', 'ssqcor', 'genvar'

    Returns
    -------
    M : np.ndarray
        Transformation matrices of dimension N x N x K, such that E^[k] = (M^[k])^T X^[k]

    Epsilon : np.ndarray
        Canonical variables of dimension N x T x K

    """

    if algorithm == 'sumcor':
        M, Epsilon = mcca_sumcor_nielsen(X)
    elif algorithm == 'maxvar':
        M, Epsilon = mcca_maxvar_minvar_kettenring(X, 'maxvar')
    elif algorithm == 'minvar':
        M, Epsilon = mcca_maxvar_minvar_kettenring(X, 'minvar')
    elif algorithm == 'ssqcor':
        M, Epsilon = mcca_ssqcor_genvar_kettenring(X, 'ssqcor', max_iter, eps, verbose)
    elif algorithm == 'genvar':
        M, Epsilon = mcca_ssqcor_genvar_kettenring(X, 'genvar', max_iter, eps, verbose)
    else:
        raise AssertionError("'algorithm' must be 'sumcor', 'maxvar', 'minvar', 'ssqcor', or 'genvar'.")

    return M, Epsilon


def mcca_sumcor_nielsen(X):
    """
    Implementation of mCCA-sumcor according to
    Nielsen, Allan Aasbjerg. "Multiset canonical correlations analysis and multispectral, truly multitemporal remote
    sensing data." IEEE transactions on image processing 11.3 (2002): 293-305.


    Info
    ----
    N: number of components, T: number of samples, K: number of datasets

    Parameters
    ----------
    X : np.ndarray
        Datasets of dimensions N x T x K, where X[:,:,k] corresponds to X^[k]

    Returns
    -------
    M : np.ndarray
        Transformation matrices of dimension N x N x K, such that E^[k] = (M^[k])^T X^[k]

    Epsilon : np.ndarray
        Canonical variables of dimension N x T x K

    """

    N, T, K = X.shape

    # make sure data is zero-mean
    check_zero_mean(X)

    # stack datasets -> [x^[1]^T, ..., x^[K]^T]^T
    X_concat = vectorize_datasets(X)
    C = np.cov(X_concat, ddof=0)

    # cut diagonal blocks of C and store them in D
    D = np.zeros_like(C)
    for k in range(K):
        D[k * N:(k + 1) * N, k * N:(k + 1) * N] = C[k * N:(k + 1) * N, k * N:(k + 1) * N]

    # solve GEVD C w_n = lambda_n D w_n
    eigvals, eigvecs = eigh(C, D)

    # take only N largest eigenvalues and corresponding eigenvectors
    eigvecs = eigvecs[:, ::-1]  # sort ascending
    M_tilde = eigvecs[:, 0:N]  # eigenvectors corresponding to N largest EVs

    # match elements of M to M^[k]
    M = np.zeros((N, N, K))
    for k in range(K):
        M[:, :, k] = M_tilde[k * N:(k + 1) * N, :]

    # calculate canonical variates
    Epsilon = np.zeros_like(X)
    for k in range(K):
        Epsilon[:, :, k] = M[:, :, k].T @ X[:, :, k]

    # normalize canonical variables to unit variance (and save scalings in transformation matrices)
    M, Epsilon = make_ccvs_unit_variance(M, Epsilon)

    return M, Epsilon


def mcca_maxvar_minvar_kettenring(X, algorithm):
    """
    Implementation of mCCA-maxvar and mCCA-minvar according to
    Kettenring, J. R. (1971). Canonical analysis of several sets of variables. Biometrika, 58(3), 433-451.

    Info
    ----
    N: number of components, T: number of samples, K: number of datasets

    Parameters
    ----------
    X : np.ndarray
        Datasets of dimensions N x T x K, where X[:,:,k] corresponds to X^[k]

    algorithm: str
        'maxvar' or 'minvar'

    Returns
    -------
    M : np.ndarray
        Transformation matrices of dimension N x N x K, such that E^[k] = (M^[k])^T X^[k]

    Epsilon : np.ndarray
        Canonical variables of dimension N x T x K

    """

    N, T, K = X.shape

    # make sure data is zero-mean
    check_zero_mean(X)

    # whiten x^[k] -> y^[k] (using Mahalanobis whitening)
    Y = np.zeros_like(X)
    for k in range(K):
        # ddof=0 means dividing by T, not T-1
        Y[:, :, k] = np.linalg.inv(sqrtm(np.cov(X[:, :, k], ddof=0))) @ X[:, :, k]

    # concatenate whitened datasets vertically
    Y_concat = vectorize_datasets(Y)

    # calculate C_yy = E[y y^T]
    C_yy = np.cov(Y_concat, ddof=0)

    # calculate the canonical variates in a deflationary way for each stage n = 1...N
    V = np.zeros((N * K, N))
    V_tilde = np.zeros((N, N, K))
    for n in range(N):

        if n == 0:
            # H_n for the first stage is the identity matrix
            H_n = np.eye(N * K)
        else:
            # update H_n matrix

            # stack K partitions of V_tilde as diagonal blocks of V_tilde_n1 (of dimensions NK x (n-1)K )
            # Kettenring eq. 9.9: D_c = diag{1_C^(s), ..., m_C^{(s)}, j_C^(s) = {j_b^(1), ..., j_b^(s-1)}
            V_list = [V_tilde[:, 0:n, k] for k in range(K)]  # V_tilde is defined later
            V_tilde_n1 = block_diag(*V_list)

            # update H_n
            H_n = np.eye(N * K) - V_tilde_n1 @ np.linalg.inv(V_tilde_n1.T @ V_tilde_n1) @ V_tilde_n1.T

        # calc v_n as first eigvec of EVD(H_n R H_n), where H_1 = I
        eigval, eigvec = eigh(H_n @ C_yy @ H_n)
        # eigvals and eigvecs should be sorted in descending order
        eigvec = eigvec[:, ::-1]

        if algorithm == 'maxvar':
            # eigenvector corresponding to largest eigenvalue
            V[:, n] = eigvec[:, 0]
        elif algorithm == 'minvar':
            # eigenvector corresponding to smallest non-zero eigenvalue
            V[:, n] = eigvec[:, N * K - n * K - 1]
        else:
            raise ValueError("'algorithm' must be 'maxvar' or 'minvar'")

        # normalize each v_n^[k] -> w_n[k] has unit norm
        for k in range(K):
            v_n_k = V[k * N:(k + 1) * N, n]
            V_tilde[:, n, k] = v_n_k / np.linalg.norm(v_n_k)

    # now all m_n^[k] are calculated (Kettenring: j_b^(s), where k=j and n=s)

    # calculating demixing matrices to multiply with x^[k] instead of y^[k]
    M = np.zeros_like(V_tilde)
    for k in range(K):
        M[:, :, k] = np.linalg.inv(sqrtm(np.cov(X[:, :, k], ddof=0))) @ V_tilde[:, :, k]

    # calculate canonical variates (they already have unit variance)
    Epsilon = np.zeros_like(Y)
    for k in range(K):
        Epsilon[:, :, k] = M[:, :, k].T @ X[:, :, k]

    return M, Epsilon


def mcca_ssqcor_genvar_kettenring(X, algorithm, max_iter=1000, eps=0.0001, verbose=False):
    """
    Implementation of mCCA-ssqcor and mCCA-genvar according to
    Kettenring, J. R. (1971). Canonical analysis of several sets of variables. Biometrika, 58(3), 433-451.

    Info
    ----
    N: number of components, T: number of samples, K: number of datasets

    Parameters
    ----------
    X : np.ndarray
        Datasets of dimensions N x T x K, where X[:,:,k] corresponds to X^[k]

    max_iter : int, optional
        Maximum number of iterations before stopping the optimization

    eps : float, optional
        Threshold value for converge. If change of theta parameter is smaller than eps, optimization will stop.

    verbose : bool, optional
        If True, print after how many iterations the algorithm stopped for each SCV

    Returns
    -------
    M : np.ndarray
        Transformation matrices of dimension N x N x K, such that E^[k] = (M^[k])^T X^[k]

    Epsilon : np.ndarray
        Canonical variables of dimension N x T x K

    """

    N, T, K = X.shape

    # make sure data is zero-mean
    check_zero_mean(X)

    # whiten x^[k] -> y^[k] (using Mahalanobis whitening)
    Y = np.zeros_like(X)
    for k in range(K):
        # ddof=0 means dividing by T, not T-1
        Y[:, :, k] = np.linalg.inv(sqrtm(np.cov(X[:, :, k], ddof=0))) @ X[:, :, k]

    # concatenate whitened datasets vertically
    Y_concat = vectorize_datasets(Y)

    # calculate C_yy = E[y y^T]
    C_yy = np.cov(Y_concat, ddof=0)

    # initialize transformation matrices (that would transform Y)
    V = 1 / np.sqrt(N) * np.ones((N, N, K))

    for n in range(N):
        theta_n = np.zeros((K, max_iter))
        for iter in range(1, max_iter):
            for k in range(K):
                if n == 0:
                    H_n_k = np.eye(N)  # A_j in eq. (12.5c) for the first CCV
                    if algorithm == 'genvar':
                        # init R_n
                        V_n = block_diag(*V[:, n, :].T).T
                        R_n = V_n.T @ C_yy @ V_n  # K x K covariance matrix of nth CCV
                else:
                    V_n_1_k = V[:, 0:n, k]  # C_j on p.445
                    H_n_k = np.eye(N) - V_n_1_k @ np.linalg.inv(V_n_1_k.T @ V_n_1_k) @ V_n_1_k.T  # A_j in eq. (12.5c)

                # [ C_yy^[k,1] m_n^[1], ..., C_yy^[k,k-1] m_n^[k-1], C_yy^[k,k+1] m_n^[k+1], ..., C_yy^[k,K] m_n^[K] ]
                N_n_k = []
                for l in range(K):
                    if l != k:
                        N_n_k.append(C_yy[k * N: (k + 1) * N, l * N: (l + 1) * N] @ V[:, n, l])
                N_n_k = np.array(N_n_k).T  # called N_j in Kettenring's paper p.445, his j is our k

                if algorithm == 'ssqcor':
                    F_n_k = N_n_k @ N_n_k.T  # P_j on p.445
                elif algorithm == 'genvar':
                    # delete kth row and kth column of nth CCV covariance matrix R_n
                    R_n_minus_k = np.delete(np.delete(R_n, k, 0), k, 1)  # K-1 x K-1 matrix Phi_j on p.445, his j=our k
                    F_n_k = N_n_k @ np.linalg.inv(R_n_minus_k) @ N_n_k.T  # Q_j on p.445
                else:
                    raise ValueError("'algorithm' must be 'ssqcor' or 'genvar'")

                # we can perform EVD of H_n F_n H_n, as we are just interested in the leading eigenvector
                eigval, eigvec = np.linalg.eigh(H_n_k @ F_n_k @ H_n_k)
                # eigvals and eigvecs should be sorted in descending order
                eigval = eigval[::-1]
                eigvec = eigvec[:, ::-1]
                V[:, n, k] = eigvec[:, 0]
                theta_n[k, iter] = 1 + eigval[0]

                if algorithm == 'genvar':
                    # update R_n
                    V_n = block_diag(*V[:, n, :].T).T
                    R_n = V_n.T @ C_yy @ V_n  # K x K covariance matrix of nth CCV

            if np.sum(np.abs(theta_n[:, iter] - theta_n[:, iter - 1])) < eps:  # eq. (12.9)
                if verbose:
                    print(f'Stopping for the {n}th CCV after {iter} iterations')
                break

    # find transformation matrices for X^[k] instead of Y^[k]
    M = np.zeros((N, N, K))
    for k in range(K):
        M[:, :, k] = np.linalg.inv(sqrtm(np.cov(X[:, :, k], ddof=0))) @ V[:, :, k]

    # calculate canonical variates (they already have unit variance)
    Epsilon = np.zeros_like(X)
    for k in range(K):
        Epsilon[:, :, k] = M[:, :, k].T @ X[:, :, k]

    return M, Epsilon
