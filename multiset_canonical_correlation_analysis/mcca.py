import numpy as np
from scipy.linalg import eigh  # can also solve generalized EVD (compared to numpy.linalg.eigh)
from scipy.linalg import sqrtm, block_diag

from .helpers import vectorize_datasets, check_zero_mean, make_ccvs_unit_variance


def mcca(X, algorithm='genvar'):
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
        M, Epsilon = mcca_sumcorr_nielsen(X)
    elif algorithm == 'maxvar':
        M, Epsilon = mcca_maxvar_kettenring(X)
    elif algorithm == 'minvar':
        M, Epsilon = mcca_minvar_kettenring(X)
    elif algorithm == 'ssqcor':
        M, Epsilon = mcca_ssqcor_kettenring(X)
    elif algorithm == 'genvar':
        M, Epsilon = mcca_genvar_kettenring(X)
    else:
        raise AssertionError("'algorithm' must be 'sumcor', 'maxvar', 'minvar', 'ssqcor', or 'genvar'.")

    return M, Epsilon


def mcca_sumcorr_nielsen(X):
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


def mcca_maxvar_kettenring(X):
    """
    Implementation of mCCA-maxvar according to
    Kettenring, J. R. (1971). Canonical analysis of several sets of variables. Biometrika, 58(3), 433-451.

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
        # eigvals and eigvecs are sorted in descending order
        V[:, n] = eigvec[:, -1]

        # normalize each v_n^[k] -> w_n[k] has unit norm
        for k in range(K):
            v_n_k = V[k * N:(k + 1) * N, n]
            V_tilde[:, n, k] = v_n_k / np.linalg.norm(v_n_k)

    # now all b_n^[k] are calculated (Kettenring: j_b^(s), where k=j and n=s)

    # calculating demixing matrices to multiply with x^[k] instead of y^[k]
    M = np.zeros_like(V_tilde)
    for k in range(K):
        M[:, :, k] = np.linalg.inv(sqrtm(np.cov(X[:, :, k], ddof=0))) @ V_tilde[:, :, k]

    # calculate canonical variates (they already have unit variance)
    Epsilon = np.zeros_like(Y)
    for k in range(K):
        Epsilon[:, :, k] = M[:, :, k].T @ X[:, :, k]

    return M, Epsilon


def mcca_minvar_kettenring(X):
    """
    Implementation of mCCA-minvar according to
    Kettenring, J. R. (1971). Canonical analysis of several sets of variables. Biometrika, 58(3), 433-451.

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
        # eigvals and eigvecs are sorted in descending order. Eigenvector corresponding to last non-zero eigenvalue
        V[:, n] = eigvec[:, -(N - n) * K]

        # normalize each v_n^[k] -> w_n[k] has unit norm
        for k in range(K):
            v_n_k = V[k * N:(k + 1) * N, n]
            V_tilde[:, n, k] = v_n_k / np.linalg.norm(v_n_k)

    # now all b_n^[k] are calculated (Kettenring: j_b^(s), where k=j and n=s)

    # calculating demixing matrices to multiply with x^[k] instead of y^[k]
    M = np.zeros_like(V_tilde)
    for k in range(K):
        M[:, :, k] = np.linalg.inv(sqrtm(np.cov(X[:, :, k], ddof=0))) @ V_tilde[:, :, k]

    # calculate canonical variates (they already have unit variance)
    Epsilon = np.zeros_like(Y)
    for k in range(K):
        Epsilon[:, :, k] = M[:, :, k].T @ X[:, :, k]

    return M, Epsilon


def mcca_ssqcor_kettenring(X, max_iter=1000, eps=0.0001, verbose=False):
    """
    Implementation of mCCA-ssqcor according to
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
    V_tilde = 1 / np.sqrt(N) * np.ones((N, N, K))

    for n in range(N):
        theta_n = np.zeros((K, max_iter))
        for iter in range(max_iter):
            for k in range(K):
                if n == 0:
                    H_n_k = np.eye(N)  # A_j in eq. (12.5c) for the first CCV
                else:
                    V_n_1_k = V_tilde[:, 0:n, k]  # C_j on p.445
                    H_n_k = np.eye(N) - V_n_1_k @ np.linalg.inv(V_n_1_k.T @ V_n_1_k) @ V_n_1_k.T  # A_j in eq. (12.5c)

                # [ C_yy^[k,1] m_n^[1], ..., C_yy^[k,k-1] m_n^[k-1], C_yy^[k,k+1] m_n^[k+1], ..., C_yy^[k,K] m_n^[K] ]
                N_n_k = []
                for l in range(K):
                    if l != k:
                        N_n_k.append(C_yy[k * N: (k + 1) * N, l * N: (l + 1) * N] @ V_tilde[:, n, l])
                N_n_k = np.array(N_n_k).T  # called N_j in Kettenring's paper p.445, his j is our k
                P_n_k = N_n_k @ N_n_k.T  # P_j on p.445

                # we can perform EVD of H_n F_n H_n, as we are just interested in the leading eigenvector
                lambda_n_k, V_n_k = np.linalg.eigh(H_n_k @ P_n_k @ H_n_k)
                V_tilde[:, n, k] = V_n_k[:, -1]
                theta_n[k, iter] = 1 + lambda_n_k[-1]

            if np.sum(np.abs(theta_n[:, iter] - theta_n[:, iter - 1])) < eps or iter == max_iter:  # eq. (12.9)
                if verbose:
                    print(f'Stopping for the {n}th CCV after {iter} iterations')
                break

    # find transformation matrices for X^[k] instead of Y^[k]
    M = np.zeros((N, N, K))
    for k in range(K):
        M[:, :, k] = np.linalg.inv(sqrtm(np.cov(X[:, :, k], ddof=0))) @ V_tilde[:, :, k]

    # calculate canonical variates (they already have unit variance)
    Epsilon = np.zeros_like(X)
    for k in range(K):
        Epsilon[:, :, k] = M[:, :, k].T @ X[:, :, k]

    return M, Epsilon


def mcca_genvar_kettenring(X, max_iter=1000, eps=0.0001, verbose=False):
    """
    Implementation of mCCA-genvar according to
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
    V_tilde = 1 / np.sqrt(N) * np.ones((N, N, K))

    for n in range(N):
        theta_n = np.zeros((K, max_iter))
        for iter in range(1, max_iter):
            for k in range(K):
                if n == 0:
                    H_n_k = np.eye(N)  # A_j in eq. (12.5c) for the first CCV
                    # init R_n
                    B_n = block_diag(*V_tilde[:, n, :].T).T
                    R_n = B_n.T @ C_yy @ B_n  # K x K covariance matrix of nth CCV
                else:
                    V_n_1_k = V_tilde[:, 0:n, k]  # C_j on p.445
                    H_n_k = np.eye(N) - V_n_1_k @ np.linalg.inv(V_n_1_k.T @ V_n_1_k) @ V_n_1_k.T  # A_j in eq. (12.5c)

                # [ C_yy^[k,1] m_n^[1], ..., C_yy^[k,k-1] m_n^[k-1], C_yy^[k,k+1] m_n^[k+1], ..., C_yy^[k,K] m_n^[K] ]
                N_n_k = []
                for l in range(K):
                    if l != k:
                        N_n_k.append(C_yy[k * N: (k + 1) * N, l * N: (l + 1) * N] @ V_tilde[:, n, l])
                N_n_k = np.array(N_n_k).T  # called N_j in Kettenring's paper p.445, his j is our k

                # delete kth row and kth column of nth CCV covariance matrix R_n
                R_n_minus_k = np.delete(np.delete(R_n, k, 0), k, 1)  # K-1 x K-1 matrix Phi_j on p.445, his j = our k
                F_n_k = N_n_k @ np.linalg.inv(R_n_minus_k) @ N_n_k.T  # Q_j on p.445

                # we can perform EVD of H_n F_n H_n, as we are just interested in the leading eigenvector
                lambda_n_k, V_n_k = np.linalg.eigh(H_n_k @ F_n_k @ H_n_k)
                V_tilde[:, n, k] = V_n_k[:, -1]
                theta_n[k, iter] = 1 + lambda_n_k[-1]

                # update R_n
                B_n = block_diag(*V_tilde[:, n, :].T).T
                R_n = B_n.T @ C_yy @ B_n  # K x K covariance matrix of nth CCV

            if np.sum(np.abs(theta_n[:, iter] - theta_n[:, iter - 1])) < eps or iter == max_iter:  # eq. (12.9)
                if verbose:
                    print(f'Stopping for the {n}th CCV after {iter} iterations')
                break

    # find transformation matrices for X^[k] instead of Y^[k]
    M = np.zeros((N, N, K))
    for k in range(K):
        M[:, :, k] = np.linalg.inv(sqrtm(np.cov(X[:, :, k], ddof=0))) @ V_tilde[:, :, k]

    # calculate canonical variates (they already have unit variance)
    Epsilon = np.zeros_like(X)
    for k in range(K):
        Epsilon[:, :, k] = M[:, :, k].T @ X[:, :, k]

    return M, Epsilon
