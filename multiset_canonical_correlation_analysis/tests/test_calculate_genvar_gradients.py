import numpy as np
from scipy.linalg import eigh  # can also solve generalized EVD (compared to numpy.linalg.eigh)
from scipy.linalg import sqrtm, block_diag

from ..helpers import vectorize_datasets, check_zero_mean, make_ccvs_unit_variance

from independent_vector_analysis.initializations import _jbss_sos, _cca


# compare Kettenring's gradient to mine for symmetric and unsymmetric matrix Sigma


def calculate_Cyy(X):
    N, T, K = X.shape
    # stack datasets -> [x^[1]^T, ..., x^[K]^T]^T
    X_concat = vectorize_datasets(X)
    C_xx = np.cov(X_concat, ddof=0)

    # whiten x^[k] -> y^[k] (using Mahalanobis whitening)
    Y = np.zeros_like(X)
    for k in range(K):
        # ddof=0 means dividing by T, not T-1
        Y[:, :, k] = np.linalg.inv(sqrtm(C_xx[k * N:(k + 1) * N, k * N:(k + 1) * N])) @ X[:, :, k]

    # concatenate whitened datasets vertically
    Y_concat = vectorize_datasets(Y)

    # calculate C_yy = E[y y^T]
    C_yy = np.cov(Y_concat, ddof=0)

    return C_yy


def compute_R_n(X, V, n):
    """
    Compute the matrix R_n where R_n^[k, l] = v_n^[k]^T C_yy v_n^[l].

    Parameters:
    - C: NK x NK symmetric covariance matrix of kth whitened dataset
    - V: N x N x K matrix, where V[:,n,k] = v_n^[k]^T is the nth column of V^[k]

    Returns:
    - X: n x n matrix
    """

    N, T, K = X.shape
    C_yy = calculate_Cyy(X)

    V_n = block_diag(*V[:, n, :].T).T
    R_n = V_n.T @ C_yy @ V_n  # K x K covariance matrix of nth CCV

    return R_n

def compute_det_R_n_like_Kettenring(X,V,n,k):
    N,T, K = X.shape
    R_n = compute_R_n(X, V, n)
    # delete kth row and kth column of nth CCV covariance matrix R_n
    R_n_minus_k = np.delete(np.delete(R_n, k, 0), k, 1)  # K-1 x K-1

    C_yy = calculate_Cyy(X)

    # N_n^[k] = [ C_yy^[k,1] m_n^[1], ..., C_yy^[k,k-1] m_n^[k-1], C_yy^[k,k+1] m_n^[k+1], ..., C_yy^[k,K] m_n^[K] ]
    N_n_k = []
    for l in range(K):
        if l != k:
            N_n_k.append(C_yy[k * N: (k + 1) * N, l * N: (l + 1) * N] @ V[:, n, l])
    N_n_k = np.array(N_n_k).T

    R_tilde_n_k = N_n_k @ np.linalg.inv(R_n_minus_k) @ N_n_k.T
    det = np.linalg.det(R_n_minus_k) * (1 - V[:, n, k].T @ R_tilde_n_k @ V[:, n, k])
    return det

def numerical_gradient_by_wnk(X, V, n, k, epsilon=1e-6):
    """
    Numerically compute the gradient of det(X) with respect to w_n^[k].

    Parameters:
    - C: d x d matrix
    - W: n x d matrix (each row is w_k)
    - k: index of the row in W to differentiate with respect to
    - epsilon: small value for finite difference

    Returns:
    - grad: gradient vector of size N (∂det(X)/∂w_k)
    """

    N, T, K = X.shape

    grad = np.zeros(N)
    # determinant for nth SCV
    original_det = np.linalg.det(compute_R_n(X, V, n))

    # use Kettenring's formula to calculate determinant and then check gradient
    #####
    # original_det = compute_det_R_n_like_Kettenring(X,V,n,k)
    ####

    for i in range(N):
        V_perturbed = V.copy()
        V_perturbed[i, n, k] += epsilon  # change ith element of v_n^[k] and compare this det to the previous one

        # the following makes V and V_perturbed super close to each other (7e-15)
        # and the gradient super small, around 3e-09, therefore should not be done
        # make V_perturbed[:,n,k] unit-norm
        # V_perturbed[:, n, k] /= np.linalg.norm(V_perturbed[:,n,k])
        # and orthogonal to other vectors in V
        # Vnk = np.delete(V_perturbed[:,:,k], n, 1)  # N x (n-1) matrix containing all but the nth transformation vector
        # Pnk = np.eye(N) - Vnk @ np.linalg.inv(Vnk.T @ Vnk) @ Vnk.T
        # V_perturbed[:, n, k] = Pnk @ V_perturbed[:,n, k]  # update v_n^[k]
        # V_perturbed[:, n, k] /= np.linalg.norm(V_perturbed[:,n, k])  # make vectors unit-norm

        det_perturbed = np.linalg.det(compute_R_n(X, V_perturbed, n))
        # det_perturbed = compute_det_R_n_like_Kettenring(X,V_perturbed,n,k)
        grad[i] = (det_perturbed - original_det) / epsilon

    return grad


def Kettenring_gradient_by_wnk(X, V, n, k):
    N, T, K = X.shape

    R_n = compute_R_n(X, V, n)
    # delete kth row and kth column of nth CCV covariance matrix R_n
    R_n_minus_k = np.delete(np.delete(R_n, k, 0), k, 1)  # K-1 x K-1

    C_yy = calculate_Cyy(X)

    # N_n^[k] = [ C_yy^[k,1] m_n^[1], ..., C_yy^[k,k-1] m_n^[k-1], C_yy^[k,k+1] m_n^[k+1], ..., C_yy^[k,K] m_n^[K] ]
    N_n_k = []
    for l in range(K):
        if l != k:
            N_n_k.append(C_yy[k * N: (k + 1) * N, l * N: (l + 1) * N] @ V[:, n, l])
    N_n_k = np.array(N_n_k).T

    R_tilde_n_k = N_n_k @ np.linalg.inv(R_n_minus_k) @ N_n_k.T
    grad_nk = -2 * np.linalg.det(R_n_minus_k) * R_tilde_n_k @ V[:, n, k]
    return grad_nk


def symmetric_matrix_gradient_by_wnk(X, V, n, k):
    N, T, K = X.shape

    R_n = compute_R_n(X, V, n)
    C_yy = calculate_Cyy(X)

    # N_n^[k] = [ C_yy^[k,1] m_n^[1], ..., C_yy^[k,k-1] m_n^[k-1], C_yy^[k,k] m_n^[k],
    #                                                       C_yy^[k,k+1] m_n^[k+1], ..., C_yy^[k,K] m_n^[K] ]
    N_n_k = []
    for l in range(K):
        N_n_k.append(C_yy[k * N: (k + 1) * N, l * N: (l + 1) * N] @ V[:, n, l])
    N_n_k = np.array(N_n_k).T

    grad_nk = 2 * np.linalg.det(R_n) * N_n_k @ np.linalg.inv(R_n)[:, k]
    return grad_nk


def unsymmetric_matrix_gradient_by_wnk(X, V, n, k, epsilon=1e-10):
    N, T, K = X.shape

    R_n = compute_R_n(X, V, n)
    C_yy = calculate_Cyy(X)

    # N_n^[k] = [ C_yy^[k,1] m_n^[1], ..., C_yy^[k,k-1] m_n^[k-1], C_yy^[k,k] m_n^[k],
    #                                                       C_yy^[k,k+1] m_n^[k+1], ..., C_yy^[k,K] m_n^[K] ]
    N_n_k = []
    for l in range(K):
        N_n_k.append(C_yy[k * N: (k + 1) * N, l * N: (l + 1) * N] @ V[:, n, l])
    N_n_k = np.array(N_n_k).T

    grad_nk = 4 * np.linalg.det(R_n) * N_n_k @ np.linalg.inv(R_n)[:, k] - 2 * np.linalg.det(R_n) * np.linalg.inv(R_n)[
        k, k] * C_yy[k * N: (k + 1) * N, k * N: (k + 1) * N] @ V[:, n, k]
    return grad_nk


# Example usage:
def test_compare_gradients():
    np.random.seed(42)
    N = 10
    K = 20
    T = 10000

    X = np.random.randn(N, T, K)
    X -= np.mean(X, axis=1, keepdims=True)

    # initialize orthogonal transformation matrices (that would transform Y)
    V = np.random.randn(N, N, K)
    for k in range(K):
        V[:, :, k] = np.linalg.solve(sqrtm(V[:, :, k] @ V[:, :, k].T), V[:, :, k])

    n = 0
    k = 5

    # numerical gradient
    numerical_grad_nk = numerical_gradient_by_wnk(X, V, n, k)
    print(f"Numerical gradient of det(X) with respect to w_{n}^[{k}]:\n{numerical_grad_nk}")

    Kettenring_grad_nk = Kettenring_gradient_by_wnk(X, V, n, k)
    print(f"Kettenring gradient of det(X) with respect to w_{n}^[{k}]:\n{Kettenring_grad_nk}")
    print(f"Difference to numerical: {np.linalg.norm(numerical_grad_nk-Kettenring_grad_nk)}")

    symmetric_grad_nk = symmetric_matrix_gradient_by_wnk(X, V, n, k)
    print(f"Symmetric gradient of det(X) with respect to w_{n}^[{k}]:\n{symmetric_grad_nk}")
    print(f"Difference to numerical: {np.linalg.norm(numerical_grad_nk-symmetric_grad_nk)}")

    unsymmetric_grad_nk = unsymmetric_matrix_gradient_by_wnk(X, V, n, k)
    print(f"Unsymmetric gradient of det(X) with respect to w_{n}^[{k}]:\n{unsymmetric_grad_nk}")
    print(f"Difference to numerical: {np.linalg.norm(numerical_grad_nk-unsymmetric_grad_nk)}")