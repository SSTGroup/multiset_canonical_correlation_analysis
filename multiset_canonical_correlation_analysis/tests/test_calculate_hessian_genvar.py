import numpy as np
from scipy.linalg import sqrtm, block_diag

from .test_calculate_genvar_gradients import compute_R_n, calculate_Cyy


def compute_Nnk(C_yy, V_n, l):
    N, K = V_n.shape
    for k in range(K):
        # [ C_yy^[k,1] m_n^[1], ..., C_yy^[k,k-1] m_n^[k-1], C_yy^[k,k+1] m_n^[k+1], ..., C_yy^[k,K] m_n^[K] ]
        N_n_l = []
        for k in range(K):
            if k != l:
                N_n_l.append(C_yy[l * N: (l + 1) * N, k * N: (k + 1) * N] @ V_n[:, k])
        N_n_l = np.array(N_n_l).T  # called N_j in Kettenring's paper p.445, his j is our l

    return N_n_l


def numerical_gradient_Nnl_by_vnki(X, V_n, k, l, epsilon=1e-6):
    """
    Numerically compute the gradient of N_n^[l] w.r.t v_n^[k](i), for l != k.

    Parameters:
        V_n: N x K matrix. V_n[:,k] is v_n^[k], i.e., the nth column of V^[k]
    - epsilon: small value for finite difference

    Returns:
    - grad: gradient vector of size N (∂det(X)/∂w_k)
    """

    N, T, K = X.shape
    C_yy = calculate_Cyy(X)

    original_Nnk = compute_Nnk(C_yy, V_n, l)

    numerical_grad = np.zeros((N, N, K - 1))
    for i in range(N):
        V_n_perturbed = V_n.copy()
        V_n_perturbed[i, k] += epsilon  # change ith element of v_n^[k] and compare this Nnk to the previous one
        Nnk_perturbed = compute_Nnk(C_yy, V_n_perturbed, l)

        numerical_grad[i, :, :] = (Nnk_perturbed - original_Nnk) / epsilon

    # compare with analytical formulation
    analytical_grad = np.zeros((N, N, K - 1))
    for i in range(N):
        if l < k:
            analytical_grad[i, :, k - 1] = C_yy[k * N: (k + 1) * N, l * N: (l + 1) * N][i, :]
        elif l > k:
            analytical_grad[i, :, k] = C_yy[k * N: (k + 1) * N, l * N: (l + 1) * N][i, :]

    return numerical_grad, analytical_grad


def numerical_gradient_Rnminusl_by_vnki(X, V_n, k, l, epsilon=1e-6):
    """
    Numerically compute the gradient of R_n^[-l] w.r.t v_n^[k](i), for l != k.

    Parameters:
        V_n: N x K matrix. V_n[:,k] is v_n^[k], i.e., the nth column of V^[k]
    - epsilon: small value for finite difference

    Returns:
    - grad: gradient vector of size N (∂det(X)/∂w_k)
    """

    N, T, K = X.shape
    C_yy = calculate_Cyy(X)

    original_Rn = compute_R_n(X, V_n)
    original_R_n_minus_l = np.delete(np.delete(original_Rn, l, 0), l, 1)  # K-1 x K-1 matrix Phi_j on p.445, his j=our l

    numerical_grad = np.zeros((N, K - 1, K - 1))
    for i in range(N):
        V_n_perturbed = V_n.copy()
        V_n_perturbed[i, k] += epsilon  # change ith element of v_n^[k] and compare this Nnk to the previous one
        Rn_perturbed = compute_R_n(X, V_n_perturbed)
        R_n_minus_l_perturbed = np.delete(np.delete(Rn_perturbed, l, 0), l, 1)  # K-1 x K-1

        numerical_grad[i, :, :] = (R_n_minus_l_perturbed - original_R_n_minus_l) / epsilon

    # compare with analytical formulation
    analytical_grad = np.zeros((N, K - 1, K - 1))
    for i in range(N):
        if l < k:
            for j in range(l):
                analytical_grad[i, k - 1, j] += C_yy[k * N: (k + 1) * N, j * N: (j + 1) * N][i, :] @ V_n[:, j]
                analytical_grad[i, j, k - 1] += C_yy[k * N: (k + 1) * N, j * N: (j + 1) * N][i, :] @ V_n[:, j]
            for j in range(l + 1, K):
                analytical_grad[i, k - 1, j - 1] += C_yy[k * N: (k + 1) * N, j * N: (j + 1) * N][i, :] @ V_n[:, j]
                analytical_grad[i, j - 1, k - 1] += C_yy[k * N: (k + 1) * N, j * N: (j + 1) * N][i, :] @ V_n[:, j]
        elif l > k:
            for j in range(l):
                analytical_grad[i, k, j] += C_yy[k * N: (k + 1) * N, j * N: (j + 1) * N][i, :] @ V_n[:, j]
                analytical_grad[i, j, k] += C_yy[k * N: (k + 1) * N, j * N: (j + 1) * N][i, :] @ V_n[:, j]
            for j in range(l + 1, K):
                analytical_grad[i, k, j - 1] += C_yy[k * N: (k + 1) * N, j * N: (j + 1) * N][i, :] @ V_n[:, j]
                analytical_grad[i, j - 1, k] += C_yy[k * N: (k + 1) * N, j * N: (j + 1) * N][i, :] @ V_n[:, j]

    return numerical_grad, analytical_grad


def numerical_hessian_k_equal_l(X, V_n, k, epsilon=1e-6):
    """
    Numerically compute the gradient of N_n^[l] w.r.t v_n^[k](i), for l != k.

    Parameters:
        V_n: N x K matrix. V_n[:,k] is v_n^[k], i.e., the nth column of V^[k]
    - epsilon: small value for finite difference

    Returns:
    - grad: gradient vector of size N (∂det(X)/∂w_k)
    """

    N, T, K = X.shape
    C_yy = calculate_Cyy(X)

    original_R_n = compute_R_n(X, V_n)
    original_R_n_minus_k = np.delete(np.delete(original_R_n, k, 0), k,
                                     1)  # K-1 x K-1 matrix Phi_j on p.445, his j=our l
    original_N_n_k = compute_Nnk(C_yy, V_n, k)
    original_R_n_tilde_k = original_N_n_k @ np.linalg.inv(original_R_n_minus_k) @ original_N_n_k.T  # Q_j on p.445
    original_grad = - 2 * np.linalg.det(original_R_n_minus_k) * V_n[:, k].T @ (original_R_n_tilde_k - np.eye(N))

    numerical_hessiankk = np.zeros((N, N))
    for i in range(N):
        V_n_perturbed = V_n.copy()
        V_n_perturbed[i, k] += epsilon  # change ith element of v_n^[k] and compare this Nnk to the previous one
        R_n_perturbed = compute_R_n(X, V_n_perturbed)
        R_n_minus_k_perturbed = np.delete(np.delete(R_n_perturbed, k, 0), k,
                                          1)  # K-1 x K-1 matrix Phi_j on p.445, his j=our l
        N_n_k_perturbed = compute_Nnk(C_yy, V_n_perturbed, k)
        R_n_tilde_k_perturbed = N_n_k_perturbed @ np.linalg.inv(R_n_minus_k_perturbed) @ N_n_k_perturbed.T
        grad_perturbed = - 2 * np.linalg.det(R_n_minus_k_perturbed) * V_n_perturbed[:, k].T @ (R_n_tilde_k_perturbed - np.eye(N))

        numerical_hessiankk[i, :] = (grad_perturbed - original_grad) / epsilon

    # compare with analytical formulation
    analytical_hessiankk = -2 * np.linalg.det(original_R_n_minus_k) * (original_R_n_tilde_k - np.eye(N))

    return numerical_hessiankk, analytical_hessiankk


# Example usage:
def test_compare_hessians():
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

    n = 2
    k = 5
    l = 3

    numerical_grad, analytical_grad = numerical_gradient_Nnl_by_vnki(X, V[:, n, :], k, l, epsilon=1e-6)
    print(
        f"Difference of symmetric and analytical gradient of N_{n}^[{l}] with respect to v_{n}^[{k}]:"
        f"\n{np.linalg.norm(numerical_grad - analytical_grad)}")

    numerical_grad, analytical_grad = numerical_gradient_Rnminusl_by_vnki(X, V[:, n, :], k, l, epsilon=1e-6)
    print(
        f"Difference of symmetric and analytical gradient of R_{n}^[-{l}] with respect to v_{n}^[{k}]:"
        f"\n{np.linalg.norm(numerical_grad - analytical_grad)}")

    numerical_hessiankk, analytical_hessiankk = numerical_hessian_k_equal_l(X, V[:, n, :], k, epsilon=1e-6)
    print(
        f"Difference of Hessians for k = l with respect to v_{n}^[{k}]:"
        f"\n{np.linalg.norm(numerical_hessiankk - analytical_hessiankk)}")
