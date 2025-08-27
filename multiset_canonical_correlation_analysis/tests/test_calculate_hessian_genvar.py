import numpy as np
from scipy.linalg import sqrtm, block_diag

from .test_calculate_genvar_gradients import compute_R_n, calculate_Cyy


def compute_Nnl(C_yy, V_n, l):
    N, K = V_n.shape
    # [ C_yy^[k,1] m_n^[1], ..., C_yy^[k,k-1] m_n^[k-1], C_yy^[k,k+1] m_n^[k+1], ..., C_yy^[k,K] m_n^[K] ]
    N_n_l = []
    for i in range(K):
        if i != l:
            N_n_l.append(C_yy[l * N: (l + 1) * N, i * N: (i + 1) * N] @ V_n[:, i])
    N_n_l = np.array(N_n_l).T  # called N_j in Kettenring's paper p.445, his j is our l

    return N_n_l


def compute_Nn_minusl_k(C_yy, V_n, k, l):
    N, K = V_n.shape
    # [ C_yy^[k,1] m_n^[1], ..., C_yy^[k,k-1] m_n^[k-1], C_yy^[k,k+1] m_n^[k+1], ..., C_yy^[k,K] m_n^[K] ]
    N_n_minusl_k = []
    for i in range(K):
        if i != k and i != l:
            N_n_minusl_k.append(C_yy[k * N: (k + 1) * N, i * N: (i + 1) * N] @ V_n[:, i])
    N_n_minusl_k = np.array(N_n_minusl_k).T  # called N_j in Kettenring's paper p.445, his j is our l

    return N_n_minusl_k


def compute_gradient_Nnl_by_vnki(X, V_n, k, l, epsilon=1e-6):
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

    original_Nnk = compute_Nnl(C_yy, V_n, l)

    numerical_grad = np.zeros((N, N, K - 1))
    for i in range(N):
        V_n_perturbed = V_n.copy()
        V_n_perturbed[i, k] += epsilon  # change ith element of v_n^[k] and compare this Nnk to the previous one
        Nnk_perturbed = compute_Nnl(C_yy, V_n_perturbed, l)

        numerical_grad[i, :, :] = (Nnk_perturbed - original_Nnk) / epsilon

    # compare with analytical formulation
    analytical_grad = np.zeros((N, N, K - 1))
    for i in range(N):
        if l < k:
            analytical_grad[i, :, k - 1] = C_yy[l * N: (l + 1) * N, k * N: (k + 1) * N][:,i]
        elif l > k:
            analytical_grad[i, :, k] = C_yy[l * N: (l + 1) * N, k * N: (k + 1) * N][:,i]

    return numerical_grad, analytical_grad


def compute_gradient_Rnminusl_by_vnki(X, V_n, k, l, epsilon=1e-6):
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
    original_R_n_minus_l = np.delete(np.delete(original_Rn, l, 0), l, 1)  # K-1 x K-1

    numerical_grad = np.zeros((N, K - 1, K - 1))
    for i in range(N):
        V_n_perturbed = V_n.copy()
        V_n_perturbed[i, k] += epsilon  # change ith element of v_n^[k] and compare this Nnk to the previous one
        Rn_perturbed = compute_R_n(X, V_n_perturbed)
        R_n_minus_l_perturbed = np.delete(np.delete(Rn_perturbed, l, 0), l, 1)  # K-1 x K-1

        numerical_grad[i, :, :] = (R_n_minus_l_perturbed - original_R_n_minus_l) / epsilon

    # compare with analytical formulation
    nameless_variable = []
    for i in range(K):
        if i != l:
            nameless_variable.append(C_yy[k * N: (k + 1) * N, i * N: (i + 1) * N] @ V_n[:, i])
    nameless_variable = np.array(nameless_variable).T
    analytical_grad = np.zeros((N, K - 1, K - 1))
    for i in range(N):
        e_K1_k = np.zeros(K-1)
        if l < k:
            e_K1_k[k-1] = 1
        elif l > k:
            e_K1_k[k] = 1
        analytical_grad[i, :, :] = np.outer(e_K1_k, nameless_variable[i,:]) + np.outer(e_K1_k, nameless_variable[i,:]).T

    return numerical_grad, analytical_grad


def compute_gradient_det_Rnminusl_by_vnki(X, V_n, k, l, epsilon=1e-6):
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

    original_R_n = compute_R_n(X, V_n)
    original_R_n_minus_l = np.delete(np.delete(original_R_n, l, 0), l, 1)  # K-1 x K-1
    original_det_R_n_minus_l = np.linalg.det(original_R_n_minus_l)

    numerical_grad = np.zeros(N)
    for i in range(N):
        V_n_perturbed = V_n.copy()
        V_n_perturbed[i, k] += epsilon  # change ith element of v_n^[k] and compare this Nnk to the previous one
        Rn_perturbed = compute_R_n(X, V_n_perturbed)
        R_n_minus_l_perturbed = np.delete(np.delete(Rn_perturbed, l, 0), l, 1)  # K-1 x K-1
        det_R_n_minus_l_perturbed = np.linalg.det(R_n_minus_l_perturbed)

        numerical_grad[i] = (det_R_n_minus_l_perturbed - original_det_R_n_minus_l) / epsilon

    # compare with analytical formulation
    if l == k:
        analytical_grad = np.zeros(N)
    else:
        original_R_n_minus_l_minus_k = np.delete(np.delete(original_R_n, [k, l], 0), [k, l], 1)  # K-1 x K-1
        original_N_n_minus_l_k = compute_Nn_minusl_k(C_yy, V_n, k, l)
        original_R_n_tilde_minus_l_k = original_N_n_minus_l_k @ np.linalg.inv(
            original_R_n_minus_l_minus_k) @ original_N_n_minus_l_k.T
        analytical_grad = -2 * np.linalg.det(original_R_n_minus_l_minus_k) * (
                original_R_n_tilde_minus_l_k - np.eye(N)) @ V_n[:, k]

    return numerical_grad, analytical_grad


def compute_gradient_Rnminusl_inverse_by_vnki(X, V_n, k, l, epsilon=1e-6):
    """
    Numerically compute the gradient of (R_n^[-l])^(-1) w.r.t v_n^[k](i), for l != k.

    Parameters:
        V_n: N x K matrix. V_n[:,k] is v_n^[k], i.e., the nth column of V^[k]
    - epsilon: small value for finite difference

    Returns:
    - grad: gradient vector of size N (∂det(X)/∂w_k)
    """

    N, T, K = X.shape
    C_yy = calculate_Cyy(X)

    original_R_n = compute_R_n(X, V_n)
    original_R_n_minus_l = np.delete(np.delete(original_R_n, l, 0), l, 1)  # K-1 x K-1
    original_R_n_minus_l_inverse = np.linalg.inv(original_R_n_minus_l)

    numerical_grad = np.zeros((N, K - 1, K - 1))
    for i in range(N):
        V_n_perturbed = V_n.copy()
        V_n_perturbed[i, k] += epsilon  # change ith element of v_n^[k] and compare this Nnk to the previous one
        R_n_perturbed = compute_R_n(X, V_n_perturbed)
        R_n_minus_l_perturbed = np.delete(np.delete(R_n_perturbed, l, 0), l, 1)  # K-1 x K-1
        R_n_minus_l_inverse_perturbed = np.linalg.inv(R_n_minus_l_perturbed)

        numerical_grad[i, :, :] = (R_n_minus_l_inverse_perturbed - original_R_n_minus_l_inverse) / epsilon

    # compare with analytical formulation
    Rnminusl_by_vnki = compute_gradient_Rnminusl_by_vnki(X, V_n, k, l, epsilon)[1]
    analytical_grad = np.zeros((N, K - 1, K - 1))
    for i in range(N):
        analytical_grad[i, :, :] = - np.linalg.inv(original_R_n_minus_l) @ Rnminusl_by_vnki[i, :, :] @ np.linalg.inv(
            original_R_n_minus_l)

    return numerical_grad, analytical_grad


def compute_gradient_Rntildel_by_vnki(X, V_n, k, l, epsilon=1e-6):
    """
    Numerically compute the gradient of R~_n^[l] w.r.t v_n^[k](i), for l != k.

    Parameters:
        V_n: N x K matrix. V_n[:,k] is v_n^[k], i.e., the nth column of V^[k]
    - epsilon: small value for finite difference

    Returns:
    - grad: gradient vector of size N (∂det(X)/∂w_k)
    """

    N, T, K = X.shape
    C_yy = calculate_Cyy(X)

    original_N_n_l = compute_Nnl(C_yy, V_n, l)
    original_R_n = compute_R_n(X, V_n)
    original_R_n_minus_l = np.delete(np.delete(original_R_n, l, 0), l, 1)  # K-1 x K-1
    original_R_n_tilde_l = original_N_n_l @ np.linalg.inv(original_R_n_minus_l) @ original_N_n_l.T

    numerical_grad = np.zeros((N, N, N))
    for i in range(N):
        V_n_perturbed = V_n.copy()
        V_n_perturbed[i, k] += epsilon  # change ith element of v_n^[k] and compare this Nnk to the previous one
        N_n_l_perturbed = compute_Nnl(C_yy, V_n_perturbed, l)
        R_n_perturbed = compute_R_n(X, V_n_perturbed)
        R_n_minus_l_perturbed = np.delete(np.delete(R_n_perturbed, l, 0), l, 1)  # K-1 x K-1
        R_n_tilde_l_perturbed = N_n_l_perturbed @ np.linalg.inv(R_n_minus_l_perturbed) @ N_n_l_perturbed.T

        numerical_grad[i, :, :] = (R_n_tilde_l_perturbed - original_R_n_tilde_l) / epsilon

    # compare with analytical formulation
    Nnl_by_vnki = compute_gradient_Nnl_by_vnki(X, V_n, k, l, epsilon)[1]
    Rnminusl_inverse_by_vnki = compute_gradient_Rnminusl_inverse_by_vnki(X, V_n, k, l, epsilon)[1]
    analytical_grad = np.zeros((N, N, N))
    for i in range(N):
        analytical_grad[i, :, :] = Nnl_by_vnki[i, :, :] @ np.linalg.inv(
            original_R_n_minus_l) @ original_N_n_l.T + original_N_n_l @ Rnminusl_inverse_by_vnki[i, :,
                                                                        :] @ original_N_n_l.T + original_N_n_l @ np.linalg.inv(
            original_R_n_minus_l) @ Nnl_by_vnki[i, :, :].T

    return numerical_grad, analytical_grad


def compute_hessian_det_R_n(X, V_n, k, l, epsilon=1e-6):
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
    original_R_n_minus_l = np.delete(np.delete(original_R_n, l, 0), l, 1)  # K-1 x K-1
    original_N_n_l = compute_Nnl(C_yy, V_n, l)
    original_R_n_tilde_l = original_N_n_l @ np.linalg.inv(original_R_n_minus_l) @ original_N_n_l.T  # Q_j on p.445
    original_grad = - 2 * np.linalg.det(original_R_n_minus_l) * V_n[:, l].T @ (original_R_n_tilde_l - np.eye(N))

    numerical_hessian = np.zeros((N, N))
    for i in range(N):
        V_n_perturbed = V_n.copy()
        V_n_perturbed[i, k] += epsilon  # change ith element of v_n^[k] and compare this Nnk to the previous one
        R_n_perturbed = compute_R_n(X, V_n_perturbed)
        R_n_minus_l_perturbed = np.delete(np.delete(R_n_perturbed, l, 0), l, 1)
        N_n_l_perturbed = compute_Nnl(C_yy, V_n_perturbed, l)
        R_n_tilde_l_perturbed = N_n_l_perturbed @ np.linalg.inv(R_n_minus_l_perturbed) @ N_n_l_perturbed.T
        grad_perturbed = - 2 * np.linalg.det(R_n_minus_l_perturbed) * V_n_perturbed[:, l].T @ (
                R_n_tilde_l_perturbed - np.eye(N))

        numerical_hessian[i, :] = (grad_perturbed - original_grad) / epsilon

    # compare with analytical formulation
    if k == l:
        analytical_hessian = -2 * np.linalg.det(original_R_n_minus_l) * (original_R_n_tilde_l - np.eye(N))
    else:
        det_Rnminusl_by_vnki = compute_gradient_det_Rnminusl_by_vnki(X, V_n, k, l, epsilon)[1]
        Rntildel_by_vnki = compute_gradient_Rntildel_by_vnki(X, V_n, k, l, epsilon)[1]
        analytical_hessian = np.zeros((N, N))
        for i in range(N):
            analytical_hessian[i, :] = -2 * det_Rnminusl_by_vnki[i] * V_n[:, l].T @ (
                        original_R_n_tilde_l - np.eye(N)) - 2 * np.linalg.det(original_R_n_minus_l) * V_n[:,
                                                                                                      l].T @ Rntildel_by_vnki[
                                                                                                             i, :, :]

    return numerical_hessian, analytical_hessian


def compute_gradient_Lagragrian(X, V_n, H_n_k, k, l, epsilon=1e-6):
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
    original_R_n_minus_l = np.delete(np.delete(original_R_n, l, 0), l, 1)  # K-1 x K-1
    original_N_n_l = compute_Nnl(C_yy, V_n, l)
    original_R_n_tilde_l = original_N_n_l @ np.linalg.inv(original_R_n_minus_l) @ original_N_n_l.T  # Q_j on p.445
    original_Lagragrian = np.linalg.det(original_R_n) +1    *V_n[:, l].T @ (original_R_n_tilde_l - np.eye(N))

    numerical_hessian = np.zeros((N, N))
    for i in range(N):
        V_n_perturbed = V_n.copy()
        V_n_perturbed[i, k] += epsilon  # change ith element of v_n^[k] and compare this Nnk to the previous one
        R_n_perturbed = compute_R_n(X, V_n_perturbed)
        R_n_minus_l_perturbed = np.delete(np.delete(R_n_perturbed, l, 0), l, 1)
        N_n_l_perturbed = compute_Nnl(C_yy, V_n_perturbed, l)
        R_n_tilde_l_perturbed = N_n_l_perturbed @ np.linalg.inv(R_n_minus_l_perturbed) @ N_n_l_perturbed.T
        grad_perturbed = - 2 * np.linalg.det(R_n_minus_l_perturbed) * V_n_perturbed[:, l].T @ (
                R_n_tilde_l_perturbed - np.eye(N))

        numerical_hessian[i, :] = (grad_perturbed - original_grad) / epsilon

    # compare with analytical formulation
    analytical_grad = -2 * np.linalg.det(original_R_n_minus_k) * H_n_k @ (original_R_n_tilde_k - np.eye(N)) @ V_n[:, k]

    return numerical_hessian, analytical_grad


def compute_hessian_Lagragrian(X, V_n, k, l, epsilon=1e-6):
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
    original_R_n_minus_l = np.delete(np.delete(original_R_n, l, 0), l, 1)  # K-1 x K-1
    original_N_n_l = compute_Nnl(C_yy, V_n, l)
    original_R_n_tilde_l = original_N_n_l @ np.linalg.inv(original_R_n_minus_l) @ original_N_n_l.T  # Q_j on p.445
    original_grad = - 2 * np.linalg.det(original_R_n_minus_l) * V_n[:, l].T @ (original_R_n_tilde_l - np.eye(N))

    numerical_hessian = np.zeros((N, N))
    for i in range(N):
        V_n_perturbed = V_n.copy()
        V_n_perturbed[i, k] += epsilon  # change ith element of v_n^[k] and compare this Nnk to the previous one
        R_n_perturbed = compute_R_n(X, V_n_perturbed)
        R_n_minus_l_perturbed = np.delete(np.delete(R_n_perturbed, l, 0), l, 1)
        N_n_l_perturbed = compute_Nnl(C_yy, V_n_perturbed, l)
        R_n_tilde_l_perturbed = N_n_l_perturbed @ np.linalg.inv(R_n_minus_l_perturbed) @ N_n_l_perturbed.T
        grad_perturbed = - 2 * np.linalg.det(R_n_minus_l_perturbed) * V_n_perturbed[:, l].T @ (
                R_n_tilde_l_perturbed - np.eye(N))

        numerical_hessian[i, :] = (grad_perturbed - original_grad) / epsilon

    # compare with analytical formulation
    if k == l:
        analytical_hessian = -2 * np.linalg.det(original_R_n_minus_l) * (original_R_n_tilde_l - np.eye(N))
    else:
        det_Rnminusl_by_vnki = compute_gradient_det_Rnminusl_by_vnki(X, V_n, k, l, epsilon)[1]
        Rntildel_by_vnki = compute_gradient_Rntildel_by_vnki(X, V_n, k, l, epsilon)[1]
        analytical_hessian = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                analytical_hessian[i, j] = -2 * det_Rnminusl_by_vnki[i] * V_n[:, l].T @ (
                        original_R_n_tilde_l[:, j] - np.eye(N)[:, j]) - 2 * np.linalg.det(
                    original_R_n_minus_l) * V_n[:, l].T @ Rntildel_by_vnki[i, :, :][:, j]

    return numerical_hessian, analytical_hessian


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

    numerical_grad, analytical_grad = compute_gradient_Nnl_by_vnki(X, V[:, n, :], k, l, epsilon=1e-6)
    print(
        f"Difference of numerical and analytical gradient of N_{n}^[{l}] with respect to v_{n}^[{k}]:"
        f"\n{np.linalg.norm(numerical_grad - analytical_grad)}")

    numerical_grad, analytical_grad = compute_gradient_Rnminusl_by_vnki(X, V[:, n, :], k, l, epsilon=1e-6)
    print(
        f"Difference of numerical and analytical gradient of R_{n}^[-{l}] with respect to v_{n}^[{k}]:"
        f"\n{np.linalg.norm(numerical_grad - analytical_grad)}")

    numerical_grad, analytical_grad = compute_gradient_det_Rnminusl_by_vnki(X, V[:, n, :], k, l, epsilon=1e-6)
    print(
        f"Difference of numerical and analytical gradient of det(R_{n}^[-{l}]) with respect to v_{n}^[{k}]:"
        f"\n{np.linalg.norm(numerical_grad - analytical_grad)}")

    numerical_grad, analytical_grad = compute_gradient_Rnminusl_inverse_by_vnki(X, V[:, n, :], k, l, epsilon=1e-6)
    print(
        f"Difference of numerical and analytical gradient of (R_{n}^[-{l}])^(-1) with respect to v_{n}^[{k}]:"
        f"\n{np.linalg.norm(numerical_grad - analytical_grad)}")

    numerical_grad, analytical_grad = compute_gradient_Rntildel_by_vnki(X, V[:, n, :], k, l, epsilon=1e-6)
    print(
        f"Difference of numerical and analytical gradient of R~_{n}^[{l}] with respect to v_{n}^[{k}]:"
        f"\n{np.linalg.norm(numerical_grad - analytical_grad)}")

    numerical_hessiankk, analytical_hessiankk = compute_hessian_det_R_n(X, V[:, n, :], k, l, epsilon=1e-6)
    print(
        f"Difference of Hessians for k = {k} and l = {l} with respect to v_{n}^[{k}]:"
        f"\n{np.linalg.norm(numerical_hessiankk - analytical_hessiankk)}")
