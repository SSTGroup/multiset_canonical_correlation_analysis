import numpy as np
from scipy.linalg import block_diag

from independent_vector_analysis.helpers_iva import _bss_isi
import time

from .. import simulations, mcca


def test_one_run_all_algorithms():
    N = 5  # SCVs
    T = 10000  # samples

    K = 100  # datasets

    scenario = 'different_lambda_max'

    use_true_C_xx = False

    if scenario == 'same_eigenvalues_same_eigenvectors':
        scv_cov = simulations.scv_covs_with_same_eigenvalues_same_eigenvectors_rank_K(N, K, alpha=[1, 1, 1, 1, 1],
                                                                                      beta=0.0)
    elif scenario == 'same_eigenvalues_different_eigenvectors':
        scv_cov = simulations.scv_covs_with_same_eigenvalues_different_eigenvectors_rank_K(N, K, alpha=[1, 1, 1, 1, 1],
                                                                                           beta=0.0)
    elif scenario == 'different_lambda_min':
        alpha = 1 - (K - np.array([0.1, 0.15, 0.2, 0.25, 0.3])) / (K - 1)
        scv_cov = simulations.scv_covs_with_rank_R(N, K, 1, alpha=alpha, beta=0.0)
    elif scenario == 'different_lambda_max':
        alpha = 1 - (K - np.array([10, 15, 20, 25, 30])) / (K - 1)
        scv_cov = simulations.scv_covs_with_rank_R(N, K, 1, alpha=alpha, beta=0.0)
    elif scenario[0:5] == 'rank_':
        scv_cov = simulations.scv_covs_with_rank_R(N, K, int(scenario[5:]), alpha=[0.9, 0.85, 0.8, 0.75, 0.7], beta=0.0)
    else:
        raise AssertionError(f"scenario '{scenario}' does not exist")

    X, A, S = simulations.generate_datasets_from_covariance_matrices(scv_cov, T)

    if use_true_C_xx:
        # true joint SCV covariance matrix
        joint_scv_cov = block_diag(*list(scv_cov.T))

        # make the permutation matrix
        P = np.zeros((N * K, N * K))
        for n in range(N):
            for k in range(K):
                P[n + k * N, n * K + k] = 1

        # generate C_xx from true C_ss
        C_ss = P @ joint_scv_cov @ P.T
        A_joint = block_diag(*list(A.T)).T
        C_xx = A_joint @ C_ss @ A_joint.T
    else:
        C_xx = None

    t_start = time.process_time()
    M_sumcor = mcca.mcca(X, 'sumcor', verbose=True, C_xx=C_xx)[0]
    W = np.moveaxis(M_sumcor, [0, 1, 2], [1, 0, 2])
    t_end = time.process_time()

    print(f'joint_isi sumcor: {_bss_isi(W, A)[1]}')
    print(f'runtime: {t_end - t_start}')

    t_start = time.process_time()
    M_maxvar = mcca.mcca(X, 'maxvar', verbose=True, C_xx=C_xx)[0]
    W = np.moveaxis(M_maxvar, [0, 1, 2], [1, 0, 2])
    t_end = time.process_time()

    print(f'joint_isi maxvar: {_bss_isi(W, A)[1]}')
    print(f'runtime: {t_end - t_start}')

    t_start = time.process_time()
    M = mcca.mcca(X, 'minvar', verbose=True, C_xx=C_xx)[0]
    W = np.moveaxis(M, [0, 1, 2], [1, 0, 2])
    t_end = time.process_time()

    print(f'joint_isi minvar: {_bss_isi(W, A)[1]}')
    print(f'runtime: {t_end - t_start}')

    t_start = time.process_time()
    M = mcca.mcca(X, 'ssqcor', verbose=True, C_xx=C_xx)[0]
    W = np.moveaxis(M, [0, 1, 2], [1, 0, 2])
    t_end = time.process_time()

    print(f'joint_isi ssqcor: {_bss_isi(W, A)[1]}')
    print(f'runtime: {t_end - t_start}')

    t_start = time.process_time()
    M = mcca.mcca(X, 'genvar', verbose=True, C_xx=C_xx)[0]
    W = np.moveaxis(M, [0, 1, 2], [1, 0, 2])
    t_end = time.process_time()

    print(f'joint_isi genvar: {_bss_isi(W, A)[1]}')
    print(f'runtime: {t_end - t_start}')


def test_save_paper_results():
    N = 5  # SCVs
    T = 10  # samples
    n_montecarlo = 50  # runs

    K = 100  # datasets

    use_true_C_xx = True

    scenarios = ['same_eigenvalues_same_eigenvectors',
                 'same_eigenvalues_different_eigenvectors',
                 'different_lambda_max', 'different_lambda_min']
    scenarios += [f'rank_{R}' for R in [1, 2, 5, 10, 20, 50]]
    simulations.save_joint_isi_and_runtime_results(N, K, T, n_montecarlo, scenarios, use_true_C_xx)
