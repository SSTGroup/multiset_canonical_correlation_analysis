from .. import simulations
from .. import visualization


def test_scv_covs_with_same_maximum_eigenvalue():
    N = 5
    K = 10
    scv_cov = simulations.scv_covs_with_same_maximum_eigenvalue(N, K)
    visualization.plot_eigenvalues(scv_cov)


def test_scv_covs_with_rank_R():
    N = 5
    K = 10
    R = 1
    beta = 0.0
    alpha = [0.9, 0.85, 0.8, 0.75, 0.7]
    scv_cov = simulations.scv_covs_with_rank_R(N, K, R, alpha, beta)
    visualization.plot_eigenvalues(scv_cov)


def test_save_results():
    N = 5  # SCVs
    T = 10000  # samples
    n_montecarlo = 50  # runs

    alpha = [0.9, 0.85, 0.8, 0.75, 0.7]
    beta = 0.0
    K = 10  # datasets

    scenarios = ['same_eigenvalues_different_eigenvectors', 'same_eigenvalues_different_eigenvectors_rank_1',
                 'same_eigenvalues_different_eigenvectors_rank_K',
                 'same_maximum_eigenvalue', 'same_minimum_eigenvalue']
    # scenarios += [f'rank_{R}' for R in range(1, 11)]

    simulations.save_joint_isi_and_runtime_results(N, K, T, n_montecarlo, scenarios, alpha=alpha, beta=beta)


def test_one_run_one_algorithm():
    N = 5  # SCVs
    T = 10000  # samples

    alpha = [0.9, 0.85, 0.8, 0.75, 0.7]
    beta = 0.0
    K = 10  # datasets

    algorithm = 'sumcor_kettenring'

    scenario = 'same_minimum_eigenvalue'

    if scenario == 'same_maximum_eigenvalue':
        scv_cov = simulations.scv_covs_with_same_maximum_eigenvalue(N, K)
    elif scenario == 'same_minimum_eigenvalue':
        scv_cov = simulations.scv_covs_with_same_minimum_eigenvalue(N, K)
    elif scenario == 'same_eigenvalues_different_eigenvectors':
        scv_cov = simulations.scv_covs_with_same_eigenvalues_different_eigenvectors(N, K)
    elif scenario == 'same_eigenvalues_different_eigenvectors_rank_1':
        scv_cov = simulations.scv_covs_with_same_eigenvalues_different_eigenvectors_rank_R(N, K, 1, alpha, beta)
    elif scenario == 'same_eigenvalues_different_eigenvectors_rank_K':
        scv_cov = simulations.scv_covs_with_same_eigenvalues_different_eigenvectors_rank_R(N, K, K,
                                                                                           alpha=[1, 1, 1, 1, 1],
                                                                                           beta=0.0)
    elif scenario == 'same_eigenvalues_different_sign_eigenvectors':
        scv_cov = simulations.scv_covs_with_same_eigenvalues_different_sign_eigenvectors(N, K)
    elif scenario[0:5] == 'rank_':
        scv_cov = simulations.scv_covs_with_rank_R(N, K, int(scenario[5:]), alpha, beta)
    else:
        raise AssertionError(f"scenario '{scenario}' does not exist")

    X, A, S = simulations.generate_datasets_from_covariance_matrices(scv_cov, T)

    M = simulations.mcca(X, algorithm)[0]
    W = np.moveaxis(M, [0, 1, 2], [1, 0, 2])

    print(f'joint_isi: {_bss_isi(W, A)[1]}')
