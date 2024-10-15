from .. import simulations
from .. import visualization


def test_scv_covs_with_same_maximum_eigenvalue():
    N = 5
    K = 10
    scv_cov = simulations.scv_covs_with_same_maximum_eigenvalue(N, K)
    visualization.plot_eigenvalues(scv_cov)


def test_scv_covs_with_rank_R():
    N = 5
    K = 100
    R = 1
    beta = 0.0
    alpha = [0.9, 0.85, 0.8, 0.75, 0.7]
    scv_cov = simulations.scv_covs_with_rank_R(N, K, R, alpha, beta)
    visualization.plot_eigenvalues(scv_cov)


def test_save_results():
    N = 5  # SCVs
    T = 10000  # samples
    n_montecarlo = 50  # runs

    K = 100  # datasets

    scenarios = ['same_eigenvalues_different_eigenvectors_rank_1', 'same_eigenvalues_different_eigenvectors_rank_K',
                 'different_lambda_max', 'different_lambda_min']
    scenarios += [f'rank_{R}' for R in [1, 2, 5, 10]]
    simulations.save_joint_isi_and_runtime_results(N, K, T, n_montecarlo, scenarios)


def test_one_run_one_algorithm():
    N = 5  # SCVs
    T = 10000  # samples

    K = 100  # datasets

    algorithm = 'sumcor'

    scenario = 'same_eigenvalues_different_eigenvectors_rank_K'

    if scenario == 'same_eigenvalues_different_eigenvectors_rank_1':
        scv_cov = simulations.scv_covs_with_same_eigenvalues_different_eigenvectors_rank_R(N, K, 1,
                                                                                           alpha=[0.9, 0.9, 0.9, 0.9,
                                                                                                  0.9],
                                                                                           beta=0.0)
    elif scenario == 'same_eigenvalues_different_eigenvectors_rank_K':
        scv_cov = simulations.scv_covs_with_same_eigenvalues_different_eigenvectors_rank_R(N, K, K,
                                                                                           alpha=[0.9, 0.9, 0.9, 0.9,
                                                                                                  0.9],
                                                                                           beta=0.0)
    elif scenario == 'different_lambda_min':
        scv_cov = simulations.scv_covs_for_maxvar_minvar(N, K, alpha=[0.1, 0.15, 0.2, 0.25, 0.3])
    elif scenario == 'different_lambda_max':
        scv_cov = simulations.scv_covs_for_maxvar_minvar(N, K, alpha=[10, 15, 20, 25, 30])
    elif scenario[0:5] == 'rank_':
        scv_cov = simulations.scv_covs_with_rank_R(N, K, int(scenario[5:]), alpha=[0.9, 0.85, 0.8, 0.75, 0.7], beta=0.0)
    else:
        raise AssertionError(f"scenario '{scenario}' does not exist")

    X, A, S = simulations.generate_datasets_from_covariance_matrices(scv_cov, T)

    t_start = time.process_time()
    M = simulations.mcca(X, algorithm, verbose=True)[0]
    W = np.moveaxis(M, [0, 1, 2], [1, 0, 2])
    t_end = time.process_time()

    print(f'joint_isi: {_bss_isi(W, A)[1]}')
    print(f'runtime: {t_end - t_start}')
