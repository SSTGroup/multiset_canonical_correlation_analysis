from .. import simulations
from .. import visualization


def test_scv_covs_with_same_maximum_eigenvalue():
    N = 5
    K = 10
    simulations.scv_covs_with_same_maximum_eigenvalue(N, K)


def test_generate_rank_R_covariance_matrices():
    N = 5
    K = 10
    R = 1
    beta = 0.1
    alpha = [0.9, 0.8, 0.7, 0.6, 0.5]
    scv_cov = simulations.scv_covs_with_rank_R(N, K, R, alpha, beta)
    visualization.plot_eigenvalues(scv_cov)


def test_save_results():
    N = 5  # SCVs
    T = 10000  # samples
    n_montecarlo = 50  # runs

    alpha = [0.9, 0.8, 0.7, 0.6, 0.5]
    beta = 0.0
    K = 10  # datasets

    filename = f'results_K_{K}'

    simulations.save_joint_isi_and_runtime_results(filename, N, K, T, n_montecarlo, alpha=alpha, beta=beta)


def test_write_results_in_table():
    K = 10
    filename = f'results_K_{K}'

    simulations.write_results_in_latex_table(filename)