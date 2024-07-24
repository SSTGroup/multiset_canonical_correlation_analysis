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
    simulations.save_joint_isi_and_runtime_results(N, K, T, n_montecarlo, scenarios, alpha=alpha, beta=beta)


def test_save_violation_results_from_multiple_files_in_one_file():
    simulations.save_violation_results_from_multiple_files_in_one_file(10, 50)


def test_save_different_R_results_from_multiple_files_in_one_file():
    simulations.save_different_R_results_from_multiple_files_in_one_file(10, 50)


def test_write_results_in_table():
    K = 10
    n_montecarlo = 50

    simulations.write_results_in_latex_table(K, n_montecarlo)
