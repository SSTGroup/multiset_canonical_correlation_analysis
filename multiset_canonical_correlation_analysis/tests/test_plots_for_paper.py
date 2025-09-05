from .. import simulations
from .. import plots_for_paper

import numpy as np


def test_plot_all_eigenvalues_for_paper():
    N = 5
    K = 100

    scv_cov1 = simulations.scv_covs_with_same_eigenvalues_same_eigenvectors_rank_K(N, K, alpha=[1, 1, 1, 1, 1],
                                                                                   beta=0.0)

    scv_cov2 = simulations.scv_covs_with_same_eigenvalues_different_eigenvectors_rank_K(N, K, alpha=[1, 1, 1, 1, 1],
                                                                                        beta=0.0)

    alpha = 1 - (K - np.array([0.1, 0.15, 0.2, 0.25, 0.3])) / (K - 1)
    scv_cov3 = simulations.scv_covs_with_rank_R(N, K, 1, alpha=alpha, beta=0.0)

    alpha = 1 - (K - np.array([10, 15, 20, 25, 30])) / (K - 1)
    scv_cov4 = simulations.scv_covs_with_rank_R(N, K, 1, alpha=alpha, beta=0.0)

    plots_for_paper.plot_all_eigenvalues_for_paper(scv_cov1, scv_cov2, scv_cov3, scv_cov4)  # , filename=f'evs')


def test_plot_all_eigenvalues_rankR_for_paper():
    N = 5
    K = 100

    scv_cov1 = simulations.scv_covs_with_rank_R(N, K, 1, alpha=[0.9, 0.85, 0.8, 0.75, 0.7], beta=0.0)
    scv_cov2 = simulations.scv_covs_with_rank_R(N, K, 10, alpha=[0.9, 0.85, 0.8, 0.75, 0.7], beta=0.0)
    scv_cov3 = simulations.scv_covs_with_rank_R(N, K, 20, alpha=[0.9, 0.85, 0.8, 0.75, 0.7], beta=0.0)
    scv_cov4 = simulations.scv_covs_with_rank_R(N, K, 50, alpha=[0.9, 0.85, 0.8, 0.75, 0.7], beta=0.0)

    plots_for_paper.plot_all_eigenvalues_rank_R_for_paper(scv_cov1, scv_cov2, scv_cov3, scv_cov4)  # ,
    # filename=f'evs_different_R')


def test_plot_identification_conditions_for_paper():
    n_montecarlo = 50

    scenarios = ['same_eigenvalues_same_eigenvectors',
                 'same_eigenvalues_different_eigenvectors',
                 'different_lambda_max', 'different_lambda_min']

    folder = f'K_100_V_10_true_C'

    simulations.save_violation_results_from_multiple_files_in_one_file(folder, scenarios, n_montecarlo)
    plots_for_paper.plot_identification_conditions_for_paper(folder, n_montecarlo, save=False)


def test_plot_rank_R_results_for_paper():
    n_montecarlo = 50

    folder1 = f'K_100_V_10000'
    folder2 = f'K_100_V_1000000'

    simulations.save_different_R_results_from_multiple_files_in_one_file(folder1, n_montecarlo)
    simulations.save_different_R_results_from_multiple_files_in_one_file(folder2, n_montecarlo)
    plots_for_paper.plot_rank_R_results_for_paper(folder1, folder2, n_montecarlo, save=False)


def test_plot_results_for_different_samples():
    folder = f'K_100_V_'
    V_values = [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000, 1000000]

    n_montecarlo = 50

    scenarios = ['different_lambda_max', 'different_lambda_min']

    for V in V_values:
        simulations.save_violation_results_from_multiple_files_in_one_file(folder + str(V), scenarios, n_montecarlo)

    plots_for_paper.plot_different_samples_results_for_paper(folder, V_values, n_montecarlo, save=False)
