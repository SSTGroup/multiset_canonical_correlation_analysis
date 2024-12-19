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

    plots_for_paper.plot_all_eigenvalues_for_paper(scv_cov1, scv_cov2, scv_cov3, scv_cov4, filename=f'evs')


def test_plot_results_for_paper():
    folder = f'K_100_T_10_true_C'
    n_montecarlo = 50

    simulations.save_violation_results_from_multiple_files_in_one_file(folder, n_montecarlo)
    simulations.save_different_R_results_from_multiple_files_in_one_file(folder, n_montecarlo)
    plots_for_paper.plot_results_for_paper(folder, n_montecarlo, save=False)


def test_plot_true_estimated_results_for_paper():
    folder1 = f'K_100_T_10_true_C'
    folder2 = f'K_100_T_10000'

    n_montecarlo = 50

    simulations.save_violation_results_from_multiple_files_in_one_file(folder1, n_montecarlo)
    simulations.save_violation_results_from_multiple_files_in_one_file(folder2, n_montecarlo)
    plots_for_paper.plot_true_estimated_results_for_paper(folder1, folder2, n_montecarlo, save=False)
