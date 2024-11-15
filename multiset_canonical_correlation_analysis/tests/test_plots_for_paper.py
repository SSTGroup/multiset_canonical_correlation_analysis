from .. import simulations
from .. import plots_for_paper

import numpy as np


def test_plot_all_eigenvalues_for_paper():
    N = 5
    K = 100

    scv_cov1 = simulations.scv_covs_with_same_eigenvalues_same_eigenvectors_rank_K(N, K,
                                                                                   alpha=[0.9, 0.9, 0.9, 0.9, 0.9],
                                                                                   beta=0.0)
    scv_cov2 = simulations.scv_covs_with_same_eigenvalues_different_eigenvectors_rank_K(N, K,
                                                                                        alpha=[0.9, 0.9, 0.9, 0.9, 0.9],
                                                                                        beta=0.0)
    alpha = 1 - (K - np.array([10, 15, 20, 25, 30])) / (K - 1)
    scv_cov3 = simulations.scv_covs_with_rank_R(N, K, 1, alpha=alpha, beta=0.0)
    alpha = 1 - (K - np.array([0.1, 0.15, 0.2, 0.25, 0.3])) / (K - 1)
    scv_cov4 = simulations.scv_covs_with_rank_R(N, K, 1, alpha=alpha, beta=0.0)
    scv_cov5 = simulations.scv_covs_with_rank_R(N, K, 50, alpha=[0.9, 0.85, 0.8, 0.75, 0.7], beta=0.0)

    plots_for_paper.plot_all_eigenvalues_for_paper(scv_cov1, scv_cov2, scv_cov3, scv_cov4, scv_cov5,
                                                    filename=f'evs_K_{K}')


def test_plot_results_for_paper():
    K = 100
    n_montecarlo = 50

    simulations.save_violation_results_from_multiple_files_in_one_file(K, n_montecarlo)
    simulations.save_different_R_results_from_multiple_files_in_one_file(K, n_montecarlo)
    plots_for_paper.plot_results_for_paper(K, n_montecarlo, save=False)
