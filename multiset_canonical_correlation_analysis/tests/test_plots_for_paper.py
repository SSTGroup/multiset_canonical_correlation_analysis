import numpy as np

from .. import simulations
from .. import plots_for_paper


def test_plot_results_for_paper():
    K = 100
    n_montecarlo = 50

    simulations.save_violation_results_from_multiple_files_in_one_file(K, n_montecarlo)
    simulations.save_different_R_results_from_multiple_files_in_one_file(K, n_montecarlo)
    plots_for_paper.plot_results_for_paper(K, n_montecarlo, save=True)


def test_plot_all_eigenvalues_for_paper():
    N = 5
    K = 100

    scv_cov1 = simulations.scv_covs_with_same_eigenvalues_different_eigenvectors_rank_R(N, K, 1,
                                                                                        alpha=[0.9, 0.9, 0.9, 0.9, 0.9],
                                                                                        beta=0.0)
    scv_cov2 = simulations.scv_covs_with_same_eigenvalues_different_eigenvectors_rank_R(N, K, K,
                                                                                        alpha=[0.9, 0.9, 0.9, 0.9, 0.9],
                                                                                        beta=0.0)
    scv_cov3 = simulations.scv_covs_for_maxvar_minvar(N, K, alpha=[10, 15, 20, 25, 30])
    scv_cov4 = simulations.scv_covs_for_maxvar_minvar(N, K, alpha=np.array([0.1, 0.15, 0.2, 0.25, 0.3]))

    plots_for_paper.plot_all_eigenvalues_for_paper(scv_cov1, scv_cov2, scv_cov3, scv_cov4, filename=f'evs_K_{K}')
