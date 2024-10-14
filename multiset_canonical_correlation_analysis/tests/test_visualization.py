from .. import simulations, visualization


def test_plot_eigenvalues():
    N = 5
    K = 100
    beta = 0.0
    alpha = [0.9, 0.85, 0.8, 0.75, 0.7]

    scv_cov = simulations.scv_covs_with_same_maximum_eigenvalue(N, K)
    visualization.plot_eigenvalues(scv_cov, filename=f'evs_K_{K}_same_lambda_max')

    scv_cov = simulations.scv_covs_with_same_minimum_eigenvalue(N, K)
    visualization.plot_eigenvalues(scv_cov, filename=f'evs_K_{K}_same_lambda_min')

    scv_cov = simulations.scv_covs_with_rank_R(N, K, 1, alpha, beta)
    visualization.plot_eigenvalues(scv_cov, filename=f'evs_K_{K}_rank_1')

    scv_cov = simulations.scv_covs_with_rank_R(N, K, K, alpha, beta)
    visualization.plot_eigenvalues(scv_cov, filename=f'evs_K_{K}_rank_{K}')


def test_plot_results():
    K = 10
    n_montecarlo = 50
    save = False

    simulations.save_violation_results_from_multiple_files_in_one_file(K, n_montecarlo)
    visualization.plot_results_with_errorbars_for_violations(K, n_montecarlo, save)

    simulations.save_different_R_results_from_multiple_files_in_one_file(K, n_montecarlo)
    visualization.plot_results_with_errorbars_for_different_R(K, n_montecarlo, save)


def test_plot_all_eigenvalues_for_paper():
    N = 5
    K = 100

    scv_cov1 = simulations.scv_covs_with_same_eigenvalues_different_eigenvectors_rank_R(N, K, 1,
                                                                                        alpha=[0.9, 0.9, 0.9, 0.9, 0.9],
                                                                                        beta=0.0)
    scv_cov2 = simulations.scv_covs_with_same_eigenvalues_different_eigenvectors_rank_R(N, K, K, alpha=[1, 1, 1, 1, 1],
                                                                                        beta=0.0)
    scv_cov3 = simulations.scv_covs_for_maxvar_minvar(N, K, alpha=[10, 15, 20, 25, 30])
    scv_cov4 = simulations.scv_covs_for_maxvar_minvar(N, K, alpha=np.array([0.1, 0.15, 0.2, 0.25, 0.3]))

    visualization.plot_all_eigenvalues_for_paper(scv_cov1, scv_cov2, scv_cov3, scv_cov4, filename=f'evs_K_{K}')
def test_write_results_in_table():
    K = 10
    n_montecarlo = 50

    visualization.write_results_in_latex_table(K, n_montecarlo)
