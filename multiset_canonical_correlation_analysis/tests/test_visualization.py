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



def test_write_results_in_table():
    K = 10
    n_montecarlo = 50

    visualization.write_results_in_latex_table(K, n_montecarlo)
