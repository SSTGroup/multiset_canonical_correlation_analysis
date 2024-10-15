from .. import simulations, visualization


def test_plot_eigenvalues():
    N = 5
    K = 100

    scv_cov = simulations.scv_covs_with_rank_R(N, K, 1, alpha=[0.9, 0.9, 0.9, 0.9, 0.9], beta=0.0)
    visualization.plot_eigenvalues(scv_cov, title=f'evs_K_{K}_rank_1', show=False)

    scv_cov = simulations.scv_covs_with_rank_R(N, K, K, alpha=[0.9, 0.9, 0.9, 0.9, 0.9], beta=0.0)
    visualization.plot_eigenvalues(scv_cov, title=f'evs_K_{K}_rank_{K}', show=False)

    scv_cov = simulations.scv_covs_for_maxvar_minvar(N, K, alpha=[10, 15, 20, 25, 30])
    visualization.plot_eigenvalues(scv_cov, title=f'evs_K_{K}_same_lambda_max', show=False)

    scv_cov = simulations.scv_covs_for_maxvar_minvar(N, K, alpha=[0.1, 0.15, 0.2, 0.25, 0.3])
    visualization.plot_eigenvalues(scv_cov, title=f'evs_K_{K}_same_lambda_min', show=True)


def test_plot_results():
    K = 100
    n_montecarlo = 50
    save = False

    simulations.save_violation_results_from_multiple_files_in_one_file(K, n_montecarlo)
    visualization.plot_results_with_errorbars_for_violations(K, n_montecarlo, save)

    simulations.save_different_R_results_from_multiple_files_in_one_file(K, n_montecarlo)
    visualization.plot_results_with_errorbars_for_different_R(K, n_montecarlo, save)


def test_write_results_in_table():
    K = 100
    n_montecarlo = 50

    visualization.write_results_in_latex_table(K, n_montecarlo)
