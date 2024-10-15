from .. import simulations
from .. import plots_for_paper


def test_plot_results_for_paper():
    K = 100
    n_montecarlo = 50

    simulations.save_violation_results_from_multiple_files_in_one_file(K, n_montecarlo)
    simulations.save_different_R_results_from_multiple_files_in_one_file(K, n_montecarlo)
    plots_for_paper.plot_results_for_paper(K, n_montecarlo, save=True)
