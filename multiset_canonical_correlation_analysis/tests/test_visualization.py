import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt

from .. import visualization


def test_plot_joint_isi_over_R():
    filename = 'K_10/K_10'
    results = np.load(Path(Path(__file__).parent.parent.parent, f'simulation_results/{filename}.npy'),
                      allow_pickle=True).item()
    visualization.plot_joint_isi_over_R(results)


def test_quick_and_dirty_plot2():
    results = np.load(Path(Path(__file__).parent.parent.parent, f'simulation_results/K_100/K_100.npy'),
                      allow_pickle=True).item()
    results_rank_1 = results['rank_1']
    R_range = [1]

    plt.figure()
    for key in ['maxvar', 'minvar', 'genvar', 'ssqcor', 'sumcor', 'ivag']:
        plt.errorbar(R_range, np.mean(results_rank_1[key]['joint_isi']),
                     np.std(results_rank_1[key]['joint_isi']),
                     linestyle=':', fmt='s', capsize=3, label=f'{key}')
    plt.xticks([1])
    plt.xlabel(r'R')
    plt.ylabel('joint ISI')
    plt.legend()
    plt.show()


def test_plot_results_with_errorbars_for_violations():
    K = 10
    n_montecarlo = 50
    visualization.plot_results_with_errorbars_for_violations(K, n_montecarlo)


def test_plot_results_with_errorbars_for_different_r():
    K = 10
    n_montecarlo = 50
    visualization.plot_results_with_errorbars_for_different_r(K, n_montecarlo)
def test_write_results_in_table():
    K = 10
    n_montecarlo = 50

    visualization.write_results_in_latex_table(K, n_montecarlo)

