import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from .helpers import calculate_eigenvalues_from_ccv_covariance_matrices


def plot_results_with_errorbars_for_violations(K, n_montecarlo):
    results = np.load(Path(Path(__file__).parent.parent, f'simulation_results/K_{K}/violations_K_{K}.npy'),
                      allow_pickle=True).item()

    # create pandas dataframe from results
    scenarios = list(results.keys())
    scenario_labels = ['same $\lambda_{\mathrm{max}}$', 'same $\lambda_{\mathrm{min}}$', r'same $\mathbf{\lambda}$',
                       r'$\mathbf{C}_n = \mathbf{D}_n \mathbf{C}_1 \mathbf{D}_n$']
    n_scenarios = len(scenario_labels)

    algorithms = list(results[scenarios[0]].keys())

    # store results for each algorithm
    joint_isi_per_algorithm = {algorithm: np.zeros((n_scenarios, n_montecarlo)) for algorithm in algorithms}
    runtime_per_algorithm = {algorithm: np.zeros((n_scenarios, n_montecarlo)) for algorithm in algorithms}
    for scenario_idx, scenario in enumerate(scenarios):
        for algorithm_idx, algorithm in enumerate(algorithms):
            joint_isi_per_algorithm[algorithm][scenario_idx, :] = results[scenario][algorithm]['joint_isi']
            runtime_per_algorithm[algorithm][scenario_idx, :] = results[scenario][algorithm]['runtime']

    plt.figure(figsize=(2 * n_scenarios, n_scenarios))
    plt.title(f'joint ISI for the different experiments (K={K})')

    for algorithm in algorithms:
        plt.errorbar(np.arange(n_scenarios), np.mean(joint_isi_per_algorithm[algorithm], axis=1),
                     np.std(joint_isi_per_algorithm[algorithm], axis=1),
                     linestyle=':', fmt='o', capsize=3.5, label=f'{algorithm}')
    plt.xticks(np.arange(n_scenarios), scenario_labels, rotation=90)
    plt.ylim([-0.1, 1.1])
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_results_with_errorbars_for_different_r(K, n_montecarlo):
    results = np.load(Path(Path(__file__).parent.parent, f'simulation_results/K_{K}/rank_r_K_{K}.npy'),
                      allow_pickle=True).item()

    scenarios = list(results.keys())
    scenario_labels = [int(scenario[5:]) for scenario in scenarios]
    n_scenarios = len(scenarios)

    algorithms = list(results[scenarios[0]].keys())

    joint_isi_per_algorithm = {algorithm: np.zeros((n_scenarios, n_montecarlo)) for algorithm in algorithms}
    runtime_per_algorithm = {algorithm: np.zeros((n_scenarios, n_montecarlo)) for algorithm in algorithms}
    for scenario_idx, scenario in enumerate(scenarios):
        for algorithm_idx, algorithm in enumerate(algorithms):
            joint_isi_per_algorithm[algorithm][scenario_idx, :] = results[scenario][algorithm]['joint_isi']
            runtime_per_algorithm[algorithm][scenario_idx, :] = results[scenario][algorithm]['runtime']

    plt.figure(figsize=(2 * n_scenarios, n_scenarios))
    plt.title(f'joint ISI for the different experiments (K={K})')

    for algorithm in algorithms:
        plt.errorbar(scenario_labels, np.mean(joint_isi_per_algorithm[algorithm], axis=1),
                     np.std(joint_isi_per_algorithm[algorithm], axis=1),
                     linestyle=':', fmt='o', capsize=3.5, label=f'{algorithm}')
    plt.xticks(scenario_labels, scenario_labels)
    plt.xlabel('rank')
    plt.ylim([-0.1, 1.1])
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_eigenvalues(scv_cov, show=True):
    plt.figure()
    Lambda = calculate_eigenvalues_from_ccv_covariance_matrices(scv_cov)
    Lambda = Lambda[:, ::-1]  # sort descending

    for n in range(scv_cov.shape[2]):
        plt.plot(Lambda[n, :], '*:', label=f'C_{n + 1}')
    plt.legend()
    if show:
        plt.show()
