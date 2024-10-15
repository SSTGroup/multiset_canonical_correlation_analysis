import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


def plot_results_for_paper(K, n_montecarlo, save=False):
    results_violations = np.load(Path(Path(__file__).parent.parent, f'simulation_results/K_{K}/violations_K_{K}.npy'),
                                 allow_pickle=True).item()
    results_different_R = np.load(Path(Path(__file__).parent.parent, f'simulation_results/K_{K}/different_R_K_{K}.npy'),
                                  allow_pickle=True).item()

    # store violation results for each algorithm
    scenarios_violations = ['same_eigenvalues_different_eigenvectors_rank_1',
                            'same_eigenvalues_different_eigenvectors_rank_K',
                            'different_lambda_max', 'different_lambda_min']  # dict keys would be in wrong order
    scenario_labels_violations = [r'A', r'B', r'C', r'D']
    n_scenarios_violations = len(scenario_labels_violations)
    algorithms = list(results_violations[scenarios_violations[0]].keys())
    joint_isi_per_algorithm_violations = {algorithm: np.zeros((n_scenarios_violations, n_montecarlo)) for algorithm in
                                          algorithms}
    runtime_per_algorithm_violations = {algorithm: np.zeros((n_scenarios_violations, n_montecarlo)) for algorithm in
                                        algorithms}
    for scenario_idx, scenario in enumerate(scenarios_violations):
        for algorithm_idx, algorithm in enumerate(algorithms):
            joint_isi_per_algorithm_violations[algorithm][scenario_idx, :] = results_violations[scenario][algorithm][
                'joint_isi']
            runtime_per_algorithm_violations[algorithm][scenario_idx, :] = results_violations[scenario][algorithm][
                'runtime']

    # store different R results for each algorithm
    scenarios_different_R = [f'rank_{R}' for R in [1, 2, 5, 10]]
    n_scenarios_different_R = len(scenarios_different_R)
    algorithms = list(results_different_R[scenarios_different_R[0]].keys())
    joint_isi_per_algorithm_different_R = {algorithm: np.zeros((n_scenarios_different_R, n_montecarlo)) for algorithm in
                                           algorithms}
    runtime_per_algorithm_different_R = {algorithm: np.zeros((n_scenarios_different_R, n_montecarlo)) for algorithm in
                                         algorithms}
    for scenario_idx, scenario in enumerate(scenarios_different_R):
        for algorithm_idx, algorithm in enumerate(algorithms):
            joint_isi_per_algorithm_different_R[algorithm][scenario_idx, :] = results_different_R[scenario][algorithm][
                'joint_isi']
            runtime_per_algorithm_different_R[algorithm][scenario_idx, :] = results_different_R[scenario][algorithm][
                'runtime']

    # plot JISI for violations and different R in one figure
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(9, 2.5))

    # violations
    for algorithm in algorithms:
        axes[0].errorbar(np.arange(n_scenarios_violations),
                         np.mean(joint_isi_per_algorithm_violations[algorithm], axis=1),
                         np.std(joint_isi_per_algorithm_violations[algorithm], axis=1),
                         linestyle=':', fmt='D', markersize=3, capsize=2, lw=1.1, label=f'{algorithm}')
    axes[0].set_xticks(np.arange(n_scenarios_violations), scenario_labels_violations, fontsize=12)
    axes[0].set_xlabel(r'Experiment', fontsize=12)
    axes[0].set_ylim([-0.05, 1.05])
    axes[0].set_yticks([0, 0.5, 1])
    axes[0].set_yticklabels([0, 0.5, 1], fontsize=12)
    axes[0].set_ylabel('jISI', fontsize=12)

    # different R
    for algorithm in algorithms:
        axes[1].errorbar(np.log([1, 2, 5, 10]), np.mean(joint_isi_per_algorithm_different_R[algorithm], axis=1),
                         np.std(joint_isi_per_algorithm_different_R[algorithm], axis=1),
                         linestyle=':', fmt='D', markersize=3, capsize=2, lw=1.1, label=f'{algorithm}')
    axes[1].set_xticks(np.log([1, 2, 5, 10, 100]), [1, 2, 5, 10, 100], fontsize=12)
    axes[1].set_xlabel(r'Rank $R$', fontsize=12)
    axes[1].set_ylim([-0.025, 0.525])
    axes[1].set_yticks([0, 0.25, 0.5], [0, 0.25, 0.5], fontsize=12)
    axes[1].set_ylabel('jISI', fontsize=12)

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    if save:
        plt.tight_layout()
        plt.savefig(f'joint_ISI_K_{K}.pdf')
    else:
        plt.title(f'joint ISI for the different experiments (K={K})')
        plt.tight_layout()

    # plot RUNTIME for violations and different R in one figure
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(9, 2.5))

    # violations
    for algorithm in algorithms:
        axes[0].errorbar(np.arange(n_scenarios_violations),
                         np.mean(runtime_per_algorithm_violations[algorithm], axis=1),
                         np.std(runtime_per_algorithm_violations[algorithm], axis=1),
                         linestyle=':', fmt='D', markersize=3, capsize=2, lw=1.1, label=f'{algorithm}')
    axes[0].set_xticks(np.arange(n_scenarios_violations), scenario_labels_violations, fontsize=12)
    axes[0].set_xlabel(r'Experiment', fontsize=12)
    axes[0].set_ylim([-10, 210])
    axes[0].set_yticks([0, 100, 200], [0, 100, 200], fontsize=12)
    axes[0].set_ylabel('runtime in seconds', fontsize=12)

    # different R
    for algorithm in algorithms:
        axes[1].errorbar(np.log([1, 2, 5, 10]), np.mean(runtime_per_algorithm_different_R[algorithm], axis=1),
                         np.std(runtime_per_algorithm_different_R[algorithm], axis=1),
                         linestyle=':', fmt='D', markersize=3, capsize=2, lw=1.1, label=f'{algorithm}')
    axes[1].set_xticks(np.log([1, 2, 5, 10, 100]), [1, 2, 5, 10, 100], fontsize=12)
    axes[1].set_xlabel(r'Rank $R$', fontsize=12)
    axes[1].set_ylim([-1, 21])
    axes[1].set_yticks([0, 10, 20], [0, 10, 20], fontsize=12)
    axes[1].set_ylabel('runtime in seconds', fontsize=12)

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    if save:
        plt.tight_layout()
        plt.savefig(f'runtime_K_{K}.pdf')
    else:
        plt.title(f'runtime for the different experiments (K={K})')
        plt.tight_layout()
        plt.show()
