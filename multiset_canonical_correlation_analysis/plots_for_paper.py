import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from .helpers import calculate_eigenvalues_from_ccv_covariance_matrices


def plot_results_for_paper(folder, n_montecarlo, save=False):
    results_violations = np.load(Path(Path(__file__).parent.parent, f'simulation_results/{folder}/violations.npy'),
                                 allow_pickle=True).item()
    results_different_R = np.load(Path(Path(__file__).parent.parent, f'simulation_results/{folder}/different_R.npy'),
                                  allow_pickle=True).item()

    # store violation results for each algorithm
    scenarios_violations = ['same_eigenvalues_same_eigenvectors',
                            'same_eigenvalues_different_eigenvectors',
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
    scenarios_different_R = [f'rank_{R}' for R in [1, 2, 5, 10, 20, 50]]
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
                         linestyle=(0, (1, 5)), fmt='D', markersize=3, capsize=2, lw=1.1, label=f'{algorithm}')
    axes[0].set_xticks(np.arange(n_scenarios_violations), scenario_labels_violations, fontsize=12)
    axes[0].set_xlabel(r'Experiment', fontsize=12)
    axes[0].set_ylim([-0.05, 1.05])
    axes[0].set_yticks([0, 0.5, 1])
    axes[0].set_yticklabels([0, 0.5, 1], fontsize=12)
    axes[0].set_ylabel('jISI', fontsize=12)

    # different R
    for algorithm in algorithms:
        axes[1].errorbar(np.log([1, 2, 5, 10, 20, 50]),
                         np.mean(joint_isi_per_algorithm_different_R[algorithm], axis=1),
                         np.std(joint_isi_per_algorithm_different_R[algorithm], axis=1),
                         linestyle=':', fmt='D', markersize=3, capsize=2, lw=1.1, label=f'{algorithm}')
    axes[1].set_xlim([np.log(0.9), np.log(55)])
    axes[1].set_xticks(np.log([1, 2, 5, 10, 20, 50]),
                       ['  $R$=1', '  $R$=2', '$R$=5    ', '$R$=10   ', ' $R$=20 ', '$R$=50  '],
                       fontsize=12)
    axes[1].set_xlabel(r'Experiment E', fontsize=12)
    axes[1].set_ylim([-0.03, 0.63])
    axes[1].set_yticks([0, 0.3, 0.6], [0, 0.3, 0.6], fontsize=12)

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    if save:
        plt.tight_layout()
        plt.savefig(f'joint_ISI.pdf')
    else:
        plt.title(f'joint ISI for the different experiments')
        plt.tight_layout()

    # plot RUNTIME for violations and different R in one figure
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(9, 2.5))

    # violations
    for algorithm in algorithms:
        axes[0].errorbar(np.arange(n_scenarios_violations),
                         np.mean(runtime_per_algorithm_violations[algorithm], axis=1),
                         np.std(runtime_per_algorithm_violations[algorithm], axis=1),
                         linestyle=(0, (1, 5)), fmt='D', markersize=3, capsize=2, lw=1.1, label=f'{algorithm}')
    axes[0].set_xticks(np.arange(n_scenarios_violations), scenario_labels_violations, fontsize=12)
    axes[0].set_xlabel(r'Experiment', fontsize=12)
    axes[0].set_ylim([-25, 525])
    axes[0].set_yticks([0, 250, 500], [0, 250, 500], fontsize=12)
    axes[0].set_ylabel('runtime in seconds', fontsize=12)

    # different R
    for algorithm in algorithms:
        axes[1].errorbar(np.log([1, 2, 5, 10, 20, 50]),
                         np.mean(runtime_per_algorithm_different_R[algorithm], axis=1),
                         np.std(runtime_per_algorithm_different_R[algorithm], axis=1),
                         linestyle=':', fmt='D', markersize=3, capsize=2, lw=1.1, label=f'{algorithm}')
    axes[1].set_xlim([np.log(0.95), np.log(52.5)])
    axes[1].set_xticks(np.log([1, 2, 5, 10, 20, 50]),
                       ['  $R$=1', '  $R$=2', '$R$=5    ', '$R$=10   ', ' $R$=20 ', '$R$=50  '],
                       fontsize=12)
    axes[1].set_xlabel(r'Experiment E', fontsize=12)
    axes[1].set_ylim([-.75, 15.75])
    axes[1].set_yticks([0, 7.5, 15], [0, 7.5, 15], fontsize=12)

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    if save:
        plt.tight_layout()
        plt.savefig(f'runtime.pdf')
    else:
        plt.title(f'runtime for the different experiments')
        plt.tight_layout()
        plt.show()


def plot_all_eigenvalues_for_paper(scv_cov1, scv_cov2, scv_cov3, scv_cov4, scv_cov5, filename=None):
    indices = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99]
    Lambda = []
    Lambda1 = calculate_eigenvalues_from_ccv_covariance_matrices(scv_cov1)
    Lambda.append(Lambda1[:, ::-1][:, indices])  # sort descending
    Lambda2 = calculate_eigenvalues_from_ccv_covariance_matrices(scv_cov2)
    Lambda.append(Lambda2[:, ::-1][:, indices])  # sort descending
    Lambda3 = calculate_eigenvalues_from_ccv_covariance_matrices(scv_cov3)
    Lambda.append(Lambda3[:, ::-1][:, indices])  # sort descending
    Lambda4 = calculate_eigenvalues_from_ccv_covariance_matrices(scv_cov4)
    Lambda.append(Lambda4[:, ::-1][:, indices])  # sort descending
    Lambda5 = calculate_eigenvalues_from_ccv_covariance_matrices(scv_cov5)
    Lambda.append(Lambda5[:, ::-1][:, indices])  # sort descending
    titles = ['Experiment A.' + '\n' + r'same $\mathbf{\delta}_n$, same $\mathbf{\Theta}_n$',
              'Experiment B.' + '\n' + r'same $\mathbf{\delta}_n$, different $\mathbf{\Theta}_n$',
              'Experiment C.' + '\n' + r'different $\delta_n^{(\mathrm{max})}$',
              'Experiment D.' + '\n' + r'different $\delta_n^{(\mathrm{min})}$',
              'Experiment E.' + '\n' + r'different $\mathbf{\delta}_n$, $R=50$']

    fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(11, 2.2))
    for ax_idx, ax in enumerate(axes):

        for n in range(scv_cov1.shape[2]):
            ax.plot(np.array(indices) + 1, Lambda[ax_idx][4 - n, :], 'D:', markersize=2.5, lw=1,
                    color=f'C{4 - n}', label=r'$\mathbf{\lambda}_' + f'{4 - n + 1}' + r'$')
        ax.set_xlabel(titles[ax_idx], fontsize=12)
        ax.set_xticks(np.array([0, 50, 100]), [0, 50, 100], fontsize=11)

    axes[0].set_ylim([-0.5, 5.5])
    axes[0].set_yticks(np.array([0, 2.5, 5]), [0, 2.5, 5], fontsize=11)

    axes[1].set_ylim([-0.5, 5.5])
    axes[1].set_yticks(np.array([0, 2.5, 5]), [0, 2.5, 5], fontsize=11)

    axes[2].set_ylim([-5, 55])
    axes[2].set_yticks(np.array([0, 25, 50]), [0, 25, 50], fontsize=11)

    axes[3].set_ylim([-0.1, 1.1])
    axes[3].set_yticks(np.array([0, 0.5, 1]), [0, 0.5, 1], fontsize=11)

    axes[4].set_ylim([-0.5, 5.5])
    axes[4].set_yticks(np.array([0, 2.5, 5]), [0, 2.5, 5], fontsize=11)

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.25)
    if filename is not None:
        plt.savefig(f'{filename}.pdf')
    else:
        plt.show()
