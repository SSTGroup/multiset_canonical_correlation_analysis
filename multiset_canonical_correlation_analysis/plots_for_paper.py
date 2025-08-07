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


def plot_true_estimated_results_for_paper(folder1, folder2, n_montecarlo, save=False):
    results_violations1 = np.load(Path(Path(__file__).parent.parent, f'simulation_results/{folder1}/violations.npy'),
                                  allow_pickle=True).item()
    results_violations2 = np.load(Path(Path(__file__).parent.parent, f'simulation_results/{folder2}/violations.npy'),
                                  allow_pickle=True).item()

    # store violation results for each algorithm
    scenarios_violations = ['same_eigenvalues_same_eigenvectors',
                            'same_eigenvalues_different_eigenvectors',
                            'different_lambda_min',
                            'different_lambda_max']  # dict keys would be in wrong order
    scenario_labels_violations = [r'A', r'B', r'C', r'D']
    n_scenarios_violations = len(scenario_labels_violations)
    algorithms = list(results_violations1[scenarios_violations[0]].keys())
    joint_isi_per_algorithm_violations1 = {algorithm: np.zeros((n_scenarios_violations, n_montecarlo)) for algorithm in
                                           algorithms}
    joint_isi_per_algorithm_violations2 = {algorithm: np.zeros((n_scenarios_violations, n_montecarlo)) for algorithm in
                                           algorithms}
    # runtime_per_algorithm_violations = {algorithm: np.zeros((n_scenarios_violations, n_montecarlo)) for algorithm in
    #                                     algorithms}
    for scenario_idx, scenario in enumerate(scenarios_violations):
        for algorithm_idx, algorithm in enumerate(algorithms):
            joint_isi_per_algorithm_violations1[algorithm][scenario_idx, :] = results_violations1[scenario][algorithm][
                'joint_isi']
            joint_isi_per_algorithm_violations2[algorithm][scenario_idx, :] = results_violations2[scenario][algorithm][
                'joint_isi']
            # runtime_per_algorithm_violations[algorithm][scenario_idx, :] = results_violations[scenario][algorithm][
            #     'runtime']

    # plot JISI for violations using true and estimated covariance matrices in one figure
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(9, 2.5))

    # true
    for algorithm in algorithms:
        axes[0].errorbar(np.arange(n_scenarios_violations),
                         np.mean(joint_isi_per_algorithm_violations1[algorithm], axis=1),
                         np.std(joint_isi_per_algorithm_violations1[algorithm], axis=1),
                         linestyle=(0, (1, 5)), fmt='D', markersize=3, capsize=2, lw=1.1, label=f'{algorithm}')
    axes[0].set_xticks(np.arange(n_scenarios_violations), scenario_labels_violations, fontsize=12)
    axes[0].set_xlabel(r'(a) Infinite samples', fontsize=12)
    axes[0].set_ylim([-0.05, 1.05])
    axes[0].set_yticks([0, 0.5, 1])
    axes[0].set_yticklabels([0, 0.5, 1], fontsize=12)
    axes[0].set_ylabel('jISI', fontsize=12)

    # estimated
    for algorithm in algorithms:
        axes[1].errorbar(np.arange(n_scenarios_violations),
                         np.mean(joint_isi_per_algorithm_violations2[algorithm], axis=1),
                         np.std(joint_isi_per_algorithm_violations2[algorithm], axis=1),
                         linestyle=(0, (1, 5)), fmt='D', markersize=3, capsize=2, lw=1.1, label=f'{algorithm}')
    axes[1].set_xticks(np.arange(n_scenarios_violations), scenario_labels_violations, fontsize=12)
    axes[1].set_xlabel(r'(b) $T=10000$ samples', fontsize=12)
    axes[1].set_ylim([-0.05, 1.05])
    axes[1].set_yticks([0, 0.5, 1])
    axes[1].set_yticklabels([0, 0.5, 1], fontsize=12)

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    if save:
        plt.tight_layout()
        plt.subplots_adjust(wspace=0.2)
        plt.savefig(f'joint_ISI.pdf')
    else:
        plt.title(f'joint ISI for the different experiments')
        plt.tight_layout()
        plt.subplots_adjust(wspace=0.2)
        plt.show()


def plot_true_estimated_rank_R_results_for_paper(folder1, folder2, n_montecarlo, save=False):
    results_differentR1 = np.load(Path(Path(__file__).parent.parent, f'simulation_results/{folder1}/different_R.npy'),
                                  allow_pickle=True).item()
    results_differentR2 = np.load(Path(Path(__file__).parent.parent, f'simulation_results/{folder2}/different_R.npy'),
                                  allow_pickle=True).item()

    # store R value results for each algorithm
    R_values = [1, 2, 5, 10, 20, 50]
    scenarios_differentR = [f'rank_{R}' for R in R_values]

    scenario_labels_differentR = [f'{R}$' for R in R_values]
    n_scenarios_differentR = len(scenario_labels_differentR)
    algorithms = list(results_differentR1[scenarios_differentR[0]].keys())
    joint_isi_per_algorithm_differentR1 = {algorithm: np.zeros((n_scenarios_differentR, n_montecarlo)) for algorithm in
                                           algorithms}
    joint_isi_per_algorithm_differentR2 = {algorithm: np.zeros((n_scenarios_differentR, n_montecarlo)) for algorithm in
                                           algorithms}

    for scenario_idx, scenario in enumerate(scenarios_differentR):
        for algorithm_idx, algorithm in enumerate(algorithms):
            joint_isi_per_algorithm_differentR1[algorithm][scenario_idx, :] = results_differentR1[scenario][algorithm][
                'joint_isi']
            joint_isi_per_algorithm_differentR2[algorithm][scenario_idx, :] = results_differentR2[scenario][algorithm][
                'joint_isi']

    # plot JISI for differentR using true and estimated covariance matrices in one figure
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(9, 2.5))

    colors = ['C0', 'C3', 'C2', 'C1', 'C4']
    # true
    axes[0].axhline(y=0.05, color='tab:gray', linestyle=':', linewidth=1.1)
    for (idx, algorithm) in enumerate(algorithms):
        axes[0].errorbar(R_values,
                         np.mean(joint_isi_per_algorithm_differentR1[algorithm], axis=1),
                         np.std(joint_isi_per_algorithm_differentR1[algorithm], axis=1),
                         color=colors[idx], linestyle=':', fmt='D', markersize=3, capsize=2, lw=1.1,
                         label=f'{algorithm}')
    axes[0].set_xscale('log')
    axes[0].set_xticks(R_values, scenario_labels_differentR, fontsize=12)
    axes[0].set_xlabel(r'effective rank $R$ \\ \\ (a) True covariance matrices', fontsize=12)
    axes[0].set_ylim([-0.05, 1.05])
    axes[0].set_yticks([0, 0.5, 1])
    axes[0].set_yticklabels([0, 0.5, 1], fontsize=12)
    axes[0].set_ylabel('jISI', fontsize=12)

    # estimated
    axes[1].axhline(y=0.05, color='tab:gray', linestyle=':', linewidth=1.1)
    for (idx, algorithm) in enumerate(algorithms):
        axes[1].errorbar(R_values,
                         np.mean(joint_isi_per_algorithm_differentR2[algorithm], axis=1),
                         np.std(joint_isi_per_algorithm_differentR2[algorithm], axis=1),
                         color=colors[idx], linestyle=':', fmt='D', markersize=3, capsize=2, lw=1.1,
                         label=f'{algorithm}')
    axes[1].set_xscale('log')
    axes[1].set_xticks(R_values, scenario_labels_differentR, fontsize=12)
    axes[1].set_xlabel(r'effective rank $R$ \\ \\ (b) Estimated covariance matrices', fontsize=12)
    axes[1].set_ylim([-0.05, 1.05])
    axes[1].set_yticks([0, 0.5, 1])
    axes[1].set_yticklabels([0, 0.5, 1], fontsize=12)

    plt.legend(loc='center left', bbox_to_anchor=(1.05, 0.5))

    if save:
        plt.tight_layout()
        plt.subplots_adjust(wspace=0.2)
        matplot2tikz.save('joint_ISI_R.tex', encoding='utf8', axis_width='7.5cm', axis_height='5cm', standalone=True)
        plt.savefig(f'joint_ISI_R.pdf')
    else:
        plt.title(f'joint ISI for the different experiments')
        plt.tight_layout()
        plt.subplots_adjust(wspace=0.2)
        plt.show()

def plot_all_eigenvalues_for_paper(scv_cov1, scv_cov2, scv_cov3, scv_cov4, filename=None):
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
    titles = ['Experiment A.' + '\n' + r'same $\mathbf{\lambda}_n$, same $\mathbf{\Theta}_n$',
              'Experiment B.' + '\n' + r'same $\mathbf{\lambda}_n$, different $\mathbf{\Theta}_n$',
              'Experiment C.' + '\n' + r'different $\lambda_n^{(\mathrm{min})}$',
              'Experiment D.' + '\n' + r'different $\lambda_n^{(\mathrm{max})}$']

    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 2.2))
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

    axes[2].set_ylim([-0.1, 1.1])
    axes[2].set_yticks(np.array([0, 0.5, 1]), [0, 0.5, 1], fontsize=11)

    axes[3].set_ylim([-5, 55])
    axes[3].set_yticks(np.array([0, 25, 50]), [0, 25, 50], fontsize=11)

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.25)
    if filename is not None:
        plt.savefig(f'{filename}.pdf')
    else:
        plt.show()
