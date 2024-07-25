import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

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

    plt.figure(figsize=(8, 4))
    plt.title(f'joint ISI for the different experiments (K={K})')

    for algorithm in algorithms:
        plt.errorbar(np.arange(n_scenarios), np.mean(joint_isi_per_algorithm[algorithm], axis=1),
                     np.std(joint_isi_per_algorithm[algorithm], axis=1),
                     linestyle=':', fmt='o', capsize=3.5, label=f'{algorithm}')
    plt.xticks(np.arange(n_scenarios), scenario_labels, rotation=90)
    plt.ylim([-0.1, 0.6])
    plt.legend()
    plt.tight_layout()

    plt.figure(figsize=(8, 4))
    plt.title(f'runtime in seconds for the different experiments (K={K})')

    for algorithm in algorithms:
        plt.errorbar(np.arange(n_scenarios), np.mean(runtime_per_algorithm[algorithm], axis=1),
                     np.std(runtime_per_algorithm[algorithm], axis=1),
                     linestyle=':', fmt='o', capsize=3.5, label=f'{algorithm}')
    plt.xticks(np.arange(n_scenarios), scenario_labels, rotation=90)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_results_with_errorbars_for_different_R(K, n_montecarlo):
    results = np.load(Path(Path(__file__).parent.parent, f'simulation_results/K_{K}/different_R_K_{K}.npy'),
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

    plt.figure(figsize=(8, 4))
    plt.title(f'joint ISI for the different experiments (K={K})')

    for algorithm in algorithms:
        plt.errorbar(scenario_labels, np.mean(joint_isi_per_algorithm[algorithm], axis=1),
                     np.std(joint_isi_per_algorithm[algorithm], axis=1),
                     linestyle=':', fmt='o', capsize=3.5, label=f'{algorithm}')
    plt.xticks(scenario_labels, scenario_labels)
    plt.xlabel('rank R')
    plt.ylim([-0.1, 0.6])
    plt.legend()
    plt.tight_layout()

    plt.figure(figsize=(8, 4))
    plt.title(f'runtime in seconds for the different experiments (K={K})')

    for algorithm in algorithms:
        plt.errorbar(scenario_labels, np.mean(runtime_per_algorithm[algorithm], axis=1),
                     np.std(runtime_per_algorithm[algorithm], axis=1),
                     linestyle=':', fmt='o', capsize=3.5, label=f'{algorithm}')
    plt.xticks(scenario_labels, scenario_labels)
    plt.xlabel('rank R')
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


def write_results_in_latex_table(K, n_montecarlo):
    results = np.load(Path(Path(__file__).parent.parent, f'simulation_results/K_{K}/K_{K}.npy'),
                      allow_pickle=True).item()

    # create pandas dataframe from results
    scenarios = list(results.keys())
    algorithms = list(results[scenarios[0]].keys())

    # store mean and std in array (as string?)
    joint_isi_mean_std_array = np.zeros((len(algorithms), len(scenarios)), dtype=object)
    runtime_mean_std_array = np.zeros((len(algorithms), len(scenarios)), dtype=object)
    for scenario_idx, scenario in enumerate(scenarios):
        for algorithm_idx, algorithm in enumerate(algorithms):
            joint_isi = results[scenario][algorithm]['joint_isi']
            joint_isi_mean_std_array[
                algorithm_idx, scenario_idx] = f'${np.round(np.mean(joint_isi), 2)} \pm {np.round(np.std(joint_isi), 2)}$'

            runtime = results[scenario][algorithm]['runtime']
            runtime_mean_std_array[
                algorithm_idx, scenario_idx] = f'${np.round(np.mean(runtime), 2)} \pm {np.round(np.std(runtime), 2)}$'

    table_headings = ['same $\lambda_{\mathrm{max}}$', 'same $\lambda_{\mathrm{min}}$', r'same $\vect{\lambda}$',
                      r'$\mat{C}_n = \mat{P}_n\tran \mat{C}_1 \mat{P}_n$',
                      '$\mat{C}_n = \mat{D}_n \mat{C}_1 \mat{D}_n$',
                      'rank $R=1$', 'rank $R=K$']
    joint_isi_df = pd.DataFrame(joint_isi_mean_std_array, columns=table_headings, index=algorithms)
    joint_isi_df.to_latex()
    runtime_df = pd.DataFrame(runtime_mean_std_array, columns=table_headings, index=algorithms)

    joint_isi_df.to_latex(Path(Path(__file__).parent.parent, f'simulation_results/joint_isi_K_{K}.tex'),
                          caption=r'joint ISI value (lower is better) for \underline{'
                                  f'$K={K}$'
                                  '} datasets, averaged across '
                                  f'{n_montecarlo} runs. '
                                  r'The sumcor algorithm is according to Nielsen \cite{Nielsen2002}, '
                                  r'the other algorithms are according to Kettenring \cite{Kettenring1971}.',
                          label='tab:jointisiresults',
                          position='!htb')

    runtime_df.to_latex(Path(Path(__file__).parent.parent, f'simulation_results/runtime_K_{K}.tex'),
                        caption=r'runtime in seconds (lower is better) for \underline{'
                                f'$K={K}$'
                                '} datasets, averaged across '
                                f'{n_montecarlo} runs.',
                        label='tab:runtimeresults',
                        position='!htb')
