import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

from .helpers import calculate_eigenvalues_from_ccv_covariance_matrices


def plot_eigenvalues(scv_cov, title=None, show=True, filename=None):
    Lambda = calculate_eigenvalues_from_ccv_covariance_matrices(scv_cov)
    Lambda = Lambda[:, ::-1]  # sort descending

    plt.figure(figsize=(6, 3.5))
    for n in range(scv_cov.shape[2]):
        plt.plot(np.arange(1, Lambda.shape[1] + 1), Lambda[n, :], 'D:', markersize=2.5, lw=1,
                 label=r'$\mathbf{\lambda}_{' + f'{n + 1}' + r'}$')
        plt.xticks(np.arange(0, Lambda.shape[1] + 1, 5))
    plt.legend()

    plt.title(title)
    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename, dpi=500)
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
