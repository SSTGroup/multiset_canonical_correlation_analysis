import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from independent_vector_analysis.consistent_iva import consistent_iva
from independent_vector_analysis.visualization import plot_scv_covs

from ..simulations import generate_datasets_from_covariance_matrices
from ..iva_simulations import generate_scvs_with_subspace_structure


def test_generate_scvs_with_subspace_structure():
    scv_cov = generate_scvs_with_subspace_structure()
    plot_scv_covs(scv_cov, n_cols=5)


def test_save_results():
    T = 10_000
    n_runs_iva = 10
    folder = f'T_{T}_rho_09'

    scv_cov = generate_scvs_with_subspace_structure()
    plot_scv_covs(scv_cov, n_cols=5)

    filename = Path(Path(__file__).parent.parent.parent, f'simulation_results/{folder}/true.npy')
    np.save(filename, scv_cov)

    X, A, S = generate_datasets_from_covariance_matrices(scv_cov, T)

    s_hat_cov_ivag = consistent_iva(X, which_iva='iva_g', n_runs=n_runs_iva)['scv_cov']
    filename = Path(Path(__file__).parent.parent.parent, f'simulation_results/{folder}/ivag.npy')
    np.save(filename, s_hat_cov_ivag)

    s_hat_cov_oivag = consistent_iva(X, which_iva='o-iva_g', n_runs=n_runs_iva)['scv_cov']
    filename = Path(Path(__file__).parent.parent.parent, f'simulation_results/{folder}/o-ivag.npy')
    np.save(filename, s_hat_cov_oivag)

    s_hat_cov_doivag = consistent_iva(X, which_iva='d-o-iva_g', n_runs=n_runs_iva)['scv_cov']
    filename = Path(Path(__file__).parent.parent.parent, f'simulation_results/{folder}/d-o-ivag.npy')
    np.save(filename, s_hat_cov_doivag)


def test_generate_figure():
    T = 10_000
    folder = f'T_{T}_rho_09'

    n_cols = 10
    n_rows = 4
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2 * n_cols, 2 * n_rows))

    # Add data to image grid and plot
    for algorithm_idx, algorithm in enumerate(['true', 'ivag', 'o-ivag', 'd-o-ivag']):
        filename = Path(Path(__file__).parent.parent.parent, f'simulation_results/{folder}/{algorithm}.npy')
        scv_cov = np.load(filename, allow_pickle=True)

        for scv_idx in range(scv_cov.shape[2]):
            axes[algorithm_idx, scv_idx].imshow(np.abs(scv_cov[:, :, scv_idx]), vmin=0, vmax=1, cmap='hot')
            axes[algorithm_idx, scv_idx].set_title(f'SCV {scv_idx + 1}', fontsize=8, pad=4)
            axes[algorithm_idx, scv_idx].set_xticks([])
            axes[algorithm_idx, scv_idx].set_yticks([])
            axes[algorithm_idx, 0].set_ylabel(algorithm)

        axes[algorithm_idx, 0].set_ylabel(algorithm)
    plt.show()
