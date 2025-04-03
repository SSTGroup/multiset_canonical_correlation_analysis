import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from independent_vector_analysis.consistent_iva import consistent_iva
from independent_vector_analysis.visualization import plot_scv_covs
from independent_vector_analysis.helpers_iva import _bss_isi

from ..simulations import scv_covs_with_rank_R
from ..mcca import mcca_ssqcor_genvar_kettenring
from .. import iva_simulations


def test_generate_scvs_with_subspace_structure():
    scv_cov = iva_simulations.generate_scvs_with_subspace_structure()
    plot_scv_covs(scv_cov, n_cols=5)


def test_generate_scvs_with_Bens_model():
    N = 10
    K = 20
    R = K
    alpha = [0.8] * K
    beta = 0.1
    scv_cov = scv_covs_with_rank_R(N, K, R, alpha, beta)
    plot_scv_covs(scv_cov, n_cols=5)


def test_save_results_one_run_checkerboard():
    T = 10000
    n_runs_iva = 1

    folder = f'T_{T}_alpha_07_beta_03'

    # scv_cov = generate_scvs_with_subspace_structure(alpha=0.7, beta=0.3)
    # plot_scv_covs(scv_cov, n_cols=5)
    #
    # X, A, S = generate_datasets_from_covariance_matrices(scv_cov, T, orthogonal_A=True)
    #
    # use_true_R_xx = True
    #
    # if use_true_R_xx:
    #     N,T,K = X.shape
    #     # true joint SCV covariance matrix
    #     joint_scv_cov = block_diag(*list(scv_cov.T))
    #
    #     # make the permutation matrix
    #     P = np.zeros((N * K, N * K))
    #     for n in range(N):
    #         for k in range(K):
    #             P[n + k * N, n * K + k] = 1
    #
    #     # generate true C_xx from true C_ss
    #     C_ss = P @ joint_scv_cov @ P.T
    #     A_joint = block_diag(*list(A.T)).T
    #     R_xx_all = A_joint @ C_ss @ A_joint.T
    #     R_xx = np.zeros((N, N, K, K), dtype=X.dtype)
    #     for k in range(K):
    #         for l in range(k, K):
    #             R_xx[:, :, k, l] = R_xx_all[k * N:(k + 1) * N, l * N:(l + 1) * N]
    #             R_xx[:, :, l, k] = R_xx[:, :, k, l].T  # R_xx is symmetric
    # else:
    #     R_xx = None
    #
    #
    # np.save('orthogonal_data.npy', {'X': X, 'A': A, 'S': S, 'R_xx': R_xx, 'scv_cov': scv_cov})

    data = np.load(Path(Path(__file__).parent.parent.parent, f'simulation_results/{folder}/orthogonal_data.npy'),
                   allow_pickle=True).item()
    X = data['X']
    A = data['A']
    S = data['S']
    scv_cov = data['scv_cov']
    R_xx = data['R_xx']

    update = 'newton'  # other options: 'gradient', 'norm_gradient' (this is not allowed in normal IVA-G)

    W_init = np.load(Path(Path(__file__).parent.parent.parent, f'simulation_results/{folder}/W_init.npy'))

    filename = Path(Path(__file__).parent.parent.parent, f'simulation_results/{folder}/true.npy')
    np.save(filename, {'scv_cov': scv_cov, 'joint_isi': 0})

    # normal IVA-G
    results = consistent_iva(X, opt_approach=update, which_iva='iva_g', W_init=W_init, n_runs=n_runs_iva, A=A,
                             R_xx=R_xx, whiten=False, parallel=False)
    s_hat_cov_ivag = results['scv_cov']
    jisi = results['joint_isi']
    filename = Path(Path(__file__).parent.parent.parent, f'simulation_results/{folder}/ivag_{update}.npy')
    np.save(filename, {'scv_cov': s_hat_cov_ivag, 'joint_isi': jisi})

    # new IVA-G (update demixing vectors after calculating all gradients and Hessians)
    results = consistent_iva(X, which_iva='n-o-iva_g', W_init=W_init, n_runs=n_runs_iva, A=A, opt_approach=update,
                             R_xx=R_xx, whiten=False, parallel=False)
    s_hat_cov_ivag = results['scv_cov']
    jisi = results['joint_isi']
    filename = Path(Path(__file__).parent.parent.parent, f'simulation_results/{folder}/new-ivag_{update}.npy')
    np.save(filename, {'scv_cov': s_hat_cov_ivag, 'joint_isi': jisi})

    # orthogonal IVA-G
    results = consistent_iva(X, which_iva='o-iva_g', W_init=W_init, n_runs=n_runs_iva, A=A, opt_approach=update,
                             R_xx=R_xx, whiten=False, parallel=False)
    s_hat_cov_ivag = results['scv_cov']
    jisi = results['joint_isi']
    filename = Path(Path(__file__).parent.parent.parent, f'simulation_results/{folder}/o-ivag_{update}.npy')
    np.save(filename, {'scv_cov': s_hat_cov_ivag, 'joint_isi': jisi})

    # deflationary orthogonal IVA-G
    results = consistent_iva(X, which_iva='d-o-iva_g', W_init=W_init, n_runs=n_runs_iva, A=A, opt_approach=update,
                             R_xx=R_xx, whiten=False, parallel=False)
    s_hat_cov_ivag = results['scv_cov']
    jisi = results['joint_isi']
    filename = Path(Path(__file__).parent.parent.parent, f'simulation_results/{folder}/d-o-ivag_{update}.npy')
    np.save(filename, {'scv_cov': s_hat_cov_ivag, 'joint_isi': jisi})

    # include also genvar for complete picture
    N, T, K = X.shape
    C_xx = np.zeros((N * K, N * K))
    for k in range(K):
        for l in range(K):
            C_xx[k * N:(k + 1) * N, l * N:(l + 1) * N] = R_xx[:, :, k, l]
    M, Epsilon = mcca_ssqcor_genvar_kettenring(X, algorithm='genvar', W_init=W_init, C_xx=C_xx)
    W = np.moveaxis(M, [0, 1, 2], [1, 0, 2])
    s_hat_cov = np.zeros((K, K, N))
    for n in range(N):
        # Efficient version of Sigma_n = 1/T * Y_n @ np.conj(Y_n.T) with Y_n = W_n @ X_n
        for k1 in range(K):
            for k2 in range(k1, K):
                s_hat_cov[k1, k2, n] = W[n, :, k1] @ R_xx[:, :, k1, k2] @ W[n, :, k2]
                s_hat_cov[k2, k1, n] = s_hat_cov[k1, k2, n]  # sigma_n is symemtric

    jisi = _bss_isi(W, A)[1]
    filename = Path(Path(__file__).parent.parent.parent, f'simulation_results/{folder}/genvar_{update}.npy')
    np.save(filename, {'scv_cov': s_hat_cov, 'joint_isi': jisi})


def test_plot_one_run_scv_covs_checkerboard():
    folder = f'T_10000_alpha_07_beta_03'

    algorithms = ['true', 'ivag', 'new-ivag', 'o-ivag', 'd-o-ivag', 'genvar']
    update = 'newton'
    n_cols = 10
    n_rows = len(algorithms)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2 * n_cols, 2 * n_rows))

    # Add data to image grid and plot
    for algorithm_idx, algorithm in enumerate(algorithms):
        filename = Path(Path(__file__).parent.parent.parent, f'simulation_results/{folder}/{algorithm}_{update}.npy')
        res = np.load(filename, allow_pickle=True).item()
        scv_cov = res['scv_cov']

        for scv_idx in range(scv_cov.shape[2]):
            axes[algorithm_idx, scv_idx].imshow(np.abs(scv_cov[:, :, scv_idx]), vmin=0, vmax=1, cmap='hot')
            axes[algorithm_idx, scv_idx].set_title(f'SCV {scv_idx + 1}', fontsize=8, pad=4)
            axes[algorithm_idx, scv_idx].set_xticks([])
            axes[algorithm_idx, scv_idx].set_yticks([])
        axes[algorithm_idx, 0].set_ylabel(f'{algorithm} \n jISI:{res['joint_isi']:.1e}')

    plt.show()


def test_compare_multiple_iva_options():
    N = 10  # SCVs
    K = 20  # datasets
    T = 10000  # samples

    n_montecarlo = 20  # runs

    R = 1
    alpha = 0.9
    beta = 0.01
    folder = f'T_10000_Bensmodel_alpha_09_beta_001'

    iva_simulations.save_results_of_multiple_runs(N, K, T, R, alpha, beta, n_montecarlo, folder, use_true_R_xx=True, n_runs_iva=20)


def test_save_and_print_results_from_multiple_files_in_one_file():
    n_montecarlo = 20
    folder = f'T_10000_Bensmodel_alpha_09_beta_001_new'

    iva_simulations.save_results_from_multiple_files_in_one_file(folder, n_montecarlo)
    iva_simulations.print_jisi_and_plot_W_change_of_multiple_runs(folder)


def test_inspect_one_run():
    folder = f'T_10000_Bensmodel_alpha_09_beta_001'
    iva_simulations.inspect_one_run(folder, run_idx=0)


