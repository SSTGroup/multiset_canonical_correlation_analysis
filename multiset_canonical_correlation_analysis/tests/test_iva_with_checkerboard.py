import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from scipy.linalg import block_diag
import scipy as sc

from independent_vector_analysis.consistent_iva import consistent_iva
from independent_vector_analysis.visualization import plot_scv_covs
from independent_vector_analysis.helpers_iva import _bss_isi

from ..simulations import scv_covs_with_rank_R, generate_datasets_from_covariance_matrices
from ..mcca import mcca_ssqcor_genvar_kettenring
from .. import iva_simulations
from ..visualization import plot_eigenvalues


def test_generate_scvs_with_subspace_structure():
    scv_cov = iva_simulations.generate_scvs_with_subspace_structure_new(0.3)


def test_save_results_one_run_checkerboard():
    T = 10000
    n_runs_iva = 20

    np.random.seed(0)

    # scv_cov = iva_simulations.generate_5random_correlation_scvs_with_subspace_structure(rho_d=0.3)
    # # plot_scv_covs(scv_cov, n_cols=5)
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


    # create 20 random W_init
    # np.random.seed(0)
    W_init = []
    for run in range(n_runs_iva):
        # randomly initialize
        W = np.random.randn(5, 5, 20)
        for k in range(20):
            W[:, :, k] = np.linalg.solve(sc.linalg.sqrtm(W[:, :, k] @ W[:, :, k].T), W[:, :, k])
        W_init.append(W)

    rho_distinct = [3]
    for rho in rho_distinct:
        folder = f'true_R_xx_5random_correlation_SCVs_offblock_0{rho}_20_runs'

        data = np.load(Path(Path(__file__).parent.parent.parent, f'simulation_results/{folder}/orthogonal_data.npy'),
                       allow_pickle=True).item()
        X = data['X']
        A = data['A']
        S = data['S']
        scv_cov = data['scv_cov']
        R_xx = data['R_xx']


        update = 'newton'  # other options: 'gradient', 'norm_gradient' (this is not allowed in normal IVA-G)

        # W_init = None # np.load(Path(Path(__file__).parent.parent.parent, f'simulation_results/{folder}/W_init.npy'))

        # W_init = np.zeros_like(A)
        # for k in range(A.shape[2]):
        #     W_init[:,:,k] = np.linalg.inv(A[:,:,k])
        # # make it a list so it works with following code
        # W_init = [W_init]

        parallel = False
        for update in ['newton', 'gradient']:
            filename = Path(Path(__file__).parent.parent.parent, f'simulation_results/{folder}/true_{update}.npy')
            np.save(filename, {'scv_cov': scv_cov, 'joint_isi': 0})

            # normal IVA-G
            results = consistent_iva(X, opt_approach=update, which_iva='iva_g', W_init=W_init, n_runs=n_runs_iva, A=A,
                                     R_xx=R_xx, whiten=False, parallel=parallel)
            s_hat_cov_ivag = results['scv_cov']
            jisi = results['joint_isi']
            W_change = results['W_change']
            cross_isi = results['cross_isi']
            filename = Path(Path(__file__).parent.parent.parent, f'simulation_results/{folder}/ivag_{update}.npy')
            np.save(filename, {'scv_cov': s_hat_cov_ivag, 'joint_isi': jisi, 'W_change': W_change, 'cross_isi': cross_isi})

            # new IVA-G (update demixing vectors after calculating all gradients and Hessians)
            results = consistent_iva(X, which_iva='n-o-iva_g', W_init=W_init, n_runs=n_runs_iva, A=A, opt_approach=update,
                                     R_xx=R_xx, whiten=False, parallel=parallel)
            s_hat_cov_ivag = results['scv_cov']
            jisi = results['joint_isi']
            W_change = results['W_change']
            cross_isi = results['cross_isi']
            filename = Path(Path(__file__).parent.parent.parent, f'simulation_results/{folder}/n-o-ivag_{update}.npy')
            np.save(filename, {'scv_cov': s_hat_cov_ivag, 'joint_isi': jisi, 'W_change': W_change, 'cross_isi': cross_isi})

            # orthogonal IVA-G
            results = consistent_iva(X, which_iva='o-iva_g', W_init=W_init, n_runs=n_runs_iva, A=A, opt_approach=update,
                                     R_xx=R_xx, whiten=False, parallel=parallel)
            s_hat_cov_ivag = results['scv_cov']
            jisi = results['joint_isi']
            W_change = results['W_change']
            cross_isi = results['cross_isi']
            filename = Path(Path(__file__).parent.parent.parent, f'simulation_results/{folder}/o-ivag_{update}.npy')
            np.save(filename, {'scv_cov': s_hat_cov_ivag, 'joint_isi': jisi, 'W_change': W_change, 'cross_isi': cross_isi})

            # deflationary orthogonal IVA-G
            results = consistent_iva(X, which_iva='d-o-iva_g', W_init=W_init, n_runs=n_runs_iva, A=A, opt_approach=update,
                                     R_xx=R_xx, whiten=False, parallel=parallel)
            s_hat_cov_ivag = results['scv_cov']
            jisi = results['joint_isi']
            W_change = results['W_change']
            cross_isi = results['cross_isi']
            filename = Path(Path(__file__).parent.parent.parent, f'simulation_results/{folder}/d-o-ivag_{update}.npy')
            np.save(filename, {'scv_cov': s_hat_cov_ivag, 'joint_isi': jisi, 'W_change': W_change, 'cross_isi': cross_isi})

            # include also genvar for complete picture
            N, T, K = X.shape
            C_xx = np.zeros((N * K, N * K))
            for k in range(K):
                for l in range(K):
                    C_xx[k * N:(k + 1) * N, l * N:(l + 1) * N] = R_xx[:, :, k, l]
            M, Epsilon = mcca_ssqcor_genvar_kettenring(X, algorithm='genvar', W_init=None, C_xx=C_xx)
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
    rho_distinct = [3]
    for rho in rho_distinct:
        folder = f'true_R_xx_5random_correlation_SCVs_offblock_0{rho}_20_runs'

        algorithms = ['true', 'ivag', 'n-o-ivag', 'o-ivag', 'd-o-ivag', 'genvar']
        for update in ['newton', 'gradient']:
            n_cols = 5
            n_rows = len(algorithms)
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(2 * n_cols, 2 * n_rows))

            # Add data to image grid and plot
            for algorithm_idx, algorithm in enumerate(algorithms):
                filename = Path(Path(__file__).parent.parent.parent, f'simulation_results/{folder}/{algorithm}_{update}.npy')
                res = np.load(filename, allow_pickle=True).item()
                scv_cov = res['scv_cov']

                for scv_idx in range(scv_cov.shape[2]):
                    axes[algorithm_idx, scv_idx].imshow(np.abs(scv_cov[:, :, scv_idx]), vmin=0, vmax=1, cmap='hot')
                    axes[algorithm_idx, scv_idx].set_title(f'SCV {scv_idx + 1}: logdet = {np.linalg.slogdet(scv_cov[:,:,scv_idx])[1]:2.1f}', fontsize=8, pad=4)
                    axes[algorithm_idx, scv_idx].set_xticks([])
                    axes[algorithm_idx, scv_idx].set_yticks([])
                axes[algorithm_idx, 0].set_ylabel(f'{algorithm} \n jISI:{res['joint_isi']:.1e} \n $\sum$logdet = {np.sum(np.linalg.slogdet(scv_cov.T)[1]):2.1f}')

            plt.suptitle(folder+update)
            plt.show()


def test_inspect_checkerboard_run():
    folder = f'true_R_xx_5alpha_SCVs_offblock_03_true_W_init'
    iva_simulations.inspect_checkerboard_run(folder)
