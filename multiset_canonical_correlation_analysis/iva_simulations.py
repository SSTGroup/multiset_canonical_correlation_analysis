import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import block_diag
from pathlib import Path

from independent_vector_analysis.consistent_iva import consistent_iva
from independent_vector_analysis.helpers_iva import _bss_isi

from .mcca import mcca_ssqcor_genvar_kettenring
from .simulations import scv_covs_with_rank_R, generate_datasets_from_covariance_matrices
from .helpers import calculate_eigenvalues_from_ccv_covariance_matrices
from .visualization import plot_eigenvalues


def scv_covs_with_blocks(N, K, indices, alpha, beta):
    """
    Return SCV covariance matrices of dimension (K,K,N) that are generated as
    C = alpha v v.T + beta Q Q^T + (1-alpha-beta) I,
    where Q is of dimensions (K,K), and v is a vector with 1s on the positions defined by indices


    Parameters
    ----------
    N : int
        number of SCVs

    K : int
        number of datasets

    R : int
        low rank of the model

    Returns
    -------
    scv_cov : np.ndarray
        Array of dimensions (K, K, N) that contains the SCV covariance matrices

    """
    scv_cov = np.zeros((K, K, N))
    for n in range(N):
        if alpha[n] + beta > 1:
            raise ValueError("alpha + beta must be smaller or equal to 1")
        vector = np.zeros((K, 1))
        vector[indices, 0] = 1
        temp_variability_term = np.random.randn(K, K)
        temp_variability_term /= np.linalg.norm(temp_variability_term, axis=1, keepdims=True)
        scv_cov[:, :, n] = alpha[n] * (vector @ vector.T) + beta * (
                temp_variability_term @ temp_variability_term.T) + (1 - alpha[n] - beta) * np.eye(K)

    return scv_cov


def generate_scvs_with_subspace_structure(alpha=0.9, beta=0.1):
    N = 30
    K = 20
    scv_cov = np.zeros((K, K, N))

    # SCVs 1-14 are common
    scv_cov[:, :, 0:14] = scv_covs_with_blocks(14, K, np.arange(K), alpha=[alpha] * 14, beta=beta)

    # SCVs 15-20 have patterns
    cov = scv_covs_with_blocks(1, K, [0, 1, 2, 5, 7, 9, 10, 11, 13, 15], alpha=[alpha], beta=beta)[:, :, 0]
    np.fill_diagonal(cov, 1)
    scv_cov[:, :, 14] = cov

    cov = scv_covs_with_blocks(1, K, [0, 1, 2, 3, 4, 5, 15, 16, 17], alpha=[alpha], beta=beta)[:, :, 0]
    np.fill_diagonal(cov, 1)
    scv_cov[:, :, 15] = cov

    cov = scv_covs_with_blocks(1, K, [4, 5, 7, 8, 9, 17, 18, 19], alpha=[alpha], beta=beta)[:, :, 0]
    np.fill_diagonal(cov, 1)
    scv_cov[:, :, 16] = cov

    cov = scv_covs_with_blocks(1, K, [9, 11, 13, 15, 16, 17, 19], alpha=[alpha], beta=beta)[:, :, 0]
    np.fill_diagonal(cov, 1)
    scv_cov[:, :, 17] = cov

    cov = scv_covs_with_blocks(1, K, [0, 1, 4, 7, 11, 15], alpha=[alpha], beta=beta)[:, :, 0]
    np.fill_diagonal(cov, 1)
    scv_cov[:, :, 18] = cov

    cov = scv_covs_with_blocks(1, K, [1, 3, 4, 7, 15], alpha=[alpha], beta=beta)[:, :, 0]
    np.fill_diagonal(cov, 1)
    scv_cov[:, :, 19] = cov

    scv_cov[:, :, 20:30] = scv_covs_with_blocks(10, K, np.arange(K), alpha=[0.0] * 10, beta=beta)

    return scv_cov[:, :, 12:22]


def save_results_of_multiple_runs(N, K, T, R, alpha, beta, n_montecarlo, folder, use_true_R_xx=True, n_runs_iva=1):
    updates = ['newton', 'gradient', 'norm_gradient']
    which_ivags = ['n-o-iva_g', 'o-iva_g', 'd-o-iva_g']

    for run in range(n_montecarlo):
        print(f'Start run {run}...')

        scv_cov = scv_covs_with_rank_R(N, K, R, [alpha] * K, beta)

        X, A, S = generate_datasets_from_covariance_matrices(scv_cov, T, orthogonal_A=True)

        if use_true_R_xx:
            N, T, K = X.shape
            # true joint SCV covariance matrix
            joint_scv_cov = block_diag(*list(scv_cov.T))

            # make the permutation matrix
            P = np.zeros((N * K, N * K))
            for n in range(N):
                for k in range(K):
                    P[n + k * N, n * K + k] = 1

            # generate true C_xx from true C_ss
            C_ss = P @ joint_scv_cov @ P.T
            A_joint = block_diag(*list(A.T)).T
            R_xx_all = A_joint @ C_ss @ A_joint.T
            R_xx = np.zeros((N, N, K, K), dtype=X.dtype)
            for k in range(K):
                for l in range(k, K):
                    R_xx[:, :, k, l] = R_xx_all[k * N:(k + 1) * N, l * N:(l + 1) * N]
                    R_xx[:, :, l, k] = R_xx[:, :, k, l].T  # R_xx is symmetric
        else:
            R_xx = None

        W_init = np.load(Path(Path(__file__).parent.parent, f'simulation_results/{folder}/W_init.npy'))

        filename = Path(Path(__file__).parent.parent, f'simulation_results/{folder}/true_run{run}.npy')
        np.save(filename, {'scv_cov': scv_cov, 'joint_isi': 0})

        for which_ivag in which_ivags:
            for update in updates:
                results = consistent_iva(X, which_iva=which_ivag, W_init=W_init, n_runs=n_runs_iva, A=A,
                                         R_xx=R_xx, whiten=False, parallel=False, opt_approach=update)
                filename = Path(Path(__file__).parent.parent,
                                f'simulation_results/{folder}/{which_ivag}_{update}_run{run}.npy')
                np.save(filename, {key: results[key] for key in ('joint_isi', 'W_change', 'scv_cov')})

        # genvar
        N, T, K = X.shape
        C_xx = np.zeros((N * K, N * K))
        for k in range(K):
            for l in range(K):
                C_xx[k * N:(k + 1) * N, l * N:(l + 1) * N] = R_xx[:, :, k, l]
        M, Epsilon = mcca_ssqcor_genvar_kettenring(X, algorithm='genvar', W_init=W_init, C_xx=C_xx)
        W = np.moveaxis(M, [0, 1, 2], [1, 0, 2])
        filename = Path(Path(__file__).parent.parent,
                        f'simulation_results/{folder}/genvar_run{run}.npy')
        np.save(filename, {'joint_isi': _bss_isi(W, A)[1]})


def save_results_from_multiple_files_in_one_file(folder, n_montecarlo):
    which_ivags = ['n-o-iva_g', 'o-iva_g', 'd-o-iva_g']
    updates = ['newton', 'gradient', 'norm_gradient']

    results = {}
    for which_ivag_idx, which_ivag in enumerate(which_ivags):
        results[which_ivag] = {}
        for update_idx, update in enumerate(updates):
            joint_isi = np.zeros(n_montecarlo)
            W_change = []
            for run in range(n_montecarlo):
                filename = Path(Path(__file__).parent.parent,
                                f'simulation_results/{folder}/{which_ivag}_{update}_run{run}.npy')
                results_tmp = np.load(filename, allow_pickle=True).item()
                joint_isi[run] = results_tmp['joint_isi']
                W_change.append(results_tmp['W_change'])

            results[which_ivag][update] = {'joint_isi': joint_isi, 'W_change': W_change}

    joint_isi = np.zeros(n_montecarlo)
    for run in range(n_montecarlo):
        filename = Path(Path(__file__).parent.parent,
                        f'simulation_results/{folder}/genvar_run{run}.npy')
        results_tmp = np.load(filename, allow_pickle=True).item()
        joint_isi[run] = results_tmp['joint_isi']

    results['genvar'] = {'joint_isi': joint_isi, 'W_change': None}

    print(f'Save run as simulation_results/{folder}/results.npy.')
    np.save(Path(Path(__file__).parent.parent, f'simulation_results/{folder}/results.npy'), results)


def print_jisi_and_plot_W_change_of_multiple_runs(folder):
    which_ivags = ['n-o-iva_g', 'o-iva_g', 'd-o-iva_g']
    updates = ['newton', 'gradient', 'norm_gradient']

    filename = Path(Path(__file__).parent.parent, f'simulation_results/{folder}/results.npy')
    results = np.load(filename, allow_pickle=True).item()

    for which_ivag in which_ivags:
        for update in updates:
            res = results[which_ivag][update]
            jisi = res['joint_isi']
            print(which_ivag, update)
            print(f'{np.mean(jisi):1.1e}\pm{np.std(jisi):1.1e}')

            plt.figure()
            W_change = res['W_change']
            for run in range(len(W_change)):

                if type(W_change[run]) is dict:
                    for n in range(len(W_change[run].keys())):
                        if len(W_change[run][n]) < 1024: # print last value of W_change only if not converged
                            plt.plot(W_change[run][n])
                        else:
                            plt.plot(W_change[run][n], label=f'{W_change[run][n][-1]:1.1e}')

                else:
                    if len(W_change[run]) < 1024:
                        plt.plot(W_change[run])
                    else:
                        plt.plot(W_change[run], label=f'{W_change[run][-1]:1.1e}')
            plt.title(f'{which_ivag} with {update} update.')
            plt.legend()
    plt.show()

    jisi = results['genvar']['joint_isi']
    print('genvar')
    print(f'{np.mean(jisi):1.1e}\pm{np.std(jisi):1.1e}')


def inspect_one_run(folder, run_idx):
    filename = Path(Path(__file__).parent.parent, f'simulation_results/{folder}/true_run{run_idx}.npy')
    scv_cov = np.load(filename, allow_pickle=True).item()['scv_cov']
    plot_eigenvalues(scv_cov)

    which_ivags = ['n-o-iva_g', 'o-iva_g', 'd-o-iva_g']
    updates = ['newton', 'gradient', 'norm_gradient']
    plt.figure()
    for which_ivag_idx, which_ivag in enumerate(which_ivags):
        for update_idx, update in enumerate(updates):
            filename = Path(Path(__file__).parent.parent,
                            f'simulation_results/{folder}/{which_ivag}_{update}_run{run_idx}.npy')
            W_change = np.load(filename, allow_pickle=True).item()['W_change']
            if type(W_change) is dict:
                for n in range(len(W_change.keys())):
                    plt.plot(W_change[n], 'r', label=f'{which_ivag} {update}')
            else:
                plt.plot(W_change, label=f'{which_ivag} {update}')
    plt.legend()
    plt.show()
    pass
