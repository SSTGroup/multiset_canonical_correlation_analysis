import numpy as np
from scipy.stats import random_correlation
from pathlib import Path

from independent_vector_analysis.helpers_iva import _bss_isi
from independent_vector_analysis.data_generation import MGGD_generation

from .helpers import calculate_eigenvalues_from_ccv_covariance_matrices
from .mcca import mcca

import time


def generate_datasets_from_covariance_matrices(scv_cov, T):
    # generate sources
    K, _, N = scv_cov.shape
    S = np.zeros((N, T, K))
    for n in range(N):
        S_temp = MGGD_generation(T, cov=scv_cov[:, :, n])[0]
        # make sources zero-mean and unit-variance
        S_temp -= np.mean(S_temp, axis=1, keepdims=True)
        S_temp /= np.std(S_temp, axis=1, keepdims=True)
        S[n, :, :] = S_temp.T
    # create mixing matrices
    A = np.random.randn(N, N, K)

    X = np.zeros((N, T, K))
    for k in range(K):
        X[:, :, k] = A[:, :, k] @ S[:, :, k]

    return X, A, S


def scv_covs_with_same_eigenvalues_same_eigenvectors_rank_K(N, K, alpha, beta):
    """
    Return SCV covariance matrices of dimensions (K,K,N) that have the same eigenvalues but different eigenvectors (to
    test if genvar can still identify the sources thanks to the eigenvectors).


    Parameters
    ----------
    N : int
        number of SCVs

    K : int
        number of datasets

    Returns
    -------
    scv_cov : np.ndarray
        Array of dimensions (K, K, N) that contains the SCV covariance matrices

    """

    # create N random SCV covariance matrices
    scv_cov = np.zeros((K, K, N))

    # same eigenvalue profile for all SCVs (generated as for rank K)
    scv_cov[:, :, 0] = scv_covs_with_rank_R(N, K, K, alpha, beta)[:, :, 0]
    Lambda, eigvec = np.linalg.eigh(scv_cov[:, :, 0])

    for n in range(1, N):
        # make matrix with diagonal elements +-1
        diag_elements = np.random.uniform(-1, 1, K)
        diag_elements = np.sign(diag_elements)
        D = np.diag(diag_elements)
        U = D @ eigvec
        scv_cov[:, :, n] = U @ np.diag(Lambda) @ U.T

    return scv_cov


def scv_covs_with_same_eigenvalues_different_eigenvectors_rank_K(N, K, alpha, beta):
    """
    Return SCV covariance matrices of dimensions (K,K,N) that have the same eigenvalues but different eigenvectors (to
    test if genvar can still identify the sources thanks to the eigenvectors).


    Parameters
    ----------
    N : int
        number of SCVs

    K : int
        number of datasets

    Returns
    -------
    scv_cov : np.ndarray
        Array of dimensions (K, K, N) that contains the SCV covariance matrices

    """

    # same eigenvalue profile for all SCVs (generated as for rank K)
    Lambda = calculate_eigenvalues_from_ccv_covariance_matrices(scv_covs_with_rank_R(N, K, K, alpha, beta))[0, :]

    # create N random SCV covariance matrices
    scv_cov = np.zeros((K, K, N))
    for n in range(N):
        scv_cov[:, :, n] = random_correlation.rvs(Lambda)

    return scv_cov


def scv_covs_with_rank_R(N, K, R, alpha, beta):
    """
    Return SCV covariance matrices of dimension (K,K,N) that are generated as
    C = alpha Q Q^T + beta G G^T + (1-alpha-beta) I, where Q is of dimensions (K,R) and G is of dimensions (K,K)


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
    while True:  # make sure that second-largest EVs of all SCVs are smaller than largest EVs of all SCVs
        scv_cov = np.zeros((K, K, N))
        for n in range(N):
            if alpha[n] + beta > 1:
                raise ValueError("alpha + beta must be smaller or equal to 1")
            temp_rank_term = np.random.randn(K, R)
            temp_rank_term /= np.linalg.norm(temp_rank_term, axis=1, keepdims=True)
            temp_variability_term = np.random.randn(K, K)
            temp_variability_term /= np.linalg.norm(temp_variability_term, axis=1, keepdims=True)
            scv_cov[:, :, n] = alpha[n] * (temp_rank_term @ temp_rank_term.T) + beta * (
                    temp_variability_term @ temp_variability_term.T) + (1 - alpha[n] - beta) * np.eye(K)

        # we assume that for R=1, if alpha is chosen properly, this does not need to be checked
        # (as for violating sumcor, the code would not run if this was tested)
        if R > 1:
            Lambda = calculate_eigenvalues_from_ccv_covariance_matrices(scv_cov)
            Lambda = Lambda[:, ::-1]  # sort descending

            # largest EVs of all SCVs should be bigger than second largest SCVs + some margin, otherwise recreate
            if np.min(Lambda[:, 0]) > np.max(Lambda[:, 1]) + 1 / K:
                break
        else:
            break

    return scv_cov


def scv_covs_for_maxvar_minvar(N, K, alpha):
    Lambda = np.zeros((N, K))
    for n in range(N):
        Lambda[n, -1] = alpha[n]
        Lambda[n, :-1] = (K - alpha[n]) / (K - 1)
    # create N random SCV covariance matrices
    scv_cov = np.zeros((K, K, N))
    for n in range(N):
        scv_cov[:, :, n] = random_correlation.rvs(Lambda[n, :])

    # # this is to check if eigenvalues of minvar are the same
    # A = np.linalg.eigh(scv_cov[:, :, 2])[1]
    # B = np.linalg.eigh(scv_cov[:, :, 1])[1]
    # sign_vector = np.isclose(A[:, 0], B[:, 0])
    # sign_vector = sign_vector * 2 - 1
    # calculate_matrix_ranks(np.diag(sign_vector) @ A[:, 1:], B[:, 1:])
    #
    # # this is to check if eigenvalues of maxvar are the same
    # A = np.linalg.eigh(scv_cov[:, :, 2])[1]
    # B = np.linalg.eigh(scv_cov[:, :, 1])[1]
    # sign_vector = np.isclose(A[:, -1], B[:, -1])
    # sign_vector = sign_vector * 2 - 1
    # calculate_matrix_ranks(np.diag(sign_vector) @ A[:, :-1], B[:, :-1])

    return scv_cov


def save_joint_isi_and_runtime_results(N, K, T, n_montecarlo, scenarios, use_true_C_xx):
    folder = f'K_{K}_T_{T}'
    if use_true_C_xx:
        folder += '_true_C'

    algorithms = ['sumcor', 'maxvar', 'minvar', 'ssqcor', 'genvar']

    for run in range(n_montecarlo):
        print(f'Start run {run}...')

        for scenario_idx, scenario in enumerate(scenarios):
            print(f'Simulations for {scenario}')

            if scenario == 'same_eigenvalues_same_eigenvectors':
                scv_cov = scv_covs_with_same_eigenvalues_same_eigenvectors_rank_K(N, K,
                                                                                  alpha=[1, 1, 1, 1, 1],
                                                                                  beta=0.0)
            elif scenario == 'same_eigenvalues_different_eigenvectors':
                scv_cov = scv_covs_with_same_eigenvalues_different_eigenvectors_rank_K(N, K,
                                                                                       alpha=[1, 1, 1, 1, 1],
                                                                                       beta=0.0)
            elif scenario == 'different_lambda_min':
                alpha = 1 - (K - np.array([0.1, 0.15, 0.2, 0.25, 0.3])) / (K - 1)
                scv_cov = scv_covs_with_rank_R(N, K, 1, alpha=alpha, beta=0.0)
            elif scenario == 'different_lambda_max':
                alpha = 1 - (K - np.array([10, 15, 20, 25, 30])) / (K - 1)
                scv_cov = scv_covs_with_rank_R(N, K, 1, alpha=alpha, beta=0.0)
            elif scenario[0:5] == 'rank_':
                scv_cov = scv_covs_with_rank_R(N, K, int(scenario[5:]), alpha=[0.9, 0.85, 0.8, 0.75, 0.7], beta=0.0)
            else:
                raise AssertionError(f"scenario '{scenario}' does not exist")

            X, A, S = generate_datasets_from_covariance_matrices(scv_cov, T)

            if use_true_C_xx:
                # true joint SCV covariance matrix
                joint_scv_cov = block_diag(*list(scv_cov.T))

                # make the permutation matrix
                P = np.zeros((N * K, N * K))
                for n in range(N):
                    for k in range(K):
                        P[n + k * N, n * K + k] = 1

                # generate C_xx from true C_ss
                C_ss = P @ joint_scv_cov @ P.T
                A_joint = block_diag(*list(A.T)).T
                C_xx = A_joint @ C_ss @ A_joint.T
            else:
                C_xx = None

            for algorithm_idx, algorithm in enumerate(algorithms):
                # if algorithm is not 'ivag':
                t_start = time.process_time()
                M = mcca(X, algorithm, C_xx=C_xx)[0]
                W = np.moveaxis(M, [0, 1, 2], [1, 0, 2])
                t_end = time.process_time()

                filename = Path(Path(__file__).parent.parent,
                                f'simulation_results/{folder}/{scenario}_{algorithm}_run{run}.npy')
                np.save(filename, {'joint_isi': _bss_isi(W, A)[1], 'runtime': t_end - t_start})


def save_violation_results_from_multiple_files_in_one_file(folder, n_montecarlo):
    scenarios = ['same_eigenvalues_same_eigenvectors',
                 'same_eigenvalues_different_eigenvectors',
                 'different_lambda_max', 'different_lambda_min']

    algorithms = ['sumcor', 'maxvar', 'minvar', 'ssqcor', 'genvar']

    results = {}
    for scenario_idx, scenario in enumerate(scenarios):
        results[scenario] = {}
        for algorithm_idx, algorithm in enumerate(algorithms):
            joint_isi = np.zeros(n_montecarlo)
            runtime = np.zeros(n_montecarlo)
            for run in range(n_montecarlo):
                filename = Path(Path(__file__).parent.parent,
                                f'simulation_results/{folder}/{scenario}_{algorithm}_run{run}.npy')
                results_tmp = np.load(filename, allow_pickle=True).item()
                runtime[run] = results_tmp['runtime']
                joint_isi[run] = results_tmp['joint_isi']

            results[scenario][algorithm] = {'joint_isi': joint_isi, 'runtime': runtime}

    print(f'Save run as simulation_results/{folder}/violations.npy.')
    np.save(Path(Path(__file__).parent.parent, f'simulation_results/{folder}/violations.npy'), results)


def save_different_R_results_from_multiple_files_in_one_file(folder, n_montecarlo):
    scenarios = [f'rank_{R}' for R in [1, 2, 5, 10, 20, 50]]

    algorithms = ['sumcor', 'maxvar', 'minvar', 'ssqcor', 'genvar']

    results = {}
    for scenario_idx, scenario in enumerate(scenarios):
        results[scenario] = {}
        for algorithm_idx, algorithm in enumerate(algorithms):
            joint_isi = np.zeros(n_montecarlo)
            runtime = np.zeros(n_montecarlo)
            for run in range(n_montecarlo):
                filename = Path(Path(__file__).parent.parent,
                                f'simulation_results/{folder}/{scenario}_{algorithm}_run{run}.npy')
                results_tmp = np.load(filename, allow_pickle=True).item()
                runtime[run] = results_tmp['runtime']
                joint_isi[run] = results_tmp['joint_isi']

            results[scenario][algorithm] = {'joint_isi': joint_isi, 'runtime': runtime}

    print(f'Save run as simulation_results/{folder}/different_R.npy.')
    np.save(Path(Path(__file__).parent.parent, f'simulation_results/{folder}/different_R.npy'), results)
