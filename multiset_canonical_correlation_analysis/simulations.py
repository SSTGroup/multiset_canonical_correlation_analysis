import matplotlib.pyplot as plt
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


def scv_covs_with_same_maximum_eigenvalue(N, K):
    """
    Return SCV covariance matrices of dimensions (K,K,N) that have the same maximum eigenvalue (to violate the maxvar
    source identification condition).


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

    Lambda = np.zeros((N, K))
    for n in range(N):
        lambda_K_1 = -1
        while lambda_K_1 < 0.2:  # this EV should be bigger than 0.2 to not violate minvar
            Lambda[n, 0] = (K - 1) / 2
            Lambda[n, 1:K - 2] = np.random.uniform(0.2, 1, K - 3)
            Lambda[n, K - 2] = np.random.uniform(0.05, 0.15)  # only this EV should be smaller than 0.2
            lambda_K_1 = K - np.sum(Lambda[n, :])
        Lambda[n, K - 1] = lambda_K_1

    # create covariance matrices with these eigenvalue profiles
    scv_cov = np.zeros((K, K, N))
    for n in range(N):
        scv_cov[:, :, n] = random_correlation.rvs(Lambda[n, :])

    return scv_cov


def scv_covs_with_same_minimum_eigenvalue(N, K):
    """
    Return SCV covariance matrices of dimensions (K,K,N) that have the same minimum eigenvalue (to violate the minvar
    source identification condition).


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

    # all generated eigenvalues must be non-negative, and their sum must equal K
    Lambda = np.zeros((N, K))
    for n in range(N):
        lambda_K_1 = -1
        while lambda_K_1 < 0:  # this EV should be bigger than 0 to have all positive eigenvalues
            Lambda[n, 0] = 0.1
            Lambda[n, 1:K - 1] = np.random.uniform(0.2, 1, K - 2)
            lambda_K_1 = K - np.sum(Lambda[n, :])
        Lambda[n, K - 1] = lambda_K_1

    # create covariance matrices with these eigenvalue profiles
    scv_cov = np.zeros((K, K, N))
    for n in range(N):
        scv_cov[:, :, n] = random_correlation.rvs(Lambda[n, :])

    return scv_cov


def scv_covs_with_same_eigenvalues_different_eigenvectors(N, K):
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

    # same eigenvalue profile for all SCVs (generated as for violating minvar)
    Lambda = np.zeros(K)
    Lambda[0] = 0.1
    Lambda[1:K - 1] = np.random.uniform(0.2, 1, K - 2)
    Lambda[K - 1] = K - np.sum(Lambda)

    # create N random SCV covariance matrices
    scv_cov = np.zeros((K, K, N))
    for n in range(N):
        scv_cov[:, :, n] = random_correlation.rvs(Lambda)

    return scv_cov


def scv_covs_with_same_eigenvalues_different_eigenvectors_rank_R(N, K, R, alpha, beta):
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
    Lambda = calculate_eigenvalues_from_ccv_covariance_matrices(scv_covs_with_rank_R(N, K, R, alpha, beta))[0, :]

    # create N random SCV covariance matrices
    scv_cov = np.zeros((K, K, N))
    for n in range(N):
        scv_cov[:, :, n] = random_correlation.rvs(Lambda)

    return scv_cov


def scv_covs_with_same_eigenvalues_different_sign_eigenvectors(N, K):
    """
    Return SCV covariance matrices of dimension (K,K,N) that have the same eigenvalues and eigenvectors just differing
    in the sign:
    U_n = D_n U_1, where D_n is a diagonal matrix with diagonal elements +-1.
    This violates all mCCA conditions including genvar.


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

    # same eigenvalue profile for all SCVs (generated as for violating minvar)
    Lambda = np.zeros(K)
    Lambda[0] = 0.1
    Lambda[1:K - 1] = np.random.uniform(0.2, 1, K - 2)
    Lambda[K - 1] = K - np.sum(Lambda)

    # create covariance matrix for 1st SCV
    scv_cov = np.zeros((K, K, N))
    scv_cov[:, :, 0] = random_correlation.rvs(Lambda)

    for n in range(1, N):
        # create diagonal matrix with +-1 on the diagonal
        D = np.diag(np.sign(np.random.uniform(-1, 1, K)))
        scv_cov[:, :, n] = D @ scv_cov[:, :, 0] @ D
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
    while True:  # make sure that second largest EVs of all SCVs are smaller than largest EVs of all SCVs
        scv_cov = np.zeros((K, K, N))
        for n in range(N):
            if alpha[n] + beta > 1:
                raise ValueError("alpha + beta must be smaller or equal to 1")
            temp_rank_term = np.random.randn(K, R)
            temp_rank_term /= np.linalg.norm(temp_rank_term, axis=1, keepdims=True)
            temp_variability_term = np.random.randn(K, K)
            temp_variability_term /= np.linalg.norm(temp_variability_term, axis=1, keepdims=True)
            scv_cov[:, :, n] = alpha[n] * (temp_rank_term @ temp_rank_term.T) + beta * (
                    temp_variability_term @ temp_variability_term.T) ** 2 + (1 - alpha[n] - beta) * np.eye(K)

        Lambda = calculate_eigenvalues_from_ccv_covariance_matrices(scv_cov)
        Lambda = Lambda[:, ::-1]  # sort descending

        # largest EVs of all SCVs should be bigger than second largest SCVs + some margin, otherwise recreate
        if np.min(Lambda[:, 0]) > np.max(Lambda[:, 1]) + 1 / K:
            break
    return scv_cov


def scv_covs_for_minvar(N, K, alpha):
    Lambda = np.zeros((N, K))
    for n in range(N):
        Lambda[n, -1] = alpha[n]
        Lambda[n, :-1] = (K - alpha[n]) / (K - 1)
    # create N random SCV covariance matrices
    scv_cov = np.zeros((K, K, N))
    for n in range(N):
        scv_cov[:, :, n] = random_correlation.rvs(Lambda[n, :])

    return scv_cov
def save_joint_isi_and_runtime_results(N, K, T, n_montecarlo, scenarios, **kwargs):
    algorithms = ['sumcor', 'maxvar', 'minvar', 'ssqcor', 'genvar']

    for scenario_idx, scenario in enumerate(scenarios):
        print(f'Simulations for {scenario}')

        for run in range(n_montecarlo):
            print(f'Start run {run}...')

            if scenario == 'same_maximum_eigenvalue':
                scv_cov = scv_covs_with_same_maximum_eigenvalue(N, K)
            elif scenario == 'same_minimum_eigenvalue':
                scv_cov = scv_covs_with_same_minimum_eigenvalue(N, K)
            elif scenario == 'same_eigenvalues_different_eigenvectors':
                scv_cov = scv_covs_with_same_eigenvalues_different_eigenvectors(N, K)
            elif scenario == 'same_eigenvalues_different_eigenvectors_rank_1':
                scv_cov = scv_covs_with_same_eigenvalues_different_eigenvectors_rank_R(N, K, 1, **kwargs)
            elif scenario == 'same_eigenvalues_different_eigenvectors_rank_K':
                scv_cov = scv_covs_with_same_eigenvalues_different_eigenvectors_rank_R(N, K, K, alpha=[1, 1, 1, 1, 1],
                                                                                       beta=0.0)
            elif scenario == 'same_eigenvalues_different_sign_eigenvectors':
                scv_cov = scv_covs_with_same_eigenvalues_different_sign_eigenvectors(N, K)
            elif scenario[0:5] == 'rank_':
                scv_cov = scv_covs_with_rank_R(N, K, int(scenario[5:]), **kwargs)
            else:
                raise AssertionError(f"scenario '{scenario}' does not exist")

            X, A, S = generate_datasets_from_covariance_matrices(scv_cov, T)

            for algorithm_idx, algorithm in enumerate(algorithms):
                t_start = time.process_time()
                M = mcca(X, algorithm)[0]
                W = np.moveaxis(M, [0, 1, 2], [1, 0, 2])
                t_end = time.process_time()

                filename = Path(Path(__file__).parent.parent,
                                f'simulation_results/K_{K}/K_{K}_{scenario}_{algorithm}_run{run}.npy')
                np.save(filename, {'joint_isi': _bss_isi(W, A)[1], 'runtime': t_end - t_start})


def save_violation_results_from_multiple_files_in_one_file(K, n_montecarlo):
    scenarios = ['same_maximum_eigenvalue', 'same_minimum_eigenvalue',
                 'same_eigenvalues_different_eigenvectors',
                 'same_eigenvalues_different_eigenvectors_rank_K', 'same_eigenvalues_different_eigenvectors_rank_1'
                 ]

    algorithms = ['sumcor', 'maxvar', 'minvar', 'ssqcor', 'genvar']

    results = {}
    for scenario_idx, scenario in enumerate(scenarios):
        results[scenario] = {}
        for algorithm_idx, algorithm in enumerate(algorithms):
            joint_isi = np.zeros(n_montecarlo)
            runtime = np.zeros(n_montecarlo)
            for run in range(n_montecarlo):
                filename = Path(Path(__file__).parent.parent,
                                f'simulation_results/K_{K}/K_{K}_{scenario}_{algorithm}_run{run}.npy')
                results_tmp = np.load(filename, allow_pickle=True).item()
                runtime[run] = results_tmp['runtime']
                joint_isi[run] = results_tmp['joint_isi']

            results[scenario][algorithm] = {'joint_isi': joint_isi, 'runtime': runtime}

    print(f'Save run as simulation_results/K_{K}/violations_K_{K}.npy.')
    np.save(Path(Path(__file__).parent.parent, f'simulation_results/K_{K}/violations_K_{K}.npy'), results)


def save_different_R_results_from_multiple_files_in_one_file(K, n_montecarlo):
    scenarios = [f'rank_{R}' for R in range(1, 11)]

    algorithms = ['sumcor', 'maxvar', 'minvar', 'ssqcor', 'genvar']

    results = {}
    for scenario_idx, scenario in enumerate(scenarios):
        results[scenario] = {}
        for algorithm_idx, algorithm in enumerate(algorithms):
            joint_isi = np.zeros(n_montecarlo)
            runtime = np.zeros(n_montecarlo)
            for run in range(n_montecarlo):
                filename = Path(Path(__file__).parent.parent,
                                f'simulation_results/K_{K}/K_{K}_{scenario}_{algorithm}_run{run}.npy')
                results_tmp = np.load(filename, allow_pickle=True).item()
                runtime[run] = results_tmp['runtime']
                joint_isi[run] = results_tmp['joint_isi']

            results[scenario][algorithm] = {'joint_isi': joint_isi, 'runtime': runtime}

    print(f'Save run as simulation_results/K_{K}/different_R_K_{K}.npy.')
    np.save(Path(Path(__file__).parent.parent, f'simulation_results/K_{K}/different_R_K_{K}.npy'), results)
