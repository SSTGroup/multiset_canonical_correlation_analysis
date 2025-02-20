# Multiset Canonical Correlation Analysis
   
Python implementation of multiset Canonical Correlation Analysis (mCCA) methods:
sumcor, maxvar, minvar, ssqcor, genvar.

- **Source-code:** https://github.com/SSTGroup/multiset_canonical_correlation_analysis


## Installing multiset_canonical_correlation_analysis

The only pre-requisite is to have **Python 3** (>= version 3.11) installed.
The mCCA package can be installed with

    pip install multiset_canonical_correlation_analysis

Required third party packages will automatically be installed.


## Quickstart

First, the imports:

    import numpy as np
    from multiset_canonical_correlation_analysis import mcca, simulations
    from independent_vector_analysis.helpers_iva import _bss_isi


Create $K=100$ datasets with $T=10000$ samples of $N=5$ source components in each dataset.
The $N$ SCVs all have a different maximum eigenvalue.

    N = 5
    K = 100
    T = 10000
    alpha = 1 - (K - np.array([10, 15, 20, 25, 30])) / (K - 1)
    scv_cov = simulations.scv_covs_with_rank_R(N, K, 1, alpha=alpha, beta=0.0)
    X, A, S = simulations.generate_datasets_from_covariance_matrices(scv_cov, T)

where
* `S` : true sources of dimensions N x T x K
* `A` : mixing matrix of dimensions N x N x K
* `X` : observed datasets of dimensions N x T x K

Apply mCCA-genvar to find the canonical variables.

    transform, U = mcca.mcca(X, 'genvar')

where
* `transform` : transformation matrix of dimensions N x N x K
* `U` : canonical variables of dimensions N x T x K, with `U[:,:,k] = transform[:,:,k].T @ X[:,:,k]`
 
Calculate the jISI to evaluate JBSS performance.

    W = np.moveaxis(T, [0, 1, 2], [1, 0, 2])
    print(f'joint_isi genvar: {_bss_isi(W, A)[1]}')

A jISI smaller than 0.05 means successful JBSS, i.e., the canonical variables are permuted and scaled versions of the true sources.


## Contact

In case of questions, suggestions, problems etc., please send an email to isabell.lehmann@sst.upb.de, or open an issue here on Github.

## Citing

If you use this package in an academic paper, please cite [1].

    @article{Lehmann2025,
      title={A Comprehensive Guide to Multiset Canonical Correlation Analysis and its Application to Joint Blind Source Separation},
      author={Lehmann, Isabell and Gabrielson, Ben and Hasija, Tanuj and Adali, T{\"u}lay},
      journal={tba},
      year={tba},
      publisher={tba}
    }

[1] I. Lehmann, B. Gabrielson, et al.,
**A Comprehensive Guide to Multiset Canonical Correlation Analysis and its Application to Joint Blind Source Separation**,
*submitted in 2025*
