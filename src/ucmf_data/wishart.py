"""Finite-snapshot CSM sampling."""

import numpy as np


def sample_complex_wishart_covariance(scale, df, rng):
    """Return W / df with W ~ W_C(df, scale), so that its expectation is scale.

    scale is a Hermitian positive-definite (d, d) matrix; df > d - 1 is the
    equivalent number of independent complex snapshots.
    """
    dim = scale.shape[0]
    df = float(df)
    complex_dtype = np.result_type(scale.dtype, np.complex64)
    chol = np.linalg.cholesky(scale).astype(complex_dtype, copy=False)

    bartlett = np.zeros((dim, dim), dtype=complex_dtype)
    lower = np.tril_indices(dim, k=-1)
    n_lower = lower[0].size
    if n_lower:
        covariances = rng.normal(size=n_lower) + 1j * rng.normal(size=n_lower)
        bartlett[lower] = (covariances * np.sqrt(0.5)).astype(complex_dtype, copy=False)

    # Lower-triangular complex Bartlett factor. For zero-based row k, the k
    # strict-lower entries contribute k in expectation, so the diagonal Gamma
    # shape is df - k to make each row norm have expectation df.
    diag_shapes = df - np.arange(dim, dtype=float)
    bartlett[np.diag_indices(dim)] = np.sqrt(rng.gamma(shape=diag_shapes, scale=1.0)).astype(complex_dtype, copy=False)

    factor = chol @ bartlett
    return (factor @ factor.conjugate().T) / df
