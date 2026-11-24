"""Frobenius-isometric real vectorization of CSMs.

Rows follow the PhysicalModel / BeamformerCMF ordering: real parts of the lower
triangle of CSM.T, then imaginary parts of its off-diagonal entries. Real and
imaginary off-diagonal entries are weighted by sqrt(2), so the Euclidean norm
equals the Hermitian Frobenius norm. r_diag removes the real diagonal entries.
"""

import numpy as np


def vectorize_csm(csm, r_diag):
    """Vectorize CSMs (..., M, M) to real vectors (..., M_eff); M_eff = M*M or M*(M-1) for r_diag."""
    n_mics = csm.shape[-1]
    lower = np.tril(np.ones((n_mics, n_mics))).reshape(-1) > 0
    off_diagonal = (np.eye(n_mics).reshape(-1) == 0)[lower]
    rows = np.hstack([off_diagonal if r_diag else np.ones(off_diagonal.size, dtype=bool), off_diagonal])
    weights = np.where(off_diagonal, np.sqrt(2.0), 1.0)
    weights = np.concatenate([weights, weights])[rows]

    flattened = np.swapaxes(csm, -1, -2).reshape(*csm.shape[:-2], n_mics * n_mics)[..., lower]
    stacked = np.concatenate([flattened.real, flattened.imag], axis=-1)
    return stacked[..., rows] * weights.astype(np.result_type(stacked.dtype, np.float32))
