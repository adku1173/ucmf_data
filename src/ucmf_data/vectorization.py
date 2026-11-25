"""Real vectorization of CSMs, delegated to ucmf so that data and solvers share one convention.

ucmf's ``HermitianVectorization`` keeps the lower triangle of CSM.T (BeamformerCMF ordering),
real parts first, then imaginary off-diagonal parts, weighted by sqrt(2) off the diagonal, so
the Euclidean norm equals the Hermitian Frobenius norm. r_diag removes the real diagonal entries.
"""

import torch
from ucmf.optim.operator import HermitianVectorization


def vectorize_csm(csm, r_diag):
    """Vectorize CSMs (..., M, M) to real vectors (..., M_eff); M_eff = M*M or M*(M-1) for r_diag."""
    csm = torch.from_numpy(csm)
    return HermitianVectorization(csm.shape[-1], r_diag=r_diag)(csm.real, csm.imag).numpy()


def kept_rows(n_mics, r_diag):
    """Boolean mask over the full vectorization (M*M rows) of the rows kept for r_diag."""
    return (~HermitianVectorization(n_mics).diagonal_rows() | (not r_diag)).numpy()
