"""Load the single-frequency datasets written by ``store_training_xy``.

Only complete frequencies count, i.e. directories ``he_<He>/`` holding ``meta.npz``.
Arrays are returned as float32 NumPy arrays; rows of y and x correspond. The files hold
the full vectorization (r_diag=False); r_diag=True drops the CSM diagonal rows of y and A
while loading (kept_rows). H is independent of r_diag, which ucmf's CMF operators take.

A band of frequencies is a bank of dictionaries for training: ``load_bank`` stacks one
dictionary per frequency of the band, and ``load_band`` gives every row the position
``index`` of its frequency in that bank (both ordered by ``band_he``).
"""

from math import isqrt
from pathlib import Path

from ucmf_data.vectorization import kept_rows

import numpy as np

FEATURES = ('y', 'x')


def stored_he(root):
    """Sorted Helmholtz numbers of all complete frequencies under root."""
    return np.array(sorted(float(meta.parent.name[3:]) for meta in Path(root).glob('he_*/meta.npz')))


def frequency_dir(root, he):
    """Directory of the stored frequency nearest to he; it also holds meta.npz and test.h5."""
    axis = stored_he(root)
    return directory_of(root, axis[np.argmin(np.abs(axis - he))])


def directory_of(root, he):
    """Directory of the stored frequency he, as listed by stored_he."""
    return Path(root) / f'he_{he:07.4f}'


def read(directory, split, feature):
    """Memory-map the single .npy file of one split and feature."""
    (file,) = (Path(directory) / f'{split}_{feature}').glob('*.npy')
    return np.load(file, mmap_mode='r')


def read_rows(directory, split, feature, size=None, r_diag=False):
    """Read the first size rows (all for None) of one split and feature; y without diagonal for r_diag."""
    array = read(directory, split, feature)[:size]
    if feature == 'y':
        return array[:, kept_rows(isqrt(array.shape[1]), r_diag)]
    return np.array(array)


def load_frequency(root, he, split, r_diag=False):
    """Load all rows of the stored frequency nearest to he as {'y': (N, M_eff), 'x': (N, N_grid)}."""
    directory = frequency_dir(root, he)
    return {feature: read_rows(directory, split, feature, r_diag=r_diag) for feature in FEATURES}


def band_he(root, lo, hi):
    """Ascending Helmholtz numbers of the stored frequencies with lo <= He <= hi; the bank order."""
    axis = stored_he(root)
    return axis[(axis >= lo) & (axis <= hi)]


def load_bank(root, lo, hi, feature='H', r_diag=False):
    """Stack one dictionary per frequency of the band, in the order of band_he.

    feature 'H' gives the nominal transfer (K, M, N_grid) complex64 for ucmf's CMF
    operators (independent of r_diag), 'A' the dense sensing matrix (K, M_eff, N_grid)
    float32 for MatrixMult, without the diagonal rows for r_diag.
    """
    arrays = []
    for he in band_he(root, lo, hi):
        with np.load(directory_of(root, he) / 'meta.npz') as meta:
            arrays.append(meta[feature])
    bank = np.stack(arrays)
    return bank[:, kept_rows(isqrt(bank.shape[1]), r_diag)] if feature == 'A' else bank


def load_band(root, lo, hi, size, seed, split, r_diag=False):
    """Load size rows of the stored frequencies with lo <= He <= hi.

    The number of rows per frequency is drawn with weights 1/He (log-uniform in He,
    as the bins are equally spaced); each frequency contributes the first rows of its
    files, so it is read as one contiguous block. Rows are grouped by frequency and
    'he' (size,) holds the frequency of each row. Each frequency must hold at least
    the drawn number of rows. 'index' (size,) holds the position of each row's frequency
    in band_he, i.e. its entry in the bank of load_bank. r_diag drops the diagonal rows of y.
    """
    band = band_he(root, lo, hi)
    weights = 1.0 / band
    counts = np.random.default_rng(seed).multinomial(size, weights / weights.sum())
    data = {
        feature: np.concatenate(
            [read_rows(directory_of(root, he), split, feature, n, r_diag) for he, n in zip(band, counts, strict=True)]
        )
        for feature in FEATURES
    }
    data['he'] = np.repeat(band, counts).astype(np.float32)
    data['index'] = np.repeat(np.arange(band.size), counts)
    return data
