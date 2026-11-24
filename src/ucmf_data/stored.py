"""Load the single-frequency datasets written by ``store_training_xy``.

Only complete frequencies count, i.e. directories ``he_<He>/`` holding ``meta.npz``.
Arrays are returned as float32 NumPy arrays; rows of y and x correspond.
"""

from pathlib import Path

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


def load_frequency(root, he, split):
    """Load all rows of the stored frequency nearest to he as {'y': (N, M), 'x': (N, N_grid)}."""
    directory = frequency_dir(root, he)
    return {feature: np.array(read(directory, split, feature)) for feature in FEATURES}


def load_band(root, lo, hi, size, seed, split):
    """Load size rows of the stored frequencies with lo <= He <= hi.

    The number of rows per frequency is drawn with weights 1/He (log-uniform in He,
    as the bins are equally spaced); each frequency contributes the first rows of its
    files, so it is read as one contiguous block. Rows are grouped by frequency and
    'he' (size,) holds the frequency of each row. Each frequency must hold at least
    the drawn number of rows.
    """
    axis = stored_he(root)
    band = axis[(axis >= lo) & (axis <= hi)]
    weights = 1.0 / band
    counts = np.random.default_rng(seed).multinomial(size, weights / weights.sum())
    data = {
        feature: np.concatenate(
            [read(directory_of(root, he), split, feature)[:n] for he, n in zip(band, counts, strict=True)]
        )
        for feature in FEATURES
    }
    data['he'] = np.repeat(band, counts).astype(np.float32)
    return data
