"""Access to the measured MIRACLE impulse responses (HDF5 via irdl)."""

from pathlib import Path

import numpy as np


def miracle_path(scenario):
    """Return the MIRACLE HDF5 file of scenario, preferring an existing irdl cache file.

    The cache lookup avoids repeated DOI/provider HTTP requests in parallel runs.
    """
    from irdl import MiracleDataset
    from irdl.cache import IRDL_CACHE_DIR

    root = Path(IRDL_CACHE_DIR) / 'MIRACLE'
    for path in (root / 'provider' / f'{scenario}.h5', root / 'ingest' / f'{scenario}.h5',
                 root / 'output' / f'{scenario}.h5'):
        if path.exists():
            return path
    return Path(MiracleDataset.get(scenario=scenario, output_format='raw'))


def _positions(file, name):
    pos = np.asarray(file[f'data/location/{name}'][()], dtype=float)
    if pos.ndim == 3 and pos.shape[-1] == 1:
        pos = pos[..., 0]
    return pos if pos.shape[0] == 3 else pos.T


def receiver_positions(file):
    """Return the measured receiver positions (3, R) of an open MIRACLE file."""
    return _positions(file, 'receiver')


def source_positions(file):
    """Return the measured source positions (3, S) of an open MIRACLE file."""
    return _positions(file, 'source')


def snap_frequency_to_fft_bin(frequency, fs, block_size):
    """Return the nearest FFT bin index of a block_size grid at fs and its frequency."""
    index = int(np.rint(float(frequency) / (float(fs) / int(block_size))))
    return index, index * float(fs) / int(block_size)


def transfer_columns_at_bin(file, sources, receivers, block_size, fft_bin):
    """Return complex transfer functions (R, S) at FFT bin fft_bin of the block_size grid.

    Each impulse response is zero-padded to the next power of two of at least
    block_size, so that fft_bin maps to an exact bin of the padded FFT.
    """
    transfer = np.empty((receivers.size, sources.size), dtype=np.complex128)
    for col, source in enumerate(sources):
        ir = np.asarray(file['data/impulse_response'][int(source)])
        if ir.ndim == 3 and ir.shape[-1] == 1:
            ir = ir[..., 0]
        nfft = 1 << (max(int(ir.shape[-1]), int(block_size)) - 1).bit_length()
        transfer[:, col] = np.fft.rfft(ir[receivers], n=nfft, axis=-1)[:, int(fft_bin) * (nfft // int(block_size))]
    return transfer
