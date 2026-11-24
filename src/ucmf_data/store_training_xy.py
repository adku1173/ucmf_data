"""Write the band [4, 16] dataset.

Run with: uv run python -m ucmf_data.store_training_xy

Every scene is evaluated at the F discrete FFT bins with 4 <= He <= 16. The split
train/val holds the lean features x and y as sharded Parquet; the test split adds
cleansc and is written as HDF5. The shared sensing matrix A and the He/frequency
axis are identical for all splits and stored once in ``meta.npz``, not per sample.
Generation is parallelized with Ray across ``TASKS`` workers.
"""

import os

# Keep every Ray worker single-threaded (see DistributedPipeline warning).
for _var in ('NUMBA_NUM_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_var, '1')

from acoupipe.datasets import Dataset

from ucmf_data import Config, Parameters
from ucmf_data.features import frequencies, sensing_matrix

import numpy as np
import ray

BAND = (4.0, 16.0)
OUT = '/mnt/data/kujawski/promotion/band_4_16'
TRAIN_SIZE = 1_000_000
VAL_SIZE = 10_000
TEST_SIZE = 10_000
TASKS = os.cpu_count()
# Leave CPUs free for the downstream float32/compress/write stage; otherwise the
# generation actor pool holds every CPU and the write stage starves (deadlock).
GEN_WORKERS = max(1, TASKS - 8)
SAMPLES_PER_TASK = 64
ROWS_PER_PARQUET_FILE = 2500
DATASET_SEED = 0


def he_bins_in_band(parameters, lo, hi):
    """Discrete FFT-bin Helmholtz numbers with lo <= He <= hi (He >= 2 below Nyquist)."""
    df = parameters.fs / parameters.block_size
    scale = parameters.aperture / parameters.c
    k = np.arange(int(np.ceil(2.0 / (df * scale))), parameters.block_size // 2)
    he = k * df * scale
    return he[(he >= lo) & (he <= hi)]


def _to_float32(batch):
    return {'x': batch['x'].astype(np.float32), 'y': batch['y'].astype(np.float32), 'idx': batch['idx']}


def write_parquet_split(dataset, split, size, path):
    """Write the lean x/y features of a split as sharded, zstd-compressed Parquet."""
    pipeline = dataset.get_ray_dataset(['x', 'y'], size=size, split=split, start_idx=0)
    pipeline.map_batches(_to_float32, batch_format='numpy').write_parquet(
        path,
        arrow_parquet_args_fn=lambda: {'compression': 'zstd'},
        min_rows_per_file=ROWS_PER_PARQUET_FILE,
    )


def main():
    os.makedirs(OUT, exist_ok=True)
    parameters = Parameters(he=he_bins_in_band(Parameters(), *BAND))
    config = Config(parameters=parameters)

    ray.init(num_cpus=TASKS, ignore_reinit_error=True)
    dataset = Dataset(config=config, tasks=GEN_WORKERS, samples_per_task=SAMPLES_PER_TASK, seed=DATASET_SEED)

    np.savez(
        os.path.join(OUT, 'meta.npz'),
        A=sensing_matrix(parameters, {}).astype(np.float32),
        he=np.asarray(parameters.he, dtype=np.float32),
        frequencies=frequencies(parameters).astype(np.float32),
        band_lo=BAND[0],
        band_hi=BAND[1],
        dataset_seed=DATASET_SEED,
        nsources=parameters.nsources,
        snr_db=parameters.snr_db,
        block_size=parameters.block_size,
        fs=parameters.fs,
        n_mics=parameters.num_mics,
        aperture=parameters.aperture,
        c=parameters.c,
    )

    write_parquet_split(dataset, 'training', TRAIN_SIZE, os.path.join(OUT, 'train'))
    write_parquet_split(dataset, 'validation', VAL_SIZE, os.path.join(OUT, 'val'))
    dataset.save_h5(['x', 'y', 'cleansc'], size=TEST_SIZE, split='test', name=os.path.join(OUT, 'test.h5'))


if __name__ == '__main__':
    main()
