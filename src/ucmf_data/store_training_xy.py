"""Write one dataset per FFT bin: all bins with 4 <= He <= 16, the first bin with
He >= 2 and the last bin below Nyquist.

Run with: uv run python -m ucmf_data.store_training_xy

Every frequency is simulated on its own and written to ``OUT/he_<He>/``: the splits
training/validation hold the lean features y and x as one uncompressed float32 .npy
file each, in the directories ``<split>_y/`` (N, M) and ``<split>_x/`` (N, N_grid), so
one frequency loads with a single sequential read. Rows of y and x correspond; their
order is arbitrary. The test split adds cleansc and is written as HDF5. y and A use
the full vectorization (r_diag=False); r_diag=True selects rows of both. ``meta.npz``
holds A, the nominal transfer H for the operators built from H, and the settings of
the frequency; it is written last and marks the frequency as complete, so an
interrupted run continues with the first incomplete frequency.
Generation is parallelized with Ray across ``GEN_WORKERS`` workers.
"""

import os

# Keep every Ray worker single-threaded (see DistributedPipeline warning).
for _var in ('NUMBA_NUM_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_var, '1')

from acoupipe.datasets import Dataset

from ucmf_data import Config, Parameters
from ucmf_data.features import frequencies, nominal_transfer, sensing_matrix

import numpy as np
import ray

OUT = '/mnt/data/kujawski/promotion/synthetic'
TRAIN_SIZE = 1_000_000  # 1_000_000 for the full dataset
VAL_SIZE = 10_000
TEST_SIZE = 10_000
TASKS = os.cpu_count()
# Leave CPUs free for the downstream write stage; otherwise the generation actor
# pool holds every CPU and the write stage starves (deadlock).
GEN_WORKERS = max(1, TASKS - 8)
SAMPLES_PER_TASK = 64
DATASET_SEED = 0
FEATURES = ('y', 'x')


def he_bins(parameters):
    """FFT-bin Helmholtz numbers: all with 4 <= He <= 16, the first with He >= 2, the last below Nyquist."""
    step = parameters.fs / parameters.block_size * parameters.aperture / parameters.c
    he = np.arange(int(np.ceil(2.0 / step)), parameters.block_size // 2) * step
    return np.concatenate([he[:1], he[(he >= 4.0) & (he <= 16.0)], he[-1:]])


def single_frequency(batch):
    """Drop the frequency axis of the one-frequency features and cast them to float32."""
    return {feature: batch[feature][:, 0].astype(np.float32) for feature in FEATURES}


def write_split(dataset, split, size, path):
    """Generate a split once and write y and x as one .npy file each, with corresponding rows."""
    rows = dataset.get_ray_dataset(list(FEATURES), size=size, split=split)
    rows = rows.map_batches(single_frequency, batch_format='numpy').repartition(1).materialize()
    for feature in FEATURES:
        rows.write_numpy(os.path.join(path, f'{split}_{feature}'), column=feature)


def write_frequency(he, path):
    """Simulate and write all splits of one He bin; skip it if meta.npz marks it complete."""
    if os.path.exists(os.path.join(path, 'meta.npz')):
        return
    os.makedirs(path, exist_ok=True)
    parameters = Parameters(he=he)
    dataset = Dataset(
        config=Config(parameters=parameters), tasks=GEN_WORKERS, samples_per_task=SAMPLES_PER_TASK, seed=DATASET_SEED
    )
    write_split(dataset, 'training', TRAIN_SIZE, path)
    write_split(dataset, 'validation', VAL_SIZE, path)
    dataset.save_h5(['x', 'y', 'cleansc'], size=TEST_SIZE, split='test', name=os.path.join(path, 'test.h5'))
    np.savez(
        os.path.join(path, 'meta.npz'),
        A=sensing_matrix(parameters, {})[0].astype(np.float32),  # (M_eff, N_grid)
        H=nominal_transfer(parameters)[0].astype(np.complex64),  # (M, N_grid)
        r_diag=parameters.r_diag,
        he=np.float32(he),
        frequency=np.float32(frequencies(parameters)[0]),
        dataset_seed=DATASET_SEED,
        nsources=parameters.nsources,
        snr_db=parameters.snr_db,
        block_size=parameters.block_size,
        fs=parameters.fs,
        n_mics=parameters.num_mics,
        aperture=parameters.aperture,
        c=parameters.c,
    )


def main():
    ray.init(num_cpus=TASKS, ignore_reinit_error=True)
    for he in he_bins(Parameters()):
        write_frequency(he, os.path.join(OUT, f'he_{he:07.4f}'))


if __name__ == '__main__':
    main()
