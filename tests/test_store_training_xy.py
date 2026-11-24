"""Single-frequency .npy files of store_training_xy."""

from pathlib import Path

from ucmf_data import Parameters
from ucmf_data import store_training_xy as store

import numpy as np
import pytest
import ray

SIZE, M, N_GRID = 12, 4, 5


class FakeDataset:
    """Stands in for acoupipe's Dataset: unordered float64 rows of one frequency."""

    def get_ray_dataset(self, features, size, split):
        rows = np.random.default_rng(0).permutation(size)
        return ray.data.from_items(
            [{'idx': i, 'y': np.full((1, M), 10.0 * i), 'x': np.full((1, N_GRID), -10.0 * i)} for i in rows]
        )


@pytest.fixture(scope='module')
def ray_session():
    ray.init(num_cpus=2, include_dashboard=False, ignore_reinit_error=True)
    yield
    ray.shutdown()


def load(directory):
    (file,) = Path(directory).glob('*.npy')
    return np.load(file)


def test_write_split_writes_corresponding_float32_rows(tmp_path, ray_session):
    store.write_split(FakeDataset(), 'training', SIZE, tmp_path)
    y, x = load(tmp_path / 'training_y'), load(tmp_path / 'training_x')
    assert y.dtype == x.dtype == np.float32
    assert y.shape == (SIZE, M) and x.shape == (SIZE, N_GRID)
    # Any row order, but every scene once and y, x of one scene in the same row.
    np.testing.assert_array_equal(np.sort(y[:, 0]), 10.0 * np.arange(SIZE))
    np.testing.assert_array_equal(x, -y[:, :1].repeat(N_GRID, axis=1))


def test_he_bins_are_band_plus_lowest_and_highest_bin():
    parameters = Parameters()
    he = store.he_bins(parameters)
    step = parameters.fs / parameters.block_size * parameters.aperture / parameters.c
    assert he[0] >= 2.0 > he[0] - step
    assert np.isclose(he[-1], (parameters.block_size // 2 - 1) * step)
    band = he[1:-1]
    assert band[0] >= 4.0 > band[0] - step and band[-1] <= 16.0 < band[-1] + step
    np.testing.assert_allclose(np.diff(band), step)
