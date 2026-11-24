"""Loading the single-frequency datasets of store_training_xy."""

from ucmf_data import stored

import numpy as np
import pytest

HE = (2.0078, 4.0155, 4.0746, 30.1755)
ROWS, M, N_GRID = 50, 4, 5


@pytest.fixture
def root(tmp_path):
    """Frequencies laid out like store_training_xy; y[i] = 1000 He + i, x = -y. The last one is incomplete."""
    for he in HE:
        directory = tmp_path / f'he_{he:07.4f}'
        for split in ('training', 'validation'):
            y = (1000 * he + np.arange(ROWS))[:, None].repeat(M, axis=1).astype(np.float32)
            for feature, values in (('y', y), ('x', -y[:, :1].repeat(N_GRID, axis=1))):
                (directory / f'{split}_{feature}').mkdir(parents=True)
                np.save(directory / f'{split}_{feature}' / 'random_name_000000.npy', values)
        if he != HE[-1]:
            np.savez(directory / 'meta.npz')
    return tmp_path


def test_stored_he_lists_complete_frequencies(root):
    np.testing.assert_allclose(stored.stored_he(root), HE[:-1])


def test_load_frequency_takes_nearest_bin(root):
    data = stored.load_frequency(root, 4.06, 'validation')
    assert data['y'].dtype == np.float32 and data['y'].shape == (ROWS, M) and data['x'].shape == (ROWS, N_GRID)
    np.testing.assert_allclose(data['y'][:, 0], 1000 * 4.0746 + np.arange(ROWS), rtol=1e-6)
    np.testing.assert_array_equal(data['x'], -data['y'][:, :1].repeat(N_GRID, axis=1))


def test_load_band_reads_first_rows_per_frequency(root):
    data = stored.load_band(root, 4.0, 16.0, 60, 0, 'training')
    assert data['y'].shape == (60, M) and data['x'].shape == (60, N_GRID) and data['he'].shape == (60,)
    assert set(np.unique(data['he'])) <= {np.float32(4.0155), np.float32(4.0746)}
    for he in np.unique(data['he']):
        rows = data['y'][data['he'] == he, 0]
        np.testing.assert_allclose(rows, 1000 * he + np.arange(rows.size), rtol=1e-6)
    np.testing.assert_array_equal(data['x'], -data['y'][:, :1].repeat(N_GRID, axis=1))


def test_load_band_weights_rows_by_inverse_he(root):
    he = stored.load_band(root, 0.0, 16.0, 30, 1, 'training')['he']
    np.testing.assert_array_equal(he, stored.load_band(root, 0.0, 16.0, 30, 1, 'training')['he'])
    assert (he == np.float32(2.0078)).sum() > (he == np.float32(4.0746)).sum()
