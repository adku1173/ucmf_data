"""Loading the single-frequency datasets of store_training_xy."""

from ucmf_data import stored
from ucmf_data.vectorization import kept_rows, vectorize_csm

import numpy as np
import pytest

HE = (2.0078, 4.0155, 4.0746, 30.1755)
ROWS, M, N_GRID = 50, 9, 5  # M = 3 * 3: full vectorization of three microphones


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
            H = np.full((3, N_GRID), he + 1j, dtype=np.complex64)
            A = np.full((M, N_GRID), he, dtype=np.float32) + np.arange(M)[:, None]
            np.savez(directory / 'meta.npz', H=H, A=A)
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


def test_load_bank_stacks_the_band_in_band_order(root):
    np.testing.assert_allclose(stored.band_he(root, 4.0, 16.0), [4.0155, 4.0746])
    H, A = stored.load_bank(root, 4.0, 16.0), stored.load_bank(root, 4.0, 16.0, 'A')
    assert H.shape == (2, 3, N_GRID) and H.dtype == np.complex64 and A.shape == (2, M, N_GRID)
    np.testing.assert_allclose(H[:, 0, 0].real, [4.0155, 4.0746], rtol=1e-6)


def test_load_band_index_points_to_the_bank_entry_of_each_row(root):
    data = stored.load_band(root, 0.0, 16.0, 40, 2, 'training')
    bank_he = stored.band_he(root, 0.0, 16.0)
    np.testing.assert_allclose(bank_he[data['index']], data['he'], rtol=1e-6)
    H = stored.load_bank(root, 0.0, 16.0)
    np.testing.assert_allclose(H[data['index'], 0, 0].real, data['he'], rtol=1e-6)


def test_kept_rows_select_the_r_diag_vectorization():
    rng = np.random.default_rng(0)
    h = rng.standard_normal((3, 2)) + 1j * rng.standard_normal((3, 2))
    csm = h @ h.conj().T
    full = vectorize_csm(csm, False)
    np.testing.assert_array_equal(vectorize_csm(csm, True), full[kept_rows(3, True)])
    assert kept_rows(3, False).all() and kept_rows(3, True).sum() == 6


def test_r_diag_drops_the_diagonal_rows_of_y_and_a(root):
    rows = kept_rows(3, True)
    full, reduced = (stored.load_band(root, 0.0, 16.0, 20, 3, 'training', r_diag=r) for r in (False, True))
    np.testing.assert_array_equal(reduced['y'], full['y'][:, rows])
    np.testing.assert_array_equal(reduced['x'], full['x'])
    reduced_frequency = stored.load_frequency(root, 4.0, 'training', r_diag=True)['y']
    np.testing.assert_array_equal(reduced_frequency, stored.load_frequency(root, 4.0, 'training')['y'][:, rows])
    A = stored.load_bank(root, 0.0, 16.0, 'A')
    np.testing.assert_array_equal(stored.load_bank(root, 0.0, 16.0, 'A', r_diag=True), A[:, rows])
    np.testing.assert_array_equal(stored.load_bank(root, 0.0, 16.0, r_diag=True), stored.load_bank(root, 0.0, 16.0))


def test_load_band_weights_rows_by_inverse_he(root):
    he = stored.load_band(root, 0.0, 16.0, 30, 1, 'training')['he']
    np.testing.assert_array_equal(he, stored.load_band(root, 0.0, 16.0, 30, 1, 'training')['he'])
    assert (he == np.float32(2.0078)).sum() > (he == np.float32(4.0746)).sum()
