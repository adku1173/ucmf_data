"""Scene sampling through acoupipe's normal generation and HDF5 paths."""

from types import SimpleNamespace

from acoupipe.datasets import Dataset

from ucmf_data import Config, MiracleA1Parameters, Parameters
from ucmf_data.geometry import aperture, vogel_subarray

import h5py
import numpy as np
import pytest
from scipy.spatial import cKDTree

FEATURES = ['nsources', 'source_locations', 'source_rms', 'mics', 'c', 'fs', 'signal_length', 'snr_db']


@pytest.fixture
def dataset():
    config = Config(parameters=Parameters(source_grid=None))
    return Dataset(config=config)


def samples(dataset, size, **kwargs):
    return list(dataset.generate(features=FEATURES, size=size, progress_bar=False, **kwargs))


def test_scene_contract_and_distributions(dataset):
    nominal = vogel_subarray(16)
    nominal_aperture = aperture(nominal)
    scenes = samples(dataset, 2048)
    counts, powers, errors, locations = [], [], [], []
    for scene in scenes:
        loc, rms = scene['source_locations'], scene['source_rms']
        count = loc.shape[1]
        assert scene['nsources'] == count
        assert loc.shape == (3, count)
        assert rms.shape == (count,)
        assert count >= 1
        cells = cKDTree(dataset.config.parameters.focus_grid.T).query(loc.T)[1]
        assert np.unique(cells).size == count
        assert np.all(np.abs(loc[:2]) <= nominal_aperture / 2)
        np.testing.assert_array_equal(loc[2], np.full(count, 0.75))
        assert np.all(rms > 0)
        np.testing.assert_array_equal(scene['mics'][2], nominal[2])
        assert (scene['c'], scene['fs'], scene['signal_length'], scene['snr_db']) == (343, 32000, 16, 20)
        counts.append(count)
        powers.extend(rms**2)
        locations.extend(loc[:2].T)
        errors.append(scene['mics'][:2] - nominal[:2])
    assert np.mean(counts) == pytest.approx(4, abs=0.15)
    assert np.mean(powers) == pytest.approx(5 * np.sqrt(np.pi / 2), abs=0.2)
    assert np.mean(np.square(powers)) == pytest.approx(50, abs=2)
    locations = np.asarray(locations)
    assert np.unique(locations[:, 0]).size == len(locations)  # Continuous, not quantised.
    assert np.mean(locations) == pytest.approx(0, abs=0.005 * nominal_aperture)
    assert np.std(locations) == pytest.approx(0.1688 * nominal_aperture, rel=0.06)
    assert np.mean(errors) == pytest.approx(0, abs=0.0001)
    assert np.std(errors) == pytest.approx(0.005, rel=0.03)


def test_reproducibility_and_split(dataset):
    first, repeated = samples(dataset, 3), samples(dataset, 3)
    resumed = samples(dataset, 1, start_idx=2)[0]
    for left, right in zip(first, repeated, strict=True):
        for name in FEATURES:
            np.testing.assert_array_equal(left[name], right[name])
    for name in FEATURES:
        np.testing.assert_array_equal(first[2][name], resumed[name])
    validation = samples(dataset, 1, split='validation')[0]
    assert not np.array_equal(first[0]['mics'], validation['mics'])
    assert not np.array_equal(first[0]['mics'], first[1]['mics'])


def test_single_source_and_zero_microphone_error():
    parameters = Parameters(nsources=1, mic_position_std=0)
    nominal = parameters.mics.copy()
    config = Config(parameters=parameters)
    assert [sampler.attribute for sampler in config.get_sampler().values()] == [
        'nsources',
        'source_locations',
        'source_rms',
        'signal_seed',
    ]
    for scene in samples(Dataset(config=config), 20):
        assert scene['source_locations'].shape == (3, 1)
        assert scene['source_rms'].size == 1
        np.testing.assert_array_equal(scene['mics'], nominal)


def test_microphone_sampler_reads_current_parameter():
    parameters = Parameters(mic_position_std=0.01)
    nominal = parameters.mics.copy()
    config = Config(parameters=parameters)
    sampler = next(sampler for sampler in config.get_sampler().values() if sampler.attribute == 'mics')
    for std in [0.01, 0.002, 0.0]:
        parameters.mic_position_std = std
        expected = nominal.copy()
        expected[:2] += np.random.default_rng(3).normal(0.0, std, size=nominal[:2].shape)
        actual = sampler.random_func(np.random.default_rng(3), parameters)
        np.testing.assert_array_equal(actual, expected)


def test_continuous_source_counts_respect_reconstruction_capacity():
    config = Config(parameters=Parameters(source_grid=None, focus_grid=grid(2), nsources=6))
    scenes = samples(Dataset(config=config), 50)
    assert max(scene['source_locations'].shape[1] for scene in scenes) == 4
    for scene in scenes:
        cells = cKDTree(config.parameters.focus_grid.T).query(scene['source_locations'].T)[1]
        assert np.unique(cells).size == scene['source_locations'].shape[1]


def grid(points):
    ap = aperture(vogel_subarray(16))
    axis = np.linspace(-ap / 2, ap / 2, points)
    x, y = np.meshgrid(axis, axis)
    return np.vstack((x.ravel(), y.ravel(), np.full(x.size, 0.5)))


def test_default_grid_sources_are_distinct_grid_points():
    source_grid = Parameters().source_grid
    config = Config()
    for scene in samples(Dataset(config=config), 200):
        loc = scene['source_locations']
        matches = np.all(np.isclose(loc[:, :, None], source_grid[:, None, :]), axis=0)
        assert np.all(matches.sum(axis=1) == 1)
        assert np.unique(np.argmax(matches, axis=1)).size == loc.shape[1]
        assert scene['source_rms'].shape == (loc.shape[1],)


def test_grid_source_counts_respect_capacity():
    config = Config(parameters=Parameters(source_grid=grid(2), nsources=6))
    counts = [scene['source_locations'].shape[1] for scene in samples(Dataset(config=config), 50)]
    assert max(counts) == 4


def test_miracle_a1_samples_without_microphone_sampler():
    parameters = MiracleA1Parameters()
    mics = parameters.mics.copy()
    config = Config(parameters=parameters)
    assert all(sampler.attribute != 'mics' for sampler in config.get_sampler().values())
    for scene in samples(Dataset(config=config), 50):
        loc = scene['source_locations']
        matches = np.all(np.isclose(loc[:, :, None], parameters.source_grid[:, None, :]), axis=0)
        assert np.all(matches.sum(axis=1) == 1)
        cells = cKDTree(parameters.focus_grid.T).query(loc.T)[1]
        assert np.unique(cells).size == loc.shape[1]
        np.testing.assert_array_equal(scene['mics'], mics)


@pytest.mark.parametrize('discrete', [False, True])
def test_only_colliding_positions_are_redrawn(discrete):
    parameters = Parameters(source_grid=grid(4) if discrete else None, focus_grid=grid(2), nsources=2)
    config = Config(parameters=parameters)
    proposals = iter([np.array([[-0.2, -0.1], [-0.2, -0.1]]), np.array([[0.2], [0.2]])])
    draw_sizes = []

    def normal(loc, scale, size):
        draw_sizes.append(size)
        return next(proposals)

    rng = SimpleNamespace(normal=normal)  # Locations cannot draw a source count.
    locations = list(config.get_sampler().values())[1].random_func(rng, parameters)
    assert draw_sizes == [(2, 2), (2, 1)]
    expected_first = np.array([-0.2, -0.2, 0.75])
    if discrete:
        index = np.argmin(np.sum((parameters.source_grid[:2] + 0.2) ** 2, axis=0))
        expected_first = parameters.source_grid[:, index]
    np.testing.assert_array_equal(locations[:, 0], expected_first)
    assert np.unique(cKDTree(parameters.focus_grid.T).query(locations.T)[1]).size == 2


def test_capacity_counts_only_reachable_reconstruction_cells():
    source_grid = np.array([[-0.2, -0.1], [-0.2, -0.1], [0.75, 0.75]])
    parameters = Parameters(source_grid=source_grid, focus_grid=grid(2), nsources=2)
    config = Config(parameters=parameters)
    for scene in samples(Dataset(config=config), 20):
        assert scene['source_locations'].shape[1] == 1


def test_fine_source_grid_respects_coarse_reconstruction_capacity():
    parameters = Parameters(source_grid=grid(8), focus_grid=grid(2), he=[4, 8], nsources=6)
    config = Config(parameters=parameters)
    scenes = list(
        Dataset(config=config).generate(
            features=['x', 'source_strength', 'source_locations'],
            size=50,
            progress_bar=False,
        )
    )
    assert max(scene['source_locations'].shape[1] for scene in scenes) == 4
    for scene in scenes:
        count = scene['source_locations'].shape[1]
        np.testing.assert_array_equal(np.count_nonzero(scene['x'], axis=1), [count, count])
        np.testing.assert_allclose(scene['x'].sum(axis=1), scene['source_strength'].sum(axis=1))


def test_nsources_sampler_keeps_initial_mean_between_cases():
    parameters = Parameters(nsources=4)
    config = Config(parameters=parameters)
    count_sampler = list(config.get_sampler().values())[0]
    assert count_sampler.attribute == 'nsources'
    draws = iter([0, 4])
    means = []

    def poisson(mean):
        means.append(mean)
        return next(draws)

    rng = SimpleNamespace(poisson=poisson)
    parameters.nsources = count_sampler.random_func(rng, parameters)
    assert parameters.nsources == 1
    parameters.nsources = count_sampler.random_func(rng, parameters)
    assert parameters.nsources == 5
    assert means == [3, 3]


def test_hdf5_matches_generated_samples(dataset, tmp_path):
    expected = samples(dataset, 3)
    path = tmp_path / 'scenes.h5'
    dataset.save_h5(features=FEATURES, size=3, name=str(path), progress_bar=False)
    with h5py.File(path, 'r') as file:
        assert len(file) == 3
        for index, scene in enumerate(expected):
            for name in FEATURES:
                np.testing.assert_array_equal(file[f'{index}/{name}'][()], scene[name])
