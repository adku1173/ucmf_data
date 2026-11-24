"""Source-strength frequency/source axes and agreement with acoupipe's Q."""

import acoular as ac
from acoupipe.datasets import Dataset
from acoupipe.datasets.features import AnalyticSourceStrengthFeature, TargetmapFeature
from acoupipe.datasets.features.spectra_analytic import PowerSpectraAnalytic
from acoupipe.datasets.synthetic.legacy_config import DatasetSyntheticConfig

from ucmf_data import Config, MiracleA1Parameters, Parameters
from ucmf_data.geometry import aperture, vogel_subarray

import h5py
import numpy as np
import pytest

LOCATIONS = np.array([[0, 0.1], [0, 0], [0.75, 0.75]])


def rect_grid():
    ap = aperture(vogel_subarray(16))
    return ac.RectGrid(x_min=-0.5 * ap, x_max=0.5 * ap, y_min=-0.5 * ap, y_max=0.5 * ap, z=0.75, increment=ap / 15)


class FixedConfig(Config):
    def create_samplers(self):
        pass


@pytest.mark.parametrize('he', [4.0, [4.0], [2.0, 4.0, 8.0], np.array([8.0, 4.0])])
@pytest.mark.parametrize('block_size', [128, 1024])
def test_source_strength_shape_and_spectral_power(he, block_size):
    parameters = Parameters(
        he=he,
        block_size=block_size,
        source_rms=np.array([2.0, 3.0], dtype=np.float32),
        source_locations=LOCATIONS,
    )
    config = FixedConfig(parameters=parameters)
    sample = next(Dataset(config=config).generate(features=['source_strength'], size=1, progress_bar=False))
    expected = np.tile(np.array([4.0, 9.0]) / (block_size // 2 + 1), (np.size(he), 1))
    assert set(sample) == {'idx', 'source_strength'}
    assert sample['source_strength'].dtype == np.float64
    np.testing.assert_array_equal(sample['source_strength'], expected)


def test_source_strength_matches_acoupipe_q_and_analytic_feature():
    p = Parameters(he=[2, 4, 8], block_size=128, source_rms=np.array([2.0, 3.0]), source_locations=LOCATIONS)
    config = FixedConfig(parameters=p)
    sample = next(Dataset(config=config).generate(features=['source_strength'], size=1, progress_bar=False))
    spectra = PowerSpectraAnalytic(
        block_size=p.block_size,
        sample_freq=p.fs,
        steer=ac.SteeringVector(mics=ac.MicGeom(pos_total=p.mics)),
    )
    DatasetSyntheticConfig._prepare_spectra_wishart(
        ac.MicGeom(pos_total=p.mics),
        spectra,
        p.source_locations,
        p.source_rms**2,
        [1, 2],
        None,
        512000,
    )
    expected = np.diagonal(spectra.Q, axis1=-2, axis2=-1).real[:3]
    np.testing.assert_array_equal(sample['source_strength'], expected)
    feature = AnalyticSourceStrengthFeature(freq_data=spectra, f=[1000.0, 2000.0, 3000.0], num=0)
    analytic = feature.get_feature_func()(sampler={}, data={'prms_sq': p.source_rms**2})
    np.testing.assert_array_equal(sample['source_strength'], analytic['source_strength_analytic'])


def test_current_he_block_size_and_rms_are_read_for_each_case():
    parameters = Parameters(he=4, source_rms=np.array([2.0]), source_locations=LOCATIONS[:, :1])
    dataset = Dataset(config=FixedConfig(parameters=parameters))
    samples = dataset.generate(features=['source_strength'], size=2, progress_bar=False)
    first = next(samples)['source_strength']
    parameters.he = [2, 4, 8]
    parameters.block_size = 128
    parameters.source_rms = np.array([3.0, 4.0])
    parameters.source_locations = LOCATIONS
    second = next(samples)['source_strength']
    np.testing.assert_array_equal(first, [[4 / 513]])
    np.testing.assert_array_equal(second, np.tile(np.array([9, 16]) / 65, (3, 1)))


def test_sampled_he_sources_and_hdf5(tmp_path):
    config = Config()
    config.sample('he', random_func=lambda rng: np.arange(1, rng.integers(2, 5), dtype=float))
    dataset = Dataset(config=config)
    features = ['source_strength', 'x', 'focus_grid', 'source_rms', 'source_locations', 'he']
    expected = list(dataset.generate(features=features, size=3, progress_bar=False))
    for scene in expected:
        strength = scene['source_strength']
        assert strength.shape == (scene['he'].size, scene['source_locations'].shape[1])
        assert scene['x'].shape == (scene['he'].size, scene['focus_grid'].shape[1])
        np.testing.assert_allclose(scene['x'].sum(axis=1), strength.sum(axis=1))
        np.testing.assert_array_equal(strength, np.tile(scene['source_rms'] ** 2 / 513, (scene['he'].size, 1)))
    path = tmp_path / 'source_strength.h5'
    dataset.save_h5(features=features, size=3, name=str(path), progress_bar=False)
    with h5py.File(path, 'r') as file:
        for index, scene in enumerate(expected):
            for name in ['source_strength', 'x', 'focus_grid']:
                np.testing.assert_array_equal(file[f'{index}/{name}'][()], scene[name])


@pytest.mark.parametrize('parameter_class', [Parameters, MiracleA1Parameters])
def test_x_uses_common_reconstruction_grid_for_synthetic_and_miracle(parameter_class):
    parameters = parameter_class(he=[4, 8], source_rms=np.array([2.0, 3.0]))
    parameters.source_locations = parameters.source_grid[:, [0, -1]]
    dataset = Dataset(config=FixedConfig(parameters=parameters))
    sample = next(dataset.generate(features=['x', 'source_strength'], size=1, progress_bar=False))
    distances = np.linalg.norm(parameters.focus_grid[:, :, None] - parameters.source_locations[:, None, :], axis=0)
    indices = np.argmin(distances, axis=0)
    expected = np.zeros((2, 256))
    for source, index in enumerate(indices):
        expected[:, index] = sample['source_strength'][:, source]
    np.testing.assert_array_equal(parameters.focus_grid, Parameters().focus_grid)
    np.testing.assert_array_equal(sample['x'], expected)


def test_x_matches_acoupipe_targetmap():
    locations = np.array([[0.013, 0.075, 0.105], [0.024, 0.080, -0.047], [0.75, 0.75, 0.75]])
    parameters = Parameters(he=[4, 8], source_locations=locations, source_rms=np.array([2.0, 3.0, 4.0]))
    dataset = Dataset(config=FixedConfig(parameters=parameters))
    sample = next(dataset.generate(features=['x', 'source_strength'], size=1, progress_bar=False))

    def strength(sampler, data):
        return {'strength': data['strength']}

    expected = TargetmapFeature.get_targetmap(
        sampler=None,
        data={'loc': locations, 'strength': sample['source_strength']},
        grid=rect_grid(),
        strength_callable=strength,
        name='targetmap',
    )['targetmap']
    np.testing.assert_array_equal(sample['x'], expected.reshape(2, -1))


def test_x_reads_current_grid():
    grid = np.array([[0, 1], [0, 0], [0.75, 0.75]])
    parameters = Parameters(focus_grid=grid, he=4, source_locations=grid.copy(), source_rms=np.array([2.0, 3.0]))
    dataset = Dataset(config=FixedConfig(parameters=parameters))
    samples = dataset.generate(features=['x'], size=2, progress_bar=False)
    first = next(samples)['x']
    parameters.focus_grid = parameters.focus_grid[:, ::-1].copy()
    second = next(samples)['x']
    np.testing.assert_array_equal(first, np.array([[4, 9]]) / 513)
    np.testing.assert_array_equal(second, first[:, ::-1])
