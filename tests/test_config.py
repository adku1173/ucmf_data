"""FA2026 Config defaults, feature selection, dependencies and HDF5 output."""

from acoupipe.datasets import Config as AcoupipeConfig
from acoupipe.datasets import Dataset
from acoupipe.datasets.features import create_feature

from ucmf_data import Config, Parameters
from ucmf_data import config as config_module

import h5py
import numpy as np
import pytest


class FixedConfig(Config):
    def create_samplers(self):
        pass


def fixed_config():
    parameters = Parameters(he=[4, 8], source_rms=np.array([2.0, 3.0]), signal_seed=7)
    parameters.source_locations = parameters.focus_grid[:, [0, -1]]
    return FixedConfig(parameters=parameters)


def sample(config, features):
    return next(Dataset(config=config).generate(features=features, size=1, progress_bar=False))


def test_default_samplers_are_registered_once_and_ready_to_generate():
    config = Config()
    assert isinstance(config, AcoupipeConfig)
    assert isinstance(config.parameters, Parameters)
    samplers = config.get_sampler()
    assert [sampler.attribute for sampler in samplers.values()] == [
        'nsources',
        'source_locations',
        'source_rms',
        'mics',
        'signal_seed',
    ]
    result = sample(config, ['x', 'csm'])
    assert set(result) == {'idx', 'x', 'csm'}
    assert result['x'].shape == (1, 256)
    assert result['csm'].shape == (1, 16, 16)
    assert all(config.get_sampler()[key] is sampler for key, sampler in samplers.items())
    assert Config().parameters is not config.parameters


def test_fixed_scene_can_disable_default_samplers():
    config = fixed_config()
    assert config.get_sampler() == {}
    result = sample(config, ['source_strength'])
    np.testing.assert_array_equal(result['source_strength'], np.tile(np.array([4, 9]) / 513, (2, 1)))


@pytest.mark.parametrize(
    'features',
    [
        ['source_strength'],
        ['x'],
        ['csm'],
        ['csm', 'x', 'source_strength'],
        ['x', 'source_strength', 'csm'],
    ],
)
def test_only_requested_features_are_emitted(features):
    config = fixed_config()
    result = sample(config, features)
    assert set(result) == {'idx', *features}
    _, metadata = config.get_features(features, f=None, num=0)
    assert [feature.name for feature in metadata] == features
    if 'x' in result:
        assert result['x'].shape == (2, 256)
        np.testing.assert_allclose(result['x'].sum(axis=1), [13 / 513, 13 / 513])
    if 'csm' in result:
        assert result['csm'].shape == (2, 16, 16)


def test_source_strength_does_not_evaluate_x_or_csm():
    parameters = Parameters(source_rms=np.array([2.0, 3.0]))
    parameters.focus_grid = None
    assert parameters.source_locations is None
    assert parameters.signal_seed is None
    result = sample(FixedConfig(parameters=parameters), ['source_strength'])
    np.testing.assert_array_equal(result['source_strength'], np.array([[4, 9]]) / 513)
    assert set(result) == {'idx', 'source_strength'}


def test_parameter_only_output_uses_sampled_count():
    config = Config()
    result = sample(config, ['he', 'nsources'])
    assert set(result) == {'idx', 'he', 'nsources'}
    assert result['he'] == 4
    assert result['nsources'] == config.parameters.source_locations.shape[1]


def test_dependency_is_calculated_once_per_case(monkeypatch):
    calls = []
    original = config_module.source_strength

    def source_strength(parameters, data):
        calls.append(parameters.he)
        return original(parameters, data)

    monkeypatch.setattr(config_module, 'source_strength', source_strength)
    config = fixed_config()
    results = list(
        Dataset(config=config).generate(
            features=['csm', 'x', 'source_strength'],
            size=2,
            progress_bar=False,
        )
    )
    assert len(calls) == 2
    for result in results:
        np.testing.assert_array_equal(result['x'].sum(axis=1), result['source_strength'].sum(axis=1))


def test_hdf5_contains_only_requested_output(tmp_path):
    config = Config()
    expected = sample(config, ['csm'])
    path = tmp_path / 'csm.h5'
    Dataset(config=config).save_h5(features=['csm'], size=1, name=str(path), progress_bar=False)
    with h5py.File(path, 'r') as file:
        assert set(file['0']) == {'idx', 'csm'}
        np.testing.assert_array_equal(file['0/csm'][()], expected['csm'])


def test_custom_feature_and_parameter_output():
    def calculate(sampler):
        return {'custom': 3.0}

    custom = create_feature(calculate, name='custom', dtype=np.float64, shape=())
    result = sample(Config(), [custom, 'he'])
    assert set(result) == {'idx', 'custom', 'he'}
    assert result['custom'] == 3.0


def test_frequency_selection_stays_in_parameters():
    config = fixed_config()
    result = next(
        Dataset(config=config).generate(
            features=['source_strength'],
            f=1000,
            num=3,
            size=1,
            progress_bar=False,
        )
    )
    np.testing.assert_array_equal(result['source_strength'], np.tile(np.array([4, 9]) / 513, (2, 1)))


def test_unknown_feature_is_rejected():
    with pytest.raises(ValueError, match='Unknown feature'):
        sample(Config(), ['unknown'])
