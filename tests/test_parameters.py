"""Shared ParameterSet defaults and compatibility with the new acoupipe API."""

from acoupipe.datasets import Dataset, ParameterSet

from ucmf_data import Config, MiracleA1Parameters, Parameters
from ucmf_data.geometry import focus_grid, vogel_subarray

import numpy as np


def test_experiment_defaults():
    parameters = Parameters()
    assert Parameters.__bases__ == (ParameterSet,)
    assert parameters.c == 343.0
    assert parameters.fs == 32000.0
    assert parameters.block_size == 1024
    assert parameters.signal_length == 16.0
    assert parameters.snr_db == 20.0
    assert parameters.he == 4.0
    assert parameters.nsources == 4
    assert parameters.mic_position_std == 0.005
    assert 'f' not in parameters.to_dict()
    np.testing.assert_array_equal(parameters.focus_grid, focus_grid(vogel_subarray(16), n_grid=16, z=0.75))
    np.testing.assert_array_equal(parameters.source_grid, parameters.focus_grid)
    assert parameters.source_locations is None
    assert parameters.source_rms is None
    np.testing.assert_array_equal(parameters.mics, vogel_subarray(16))
    np.testing.assert_array_equal(parameters.nominal_mics, parameters.mics)


def test_overrides_and_roundtrip():
    parameters = Parameters(fs=16000.0, block_size=128, snr_db=None, he=[2.0, 4.0], nsources=2, mic_position_std=0.01)
    restored = Parameters.from_dict(parameters.to_dict())
    assert restored.fs == 16000.0
    assert restored.block_size == 128
    assert restored.snr_db is None
    assert restored.he == [2.0, 4.0]
    assert restored.nsources == 2
    assert restored.mic_position_std == 0.01
    np.testing.assert_array_equal(restored.mics, parameters.mics)
    np.testing.assert_array_equal(restored.focus_grid, parameters.focus_grid)
    np.testing.assert_array_equal(restored.source_grid, parameters.source_grid)


def test_custom_grid_defaults_and_explicit_source_grid():
    grid = np.array([[0.0, 0.1], [0.0, 0.0], [0.75, 0.75]])
    parameters = Parameters(focus_grid=grid)
    np.testing.assert_array_equal(parameters.focus_grid, grid)
    np.testing.assert_array_equal(parameters.source_grid, grid)
    assert not np.shares_memory(parameters.focus_grid, parameters.source_grid)
    source_grid = grid[:, :1]
    parameters = Parameters(focus_grid=grid, source_grid=source_grid)
    np.testing.assert_array_equal(parameters.source_grid, source_grid)
    continuous = Parameters(focus_grid=grid, source_grid=None)
    assert continuous.source_grid is None
    np.testing.assert_array_equal(continuous.focus_grid, grid)


def test_independent_geometry():
    first, second = Parameters(), Parameters()
    first.mics[:] = 0
    assert np.any(second.mics)


def test_parameter_output():
    parameters = Parameters()
    dataset = Dataset(config=Config(parameters=parameters))
    sample = next(dataset.generate(features=['fs', 'mics', 'snr_db'], size=1, progress_bar=False))
    assert sample['fs'] == parameters.fs
    assert sample['snr_db'] == parameters.snr_db
    np.testing.assert_array_equal(sample['mics'], parameters.mics)


def test_full_array_defaults():
    parameters = Parameters(num_mics=64)
    assert parameters.mics.shape == (3, 64)
    assert parameters.focus_grid.shape == (3, 51 * 51)
    np.testing.assert_array_equal(parameters.focus_grid, focus_grid(vogel_subarray(64), n_grid=51, z=0.75))


def test_miracle_a1_defaults():
    parameters = MiracleA1Parameters()
    assert isinstance(parameters, Parameters)
    assert parameters.scenario == 'A1'
    assert parameters.c == 343.0
    assert parameters.snr_db == 20.0
    assert parameters.block_size == 1024
    assert parameters.mic_position_std == 0.0
    np.testing.assert_array_equal(parameters.focus_grid, Parameters().focus_grid)
    np.testing.assert_array_equal(parameters.nominal_mics, Parameters().nominal_mics)
    np.testing.assert_array_equal(parameters.ref, Parameters().ref)
    assert parameters.mics.shape == (3, 16)
    radii = np.linalg.norm(parameters.mics[:2], axis=0)
    assert np.all(np.diff(radii) >= 0)
    extent = np.abs(parameters.focus_grid[:2]).max(axis=1, keepdims=True)
    assert parameters.source_grid.shape[1] > 0
    assert np.all(np.abs(parameters.source_grid[:2]) <= extent + 1e-12)
