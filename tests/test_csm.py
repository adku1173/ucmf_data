"""Finite-snapshot CSMs: expectation, reference level, reproducibility and shared dependencies."""

from acoupipe.datasets import Dataset

from ucmf_data import Config, MiracleA1Parameters, Parameters
from ucmf_data import features as feature_module
from ucmf_data.wishart import sample_complex_wishart_covariance

import numpy as np
import pytest

FEATURES = ['csm', 'source_strength', 'source_rms', 'source_locations', 'mics', 'signal_seed']


def scenes(parameters, size):
    config = Config(parameters=parameters)
    return list(Dataset(config=config).generate(features=FEATURES, size=size, progress_bar=False))


def test_wishart_samples_have_the_scale_as_expectation():
    rng = np.random.default_rng(0)
    h = rng.normal(size=(3, 2)) + 1j * rng.normal(size=(3, 2))
    scale = h @ h.conj().T + 0.5 * np.eye(3)
    draws = np.array([sample_complex_wishart_covariance(scale, 10, rng) for _ in range(20000)])
    np.testing.assert_allclose(draws.mean(axis=0), scale, atol=0.05 * np.abs(scale).max())
    np.testing.assert_allclose(draws, draws.conj().swapaxes(1, 2))


@pytest.mark.parametrize('parameter_class', [Parameters, MiracleA1Parameters])
@pytest.mark.parametrize('snr_db', [20.0, None])
def test_csm_expectation_is_referenced_signal_plus_white_noise(parameter_class, snr_db, monkeypatch):
    monkeypatch.setattr(feature_module, 'equivalent_snapshot_count', lambda *args: 1e12)
    parameters = parameter_class(he=[4.0, 8.0], snr_db=snr_db, source_rms=np.array([1.0, 2.0]), signal_seed=7)
    parameters.source_locations = parameters.source_grid[:, [0, -1]]
    miracle = parameter_class is MiracleA1Parameters
    transfer = feature_module.measured_transfer if miracle else feature_module.analytic_transfer
    data = {}
    data['transfer'], data['reference_gain'] = transfer(parameters)
    data['source_strength'] = feature_module.source_strength(parameters, data)
    ref = np.argmin(np.linalg.norm(parameters.mics - parameters.ref[:, None], axis=0))
    np.testing.assert_allclose(np.abs(data['transfer'][:, ref]), 1.0, rtol=1e-12)
    result = feature_module.csm(parameters, data)
    for matrix, h, strength in zip(result, data['transfer'], data['source_strength'], strict=True):
        expected = (h * strength) @ h.conj().T
        if snr_db is not None:
            expected += strength.sum() / 10 ** (snr_db / 10) * np.eye(h.shape[0])
        np.testing.assert_allclose(matrix, expected, atol=1e-4 * np.abs(expected).max())


def test_csm_is_reproducible_and_reuses_source_strength():
    first, repeated = scenes(Parameters(he=[4.0, 8.0]), 2), scenes(Parameters(he=[4.0, 8.0]), 2)
    for left, right in zip(first, repeated, strict=True):
        np.testing.assert_array_equal(left['csm'], right['csm'])
        np.testing.assert_allclose(left['csm'], left['csm'].conj().swapaxes(1, 2))
    assert not np.array_equal(first[0]['csm'], first[1]['csm'])


def test_measured_transfer_is_read_once_per_case(monkeypatch):
    from ucmf_data import config as config_module

    calls = []
    original = config_module.measured_transfer

    def measured_transfer(parameters):
        calls.append(parameters.source_locations)
        return original(parameters)

    monkeypatch.setattr(config_module, 'measured_transfer', measured_transfer)
    config = Config(parameters=MiracleA1Parameters(he=[4.0, 8.0]))
    features = ['source_strength', 'x', 'csm']
    results = list(Dataset(config=config).generate(features=features, size=2, progress_bar=False))
    assert len(calls) == 2
    for scene in results:
        assert set(scene) == {'idx', *features}
        np.testing.assert_allclose(scene['x'].sum(axis=1), scene['source_strength'].sum(axis=1))
