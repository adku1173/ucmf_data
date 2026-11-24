"""Consistency of the sensing matrix A with y, the CSM and the source powers."""

from acoupipe.datasets import Dataset

from ucmf_data import Config, MiracleA1Parameters, Parameters
from ucmf_data.features import analytic_transfer, frequencies, sensing_matrix, source_strength, x
from ucmf_data.vectorization import vectorize_csm

import numpy as np
import pytest

HE = [4.0, 8.0, 16.0]


def sample(parameters, features):
    return next(Dataset(config=Config(parameters=parameters)).generate(features=features, size=1, progress_bar=False))


@pytest.mark.parametrize('r_diag', [False, True])
def test_a_maps_grid_powers_to_the_vectorized_ensemble_csm(r_diag):
    parameters = Parameters(he=HE, r_diag=r_diag, source_rms=np.array([1.0, 2.0, 3.0]))
    parameters.source_locations = parameters.focus_grid[:, [0, 100, 255]]
    data = {}
    data['transfer'], data['reference_gain'] = analytic_transfer(parameters)
    data['source_strength'] = source_strength(parameters, data)
    target = x(parameters, data)
    matrix = sensing_matrix(parameters, data)
    for h, strength, a, column in zip(data['transfer'], data['source_strength'], matrix, target, strict=True):
        expected = vectorize_csm((h * strength) @ h.conj().T, r_diag)
        np.testing.assert_allclose(a @ column, expected, rtol=1e-9, atol=1e-12 * np.abs(expected).max())


@pytest.mark.parametrize('r_diag', [False, True])
def test_synthetic_a_stays_nominal_despite_perturbed_mics(r_diag):
    result = sample(Parameters(he=HE, r_diag=r_diag), ['A', 'mics'])
    assert not np.array_equal(result['mics'], Parameters().nominal_mics)
    np.testing.assert_array_equal(result['A'], sensing_matrix(Parameters(he=HE, r_diag=r_diag), {}))


def test_miracle_a_is_the_nominal_model_at_snapped_fft_bins():
    parameters = MiracleA1Parameters(he=HE)
    snapped = frequencies(parameters)
    np.testing.assert_allclose(snapped / (parameters.fs / parameters.block_size), np.rint(snapped * 1024 / 32000))
    assert not np.allclose(snapped, frequencies(Parameters(he=HE)))
    nominal = Parameters(he=snapped * parameters.aperture / parameters.c)
    np.testing.assert_allclose(sample(parameters, ['A'])['A'], sensing_matrix(nominal, {}), rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize('parameter_class', [Parameters, MiracleA1Parameters])
@pytest.mark.parametrize('r_diag', [False, True])
def test_y_is_frobenius_isometric_to_the_csm(parameter_class, r_diag):
    result = sample(parameter_class(he=HE, r_diag=r_diag), ['y', 'csm', 'A'])
    csm = result['csm']
    if r_diag:
        csm = csm - np.einsum('fii->fi', csm)[..., None] * np.eye(csm.shape[-1])
    np.testing.assert_allclose(np.linalg.norm(result['y'], axis=-1), np.linalg.norm(csm, axis=(-2, -1)), rtol=1e-12)
    assert result['y'].shape == result['A'].shape[:2]


def test_y_alone_emits_only_y_and_reuses_csm():
    alone = sample(Parameters(he=HE), ['y'])
    together = sample(Parameters(he=HE), ['y', 'csm'])
    assert set(alone) == {'idx', 'y'}
    assert alone['y'].shape == (3, 256)
    np.testing.assert_array_equal(alone['y'], together['y'])
    np.testing.assert_array_equal(together['y'], vectorize_csm(together['csm'], False))
