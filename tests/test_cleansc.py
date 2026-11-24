"""CLEAN-SC maps of the existing CSM for every He, without additional draws."""

import acoular as ac
from acoupipe.datasets import Dataset

from ucmf_data import Config, MiracleA1Parameters, Parameters
from ucmf_data import config as config_module
from ucmf_data import features as feature_module

import h5py
import numpy as np
import pytest


def sample(config, features):
    return next(Dataset(config=config).generate(features=features, size=1, progress_bar=False))


def reference(parameters, matrix, frequency):
    spectra = ac.PowerSpectraImport(csm=matrix[None], frequencies=np.array([frequency]))
    steer = ac.SteeringVector(
        grid=ac.ImportGrid(pos=parameters.focus_grid),
        mics=ac.MicGeom(pos_total=parameters.nominal_mics),
        env=ac.Environment(c=parameters.c),
        ref=parameters.ref,
    )
    beamformer = ac.BeamformerCleansc(freq_data=spectra, steer=steer, r_diag=parameters.r_diag, cached=False)
    return np.asarray(beamformer.result[:])[0]


@pytest.mark.parametrize('he', [4.0, [8.0, 4.0], np.array([8.0, 4.0, 8.0])])
@pytest.mark.parametrize('r_diag', [False, True])
def test_all_frequencies_match_acoular_in_original_order(he, r_diag):
    parameters = Parameters(he=he, r_diag=r_diag)
    result = sample(Config(parameters=parameters), ['cleansc', 'csm'])
    frequencies = np.atleast_1d(he) * parameters.c / parameters.aperture
    expected = np.array(
        [reference(parameters, matrix, frequency) for matrix, frequency in zip(result['csm'], frequencies, strict=True)]
    )
    assert result['cleansc'].shape == (np.size(he), parameters.focus_grid.shape[1])
    assert result['cleansc'].dtype == np.float64
    assert np.all(np.isfinite(result['cleansc']))
    np.testing.assert_allclose(result['cleansc'], expected, rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize('parameter_class', [Parameters, MiracleA1Parameters])
def test_power_spectra_import_receives_sorted_csms_and_effective_frequencies(parameter_class, monkeypatch):
    parameters = parameter_class(he=[8.01, 4.03])
    imports = []
    original = ac.PowerSpectraImport

    def spectra_import(**kwargs):
        imports.append(kwargs)
        return original(**kwargs)

    monkeypatch.setattr(feature_module.ac, 'PowerSpectraImport', spectra_import)
    result = sample(Config(parameters=parameters), ['cleansc', 'csm'])
    frequencies = np.asarray(parameters.he) * parameters.c / parameters.aperture
    if parameter_class is MiracleA1Parameters:
        bins = np.rint(frequencies * parameters.block_size / parameters.fs)
        frequencies = bins * parameters.fs / parameters.block_size
    order = np.argsort(frequencies)
    assert len(imports) == 1
    np.testing.assert_array_equal(imports[0]['frequencies'], frequencies[order])
    np.testing.assert_array_equal(imports[0]['csm'], result['csm'][order])
    expected = np.array(
        [reference(parameters, matrix, frequency) for matrix, frequency in zip(result['csm'], frequencies, strict=True)]
    )
    np.testing.assert_allclose(result['cleansc'], expected, rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize('requested', [['cleansc'], ['cleansc', 'csm'], ['csm', 'cleansc']])
def test_csm_dependency_is_calculated_once_per_case(requested, monkeypatch):
    calls = []
    original = config_module.csm

    def csm(parameters, data):
        calls.append(parameters.signal_seed)
        return original(parameters, data)

    monkeypatch.setattr(config_module, 'csm', csm)
    config = Config(parameters=Parameters(he=[4, 8]))
    results = list(Dataset(config=config).generate(features=requested, size=2, progress_bar=False))
    assert len(calls) == 2
    for result in results:
        assert set(result) == {'idx', *requested}
    _, metadata = config.get_features(requested, f=None, num=0)
    assert [feature.name for feature in metadata] == requested


def test_hdf5_contains_reproducible_cleansc_only(tmp_path):
    config = Config(parameters=Parameters(he=[8, 4]))
    expected = sample(config, ['cleansc'])['cleansc']
    path = tmp_path / 'cleansc.h5'
    Dataset(config=config).save_h5(features=['cleansc'], size=1, name=str(path), progress_bar=False)
    with h5py.File(path, 'r') as file:
        assert set(file['0']) == {'idx', 'cleansc'}
        np.testing.assert_array_equal(file['0/cleansc'][()], expected)
