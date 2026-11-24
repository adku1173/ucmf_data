"""Source-space and finite-snapshot microphone feature calculations."""

from functools import lru_cache

import acoular as ac
from acoular.fbeamform import calcTransfer
from acoupipe.datasets.features.spectra_analytic import PowerSpectraAnalytic

from ucmf_data.miracle import (
    miracle_path,
    receiver_positions,
    snap_frequency_to_fft_bin,
    source_positions,
    transfer_columns_at_bin,
)
from ucmf_data.parameters import MiracleA1Parameters
from ucmf_data.vectorization import vectorize_csm
from ucmf_data.wishart import sample_complex_wishart_covariance

import h5py
import numpy as np
from scipy.spatial import cKDTree


def source_strength(parameters, data):
    """Return source powers at the nominal reference point, shape (F, J).

    Input powers follow acoupipe's flat broadband spectrum, source_rms**2 per
    (block_size/2 + 1) FFT lines, and are weighted by data['reference_gain'].
    """
    powers = np.square(parameters.source_rms, dtype=float) / (parameters.block_size // 2 + 1)
    return data['reference_gain'] * powers


def x(parameters, data):
    """Project source strengths to grid (F, N); samplers guarantee unique cells."""
    strength = data['source_strength']
    indices = cKDTree(parameters.focus_grid.T).query(parameters.source_locations.T)[1]
    target = np.zeros((strength.shape[0], parameters.focus_grid.shape[1]))
    target[:, indices] = strength
    return target


def frequencies(parameters):
    """Return he * c / aperture (F,); MIRACLE snaps to the nearest FFT bins of block_size at fs."""
    exact = np.atleast_1d(parameters.he) * parameters.c / parameters.aperture
    if not isinstance(parameters, MiracleA1Parameters):
        return exact
    return np.array([snap_frequency_to_fft_bin(f, parameters.fs, parameters.block_size)[1] for f in exact])


def sensing_matrix(parameters, data):
    """Return the nominal CMF model A (F, M_eff, N) at frequencies(parameters).

    Columns hold the true-level transfer from grid points to nominal_mics,
    referenced to the nominal ref, in the vectorization convention of y for
    r_diag: column n is the vectorized unit-power CSM of grid point n.
    Independent of the sampled scene.
    """
    grid = parameters.focus_grid
    rm = np.linalg.norm(grid[:, :, None] - parameters.nominal_mics[:, None, :], axis=0)
    r0 = np.linalg.norm(grid - parameters.ref[:, None], axis=0)
    result = []
    for frequency in frequencies(parameters):
        h = calcTransfer(r0, rm, np.array(2 * np.pi * frequency / parameters.c))
        outer = h[:, :, None] * h.conj()[:, None, :]
        result.append(vectorize_csm(outer, parameters.r_diag).T)
    return np.array(result)


def y(parameters, data):
    """Return the real Frobenius-isometric vectorization (F, M_eff) of csm for r_diag."""
    return vectorize_csm(data['csm'], parameters.r_diag)


def analytic_reference_gain(parameters):
    """Return unit gains (F, J); free-field true-level transfer is referenced to ref."""
    return np.ones((np.size(parameters.he), np.size(parameters.source_rms)))


def analytic_transfer(parameters):
    """Return true-level H (F, M, J) at exact He, referenced to the nominal ref."""
    loc = parameters.source_locations
    rm = np.linalg.norm(loc[:, :, None] - parameters.mics[:, None, :], axis=0)
    r0 = np.linalg.norm(loc - parameters.ref[:, None], axis=0)
    wavenumbers = 2 * np.pi * np.atleast_1d(parameters.he) / parameters.aperture
    h = np.array([calcTransfer(r0, rm, np.array(k)).T for k in wavenumbers])
    return h, analytic_reference_gain(parameters)


def measured_transfer(parameters):
    """Return MIRACLE H (F, M, J) at nearest FFT bins with unit reference gain.

    The nominal ref is represented by the closest measured receiver. Each
    source column is divided by its raw magnitude |h_ref| there, so phases stay
    measured; the removed gains |h_ref|**2 (F, J) are returned as well.
    """
    bins = np.rint(frequencies(parameters) * parameters.block_size / parameters.fs).astype(int)
    with h5py.File(miracle_path(parameters.scenario), 'r') as file:
        receivers = cKDTree(receiver_positions(file).T).query(parameters.mics.T)[1]
        sources = cKDTree(source_positions(file).T).query(parameters.source_locations.T)[1]
        h = np.array([transfer_columns_at_bin(file, sources, receivers, parameters.block_size, b) for b in bins])
    ref = np.argmin(np.linalg.norm(parameters.mics - parameters.ref[:, None], axis=0))
    magnitude = np.abs(h[:, ref])
    return h / magnitude[:, None, :], magnitude**2


@lru_cache
def equivalent_snapshot_count(num_samples, fs, block_size, overlap, window):
    """Return acoupipe's Welch-equivalent snapshot count df_eq, cached per Welch protocol."""
    return PowerSpectraAnalytic(
        num_samples=num_samples,
        sample_freq=fs,
        block_size=block_size,
        overlap=overlap,
        window=window,
    ).df_eq


def csm(parameters, data):
    """Draw one joint signal/noise Wishart CSM per frequency, shape (F, M, M).

    data['transfer'] H and data['source_strength'] p are both referenced to the
    nominal ref, so sigma^2 = sum(p) / 10**(snr_db/10). Noiseless draws use
    source space. Acoupipe supplies the Welch-equivalent snapshot count;
    signal_seed seeds the finite-snapshot realisation of this source case.
    """
    df = equivalent_snapshot_count(
        round(parameters.signal_length * parameters.fs),
        parameters.fs,
        parameters.block_size,
        parameters.overlap,
        parameters.window,
    )
    rng = np.random.default_rng(parameters.signal_seed)
    result = []
    for h_f, strength in zip(data['transfer'], data['source_strength'], strict=True):
        if parameters.snr_db is None:
            w = sample_complex_wishart_covariance(np.diag(strength).astype(h_f.dtype), df, rng)
            result.append(h_f @ w @ h_f.conj().T)
            continue
        scale = (h_f * strength) @ h_f.conj().T
        scale = (scale + scale.conj().T) * 0.5
        scale[np.diag_indices_from(scale)] += strength.sum() / 10 ** (parameters.snr_db / 10)
        result.append(sample_complex_wishart_covariance(scale, df, rng))
    return np.array(result)


def cleansc(parameters, data):
    """Return CLEAN-SC maps (F, N) of the existing CSM, in the original He order.

    Use nominal_mics, grid, ref and sound speed from the current ParameterSet.
    Synthetic frequencies are exact; MIRACLE frequencies use the measured FFT
    bins. r_diag comes from the ParameterSet; other CLEAN-SC settings use Acoular defaults.
    """
    values = frequencies(parameters)
    order = np.argsort(values)
    spectra = ac.PowerSpectraImport(csm=data['csm'][order], frequencies=values[order])
    steer = ac.SteeringVector(
        grid=ac.ImportGrid(pos=parameters.focus_grid),
        mics=ac.MicGeom(pos_total=parameters.nominal_mics),
        env=ac.Environment(c=parameters.c),
        ref=parameters.ref,
    )
    beamformer = ac.BeamformerCleansc(freq_data=spectra, steer=steer, r_diag=parameters.r_diag, cached=False)
    return np.asarray(beamformer.result[:])[np.argsort(order)]
