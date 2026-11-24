"""Simulation states for synthetic data and MIRACLE; sampling and processing stay separate."""

from acoupipe.datasets import ParameterSet

from ucmf_data.geometry import aperture, closest_microphone_position, focus_grid, vogel_subarray
from ucmf_data.miracle import miracle_path, receiver_positions, source_positions

import h5py
import numpy as np


def default_focus_grid(num_mics):
    """Return the focus grid for a num_mics array at z=0.75 m.

    The reduced 16-channel array uses the 16 x 16 grid of experiments 000 and
    001 (256 points). The full 64-channel array uses a 51 x 51 grid as in
    Herold (2601 points).
    """
    return focus_grid(vogel_subarray(num_mics), n_grid=51 if num_mics == 64 else 16, z=0.75)


class Parameters(ParameterSet):
    """Simulation state with shared experiment defaults, overridable by keyword.

    Source locations are coordinates in metres, not grid indices; locations
    and RMS values remain unset until sampling is configured. num_mics selects
    the array variant, 16 for the reduced default and 64 for the full Vogel
    array; the microphone geometry is built from it. focus_grid is the
    reconstruction grid, defaulting to default_focus_grid(num_mics). source_grid
    defaults to a coordinate-identical copy of focus_grid; None selects
    continuous sources.
    nsources starts as the mean source count and holds the realised count after
    its sampler runs. nominal_mics is the model geometry of A; mics are
    initially nominal and may be perturbed per case. ref is the nominal microphone closest
    to the array centre and is never perturbed. r_diag selects the vectorization of y
    and A. aperture is the nominal
    aperture that maps he to frequency. source_rms describes broadband
    reference RMS, and he stores the intended Helmholtz numbers. block_size,
    overlap and window describe the Welch protocol in acoupipe notation;
    signal_seed seeds the finite-snapshot realisation. metric_radius is the
    Herold integration radius in metres, initially 0.05 times nominal aperture.
    """

    def __init__(self, **parameters):
        num_mics = parameters.get('num_mics', 16)
        mics = vogel_subarray(num_mics)
        defaults = {
            'c': 343.0,
            'fs': 32000.0,
            'block_size': 1024,
            'overlap': '50%',
            'window': 'Hanning',
            'signal_length': 16.0,
            'signal_seed': None,
            'num_mics': num_mics,
            'nominal_mics': mics.copy(),
            'mics': mics.copy(),
            'r_diag': False,
            'mic_position_std': 0.005,
            'ref': closest_microphone_position(mics).copy(),
            'aperture': aperture(mics),
            'focus_grid': default_focus_grid(num_mics),
            'nsources': 4,
            'source_locations': None,
            'source_rms': None,
            'snr_db': 20.0,
            'he': 4.0,
        }
        values = defaults | parameters
        values.setdefault('source_grid', values['focus_grid'].copy())
        values.setdefault('metric_radius', 0.05 * values['aperture'])
        super().__init__(**values)


class MiracleA1Parameters(Parameters):
    """Parameters for MIRACLE A1 as in experiment 007.

    mics are the num_mics measured receivers closest to the array centre, ordered
    by radial distance; num_mics defaults to 16 and may select the full 64-channel
    array. source_grid holds the measured A1 source positions inside the x/y
    extent of the focus grid. The reconstruction grid is the same
    default_focus_grid(num_mics) as for synthetic data. c stays at the 343 m/s reference;
    ref, aperture and nominal_mics of A stay nominal; mic_position_std defaults to zero.
    scenario selects the measured SRIR file.
    """

    def __init__(self, **parameters):
        num_mics = parameters.get('num_mics', 16)
        with h5py.File(miracle_path('A1'), 'r') as file:
            receivers = receiver_positions(file)
            sources = source_positions(file)
        mics = receivers[:, np.argsort(np.linalg.norm(receivers[:2], axis=0))[:num_mics]]
        focus = default_focus_grid(num_mics)
        tol = 1e-12
        inside = np.all(
            (sources[:2] >= focus[:2].min(axis=1, keepdims=True) - tol)
            & (sources[:2] <= focus[:2].max(axis=1, keepdims=True) + tol),
            axis=0,
        )
        defaults = {'scenario': 'A1', 'mics': mics, 'mic_position_std': 0.0, 'source_grid': sources[:, inside]}
        super().__init__(**(defaults | parameters))
