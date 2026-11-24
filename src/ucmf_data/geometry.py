"""Microphone array and focus grid geometry in metres."""

from pathlib import Path

import acoular as ac

import numpy as np


def vogel_subarray(num_mics):
    """Return positions (3, num_mics) of the innermost TUB Vogel 64 channels, by ascending radius."""
    pos = ac.MicGeom(file=Path(ac.__file__).parent / 'xml' / 'tub_vogel64.xml').pos_total
    return pos[:, np.argsort(np.linalg.norm(pos[:2], axis=0))[:num_mics]]


def aperture(mics):
    """Return the aperture in metres of microphone positions (3, M)."""
    return ac.MicGeom(pos_total=mics).aperture


def closest_microphone_position(mics):
    """Return the position (3,) of the microphone closest to the array centre in the xy-plane."""
    return mics[:, np.argmin(np.linalg.norm(mics[:2], axis=0))]


def focus_grid(mics, n_grid, z):
    """Return the square n_grid x n_grid focus grid (3, n_grid**2) spanning the array aperture at z."""
    half = 0.5 * aperture(mics)
    grid = ac.RectGrid(x_min=-half, x_max=half, y_min=-half, y_max=half, z=z, increment=2 * half / (n_grid - 1))
    return np.asarray(grid.pos, dtype=float)
