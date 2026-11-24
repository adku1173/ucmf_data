"""Register the default scene samplers used by ucmf_data.Config."""

from ucmf_data.geometry import aperture as array_aperture

import numpy as np
from scipy.spatial import cKDTree

SOURCE_POWER_RAYLEIGH_SCALE = 5.0


def attach_samplers(config):
    """Register scene samplers once via Config.create_samplers and return the Config.

    The initial parameters.nsources sets the fixed mean of 1 + Poisson(mean-1).
    A dedicated nsources sampler stores the realised count before locations are
    sampled. source_locations only reads that count. Planar proposals follow centred
    Gaussians with std 0.1688 * nominal aperture, rejected outside the nominal
    focus square. Without parameters.source_grid, sources are continuous at
    z=0.75 m. Otherwise each proposal snaps to its nearest source_grid point in
    x/y. Each accepted location must map to an unoccupied reconstruction-grid
    point; only colliding proposals are redrawn. Counts are conditioned on the
    number of available reconstruction cells (reachable from source_grid when
    discrete). Previously accepted locations are never moved.
    source_rms**2 are Rayleigh(scale=5) broadband reference powers.
    Microphone x/y errors are always relative to parameters.nominal_mics;
    with initial parameters.mic_position_std=0 (the MIRACLE default)
    no microphone sampler is registered. z and all other parameters stay fixed.
    signal_seed seeds the finite-snapshot realisation of each scene.
    """
    aperture = array_aperture(config.parameters.mics)
    mean_nsources = config.parameters.nsources

    def nsources(rng, parameters):
        grid = parameters.source_grid
        capacity = parameters.focus_grid.shape[1]
        if grid is not None:
            capacity = np.unique(cKDTree(parameters.focus_grid.T).query(grid.T)[1]).size
        count = 1 + rng.poisson(mean_nsources - 1)
        while count > capacity:
            count = 1 + rng.poisson(mean_nsources - 1)
        return count

    def focus_positions(rng, count):
        xy = rng.normal(0.0, 0.1688 * aperture, size=(2, count))
        outside = np.any(np.abs(xy) > aperture / 2, axis=0)
        while np.any(outside):
            xy[:, outside] = rng.normal(0.0, 0.1688 * aperture, size=(2, np.count_nonzero(outside)))
            outside = np.any(np.abs(xy) > aperture / 2, axis=0)
        return xy

    def source_locations(rng, parameters):
        grid = parameters.source_grid
        reconstruction = cKDTree(parameters.focus_grid.T)
        count = parameters.nsources
        locations, occupied = [], set()
        while len(locations) < count:
            xy = focus_positions(rng, count - len(locations))
            candidates = np.vstack((xy, np.full(xy.shape[1], 0.75)))
            if grid is not None:
                distances = np.sum((grid[:2, :, None] - xy[:, None, :]) ** 2, axis=0)
                candidates = grid[:, np.argmin(distances, axis=0)]
            cells = reconstruction.query(candidates.T)[1]
            for location, cell in zip(candidates.T, cells, strict=True):
                if cell not in occupied:
                    locations.append(location)
                    occupied.add(cell)
        return np.asarray(locations).T

    def source_rms(rng, parameters):
        return np.sqrt(rng.rayleigh(SOURCE_POWER_RAYLEIGH_SCALE, size=parameters.nsources))

    def microphone_positions(rng, parameters):
        mics = parameters.nominal_mics.copy()
        mics[:2] += rng.normal(0.0, parameters.mic_position_std, size=mics[:2].shape)
        return mics

    config.sample('nsources', random_func=nsources)
    config.sample('source_locations', random_func=source_locations)
    config.sample('source_rms', random_func=source_rms)
    if config.parameters.mic_position_std > 0:
        config.sample('mics', random_func=microphone_positions)
    config.sample('signal_seed', random_func=lambda rng: int(rng.integers(2**32)))
    return config
