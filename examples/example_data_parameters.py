"""Generate default features, save three scenes to HDF5 and print their contents.

Run with: uv run python examples/example_data_parameters.py
"""

# ruff: noqa: T201

from pathlib import Path

from acoupipe.datasets import Dataset

from ucmf_data import Config, Parameters

import h5py
import numpy as np

np.set_printoptions(precision=4, threshold=12)
config = Config(parameters=Parameters(he=[4.0, 8.0]))
# Optional: one log-uniform Helmholtz number per scene instead.
# config.sample('he', random_func=lambda rng: np.exp(rng.uniform(np.log(4.0), np.log(8.0))))
dataset = Dataset(config=config)
parameter_features = ['he', 'nsources', 'c', 'fs', 'block_size', 'signal_length', 'snr_db', 'mic_position_std']
features = ['signal_seed', 'mics', 'focus_grid', 'source_locations', 'source_rms', 'source_strength', 'x', 'csm']
features += ['cleansc']
path = Path(__file__).with_suffix('.h5')
dataset.save_h5(features=[*parameter_features, *features], size=3, name=str(path), progress_bar=False)
print(f'HDF5: {path} (overwritten on each run)')


def print_dataset(name, item):
    if isinstance(item, h5py.Dataset):
        print(f'\n{name}, shape={item.shape}:\n{item[()]}')


with h5py.File(path, 'r') as file:
    file.visititems(print_dataset)
