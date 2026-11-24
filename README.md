# ucmf_data

Generation of CMF training and test datasets for ucmf, based on
[acoupipe](https://github.com/adku1173/acoupipe) (branch `analysis/ray-data`)
and Acoular. This repository only writes datasets; loading them for training
is the responsibility of the consumer.

## Installation

```bash
uv sync
```

## Usage

```python
from acoupipe.datasets import Dataset

from ucmf_data import Config, MiracleA1Parameters, Parameters

config = Config(parameters=Parameters(he=[4.0, 8.0]))  # or MiracleA1Parameters(...)
dataset = Dataset(config=config, seed=0)
for scene in dataset.generate(features=['x', 'y', 'A', 'cleansc'], size=3, split='training'):
    ...
```

`Parameters` holds the simulation state (array, focus grid, Welch protocol,
SNR, Helmholtz numbers); `Config` registers the scene samplers and provides the
features below. Only requested features are computed; shared dependencies are
computed once per scene.

| Feature | Shape | Meaning |
| --- | --- | --- |
| `source_strength` | (F, J) | Source powers referenced to the nominal reference microphone |
| `x` | (F, N) | Source powers on the focus grid |
| `csm` | (F, M, M) | Finite-snapshot (Wishart) CSM including noise |
| `y` | (F, M_eff) | Frobenius-isometric real vectorization of `csm` |
| `A` | (F, M_eff, N) | Nominal analytic sensing matrix, `y ≈ A x` |
| `cleansc` | (F, N) | CLEAN-SC map of `csm` |

`Config.method_feature(name, calculate)` wraps further reconstruction methods
with timing and Herold level-error metrics.

`MiracleA1Parameters` generates measured-transfer data from the MIRACLE A1
impulse responses (downloaded and cached through `irdl`), while `A` stays the
analytic model at the measured FFT bins.

## Writing the band [4, 16] dataset

```bash
uv run python -m ucmf_data.store_training_xy
```

Writes `meta.npz` (shared `A`, `he`, frequencies and settings), the
train/validation splits (`x`, `y` as float32, sharded Parquet) and `test.h5`
(`x`, `y`, `cleansc`). Output path and sizes are set at the top of
`src/ucmf_data/store_training_xy.py`.

## Layout

- `parameters.py`, `samplers.py`, `config.py`: simulation state, scene sampling, acoupipe configuration
- `features.py`, `methods.py`: feature calculations and method evaluation
- `geometry.py`, `vectorization.py`, `wishart.py`, `miracle.py`: array/grid geometry, CSM vectorization,
  Wishart sampling and MIRACLE file access
