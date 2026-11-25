# ucmf_data

Generation of CMF training and test datasets for ucmf, based on
[acoupipe](https://github.com/adku1173/acoupipe) (branch `analysis/ray-data`)
and Acoular. `ucmf_data.stored` loads the written datasets as NumPy arrays;
ucmf itself only receives tensors.

## Installation

```bash
uv sync
```

ucmf is a dependency from the sibling checkout `../ucmf` (editable path source): the CSM
vectorization and the sensing matrix come from ucmf (`HermitianVectorization`,
`CMFDictionary`), so data and solvers share one convention. Ray would restart its workers
with `uv run` in a copied working directory, where `../ucmf` does not resolve; scripts that
use Ray set `RAY_ENABLE_UV_RUN_RUNTIME_ENV=0` before importing it (as `store_training_xy`
and `tests/conftest.py` do).

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

## Loading a band for training

```python
from ucmf_data import stored

H = stored.load_bank(root, 4.0, 16.0)  # (K, M, N_grid): one dictionary per frequency of the band
train = stored.load_band(root, 4.0, 16.0, size=100_000, seed=0, split='training')
# train['y'], train['x']: rows grouped by frequency, rows per frequency drawn with weight 1/He
# train['index']: bank entry of every row, i.e. its frequency's position in stored.band_he
```

`load_bank(..., feature='A')` gives the dense sensing matrices instead. The files hold the
full vectorization; `r_diag=True` in `load_band`, `load_frequency` and `load_bank(..., 'A')`
drops the CSM diagonal rows of `y` and `A` (`vectorization.kept_rows`). `H` does not depend
on it. In ucmf, pass `H` as `operator_kwargs={'H': H, 'r_diag': r_diag}` with the same
`r_diag`, and `index` as `index_train`/`index_val` (see ucmf's ADR 0011).

## Layout

- `parameters.py`, `samplers.py`, `config.py`: simulation state, scene sampling, acoupipe configuration
- `features.py`, `methods.py`: feature calculations and method evaluation
- `geometry.py`, `vectorization.py`, `wishart.py`, `miracle.py`: array/grid geometry, CSM vectorization,
  Wishart sampling and MIRACLE file access
