"""acoupipe-based CMF dataset generation.

Config and the parameter classes import acoupipe and Acoular; they load on first access, so
that ``ucmf_data.stored`` (loading written datasets) does not import Acoular. Acoular must be
imported before NumPy, which a training script has usually loaded already.
"""

from importlib import import_module

_LAZY = {
    'Config': 'ucmf_data.config',
    'MiracleA1Parameters': 'ucmf_data.parameters',
    'Parameters': 'ucmf_data.parameters',
}

__all__ = [
    'Config',
    'MiracleA1Parameters',
    'Parameters',
]


def __getattr__(name):
    """Import Config and the parameter classes on first access."""
    if name not in _LAZY:
        raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
    return getattr(import_module(_LAZY[name]), name)
