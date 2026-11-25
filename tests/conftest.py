"""Test setup; runs before any test module imports Ray (through acoupipe)."""

import os

# Ray workers use the test interpreter instead of re-running `uv run` in a copied working
# directory, where the relative path source ../ucmf does not resolve.
os.environ.setdefault('RAY_ENABLE_UV_RUN_RUNTIME_ENV', '0')
