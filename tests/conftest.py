"""Pytest path setup for the example-local test suite.

These tests are host-only (no OAK device needed). They import ``core`` from
the example root, which is not itself a package, so the example root is put on
``sys.path`` here.

Note: the repository-wide test suite lives in ``<repo>/tests`` and requires an
explicit ``--root-dir`` option; it does not collect these tests.
"""

import sys
from pathlib import Path

EXAMPLE_ROOT = Path(__file__).resolve().parents[1]
if str(EXAMPLE_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_ROOT))
