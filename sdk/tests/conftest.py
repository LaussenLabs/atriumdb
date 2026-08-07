# AtriumDB is a timeseries database software designed to best handle the unique features and
# challenges that arise from clinical waveform data.
#     Copyright (C) 2023  The Hospital for Sick Children
#
#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
#
#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.
#
#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Shared pytest configuration for the AtriumDB test suite.

Responsibilities:

* register the suite's markers (see ``sdk/pyproject.toml`` for the canonical list),
* default ``ATRIUMDB_WFDB_CACHE`` to the in-tree MIT-BIH cache so no invocation needs
  to pass it and no run ever reaches out to PhysioNet,
* provide backend selection (``--backend``) that is honoured both by the real
  ``@pytest.mark.parametrize`` tests and by the legacy ``_test_for_both`` shim,
* expose the ``backend`` fixture used by parametrized tests.

See ``sdk/tests/README.md`` for the documented invocations.
"""

import os
import re
from pathlib import Path

import dotenv
import pytest

from tests import testing_framework as _testing_framework  # noqa: E402
from tests.testing_framework import (  # noqa: E402
    ALL_BACKENDS,
    maria_db_available,
    parametrized_backends,
    prepare_backend,
    set_enabled_backends,
)

MARIA_DB_ENV_VARS = ("MARIA_DB_HOST", "MARIA_DB_USER", "MARIA_DB_PASSWORD", "MARIA_DB_PORT")

TESTS_DIR = Path(__file__).parent
DEFAULT_WFDB_CACHE = TESTS_DIR / "wfdb_data"
TEST_DATASETS_DIR = TESTS_DIR / "test_datasets"


def pytest_addoption(parser):
    parser.addoption(
        "--backend",
        action="store",
        default="both",
        choices=["both", "sqlite", "mariadb"],
        help="Which metadata backend(s) to exercise. Default 'both'. "
             "'sqlite' is the fast developer inner loop.",
    )


def pytest_configure(config):
    # Markers are declared in sdk/pyproject.toml too (so `-m` never warns even if this
    # conftest is not collected); registering them here keeps the two in step for
    # invocations that override the ini file.
    for marker in (
        "slow: takes more than ~30s",
        "mariadb: requires a running MariaDB",
        "network: requires network access (there should be none)",
        "mitbih: requires the MIT-BIH wfdb cache",
        "numeric_gate: protected numeric regression test - do not shrink its data",
        "nightly: full-fidelity gate; excluded from the default run, scheduled nightly",
    ):
        config.addinivalue_line("markers", marker)

    # Point the wfdb helpers at the in-tree cache unless the caller redirected it.
    # generate_wfdb re-reads this on every call, so setting it here is sufficient.
    os.environ.setdefault("ATRIUMDB_WFDB_CACHE", str(DEFAULT_WFDB_CACHE))

    # Backend selection. An explicit --backend wins; otherwise a `-m` expression that
    # deselects the mariadb marker (the fast loop) also switches the `_test_for_both`
    # shim to SQLite-only, so `-m "not mariadb"` means the same thing everywhere.
    requested = config.getoption("--backend")
    if requested != "both":
        set_enabled_backends([requested])
    else:
        markexpr = config.option.markexpr or ""
        if re.search(r"\bnot\s+mariadb\b", markexpr):
            set_enabled_backends(["sqlite"])
        else:
            set_enabled_backends(ALL_BACKENDS)

    if "mariadb" not in _testing_framework.get_enabled_backends():
        _disable_mariadb_env()


def _disable_mariadb_env():
    """SQLite-only means SQLite-only, even if a `.env` says otherwise.

    A `.env` at the repo root is visible inside the test container and usually points
    MARIA_DB_HOST at 127.0.0.1, which is not reachable from inside that container. Tests
    that read the variables directly (rather than going through `testing_framework`)
    then try to connect and *error* where they meant to skip. When the run has been
    asked for SQLite only, drop the variables and stop `load_dotenv()` from putting them
    back, so every test sees an unconfigured MariaDB and skips cleanly.

    This only ever runs when the caller explicitly selected SQLite
    (`--backend sqlite` or `-m "not mariadb"`); the default run is untouched.
    """
    for key in MARIA_DB_ENV_VARS:
        os.environ.pop(key, None)

    def _no_dotenv(*args, **kwargs):
        return False

    # Patch the module attribute (for modules imported later during collection) and the
    # already-bound reference in testing_framework, which conftest imported above.
    dotenv.load_dotenv = _no_dotenv
    _testing_framework.load_dotenv = _no_dotenv


@pytest.fixture(scope="session")
def wfdb_cache_dir():
    """Directory holding the cached MIT-BIH records."""
    return Path(os.environ.get("ATRIUMDB_WFDB_CACHE", DEFAULT_WFDB_CACHE))


@pytest.fixture(scope="session")
def mariadb_available():
    return maria_db_available()


@pytest.fixture(params=parametrized_backends())
def backend(request):
    """The metadata backend under test: ``'sqlite'`` or ``'mariadb'``.

    Gives each backend its own test id and lets it be selected with ``-k``/``-m``.
    Skips cleanly when MariaDB is unconfigured or deselected.
    """
    return request.param


@pytest.fixture
def dataset_for_backend(backend):
    """Factory: ``db_name -> (db_type, dataset_location, connection_params)``.

    The returned triple is exactly what ``_test_for_both`` hands a test body, so a
    ``_test_for_both`` call site converts to parametrization without touching the body.
    """
    def _make(db_name):
        return prepare_backend(db_name, backend)

    return _make
