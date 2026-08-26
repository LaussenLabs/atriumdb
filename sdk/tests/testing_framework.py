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

import os
import shutil
import warnings
from pathlib import Path
from dotenv import load_dotenv
import numpy as np
import pytest

from atriumdb import AtriumSDK
from atriumdb.sql_handler.maria.maria_handler import MariaDBHandler


DEFAULT_MARIA_DB_PORT = 3306

#: The backends the current pytest invocation should exercise.  Both are enabled by
#: default; ``sdk/tests/conftest.py`` narrows this when ``--backend`` is given or when
#: the ``-m`` expression deselects the ``mariadb`` marker.
#: This exists so that ``_test_for_both`` -- which is a plain function and therefore
#: invisible to pytest's marker machinery -- obeys the same selection as the properly
#: parametrized tests.
ALL_BACKENDS = ("mariadb", "sqlite")
_ENABLED_BACKENDS = list(ALL_BACKENDS)


def set_enabled_backends(backends):
    """Restrict which backends `_test_for_both` / `parametrized_backends` will run."""
    global _ENABLED_BACKENDS
    _ENABLED_BACKENDS = [backend for backend in ALL_BACKENDS if backend in set(backends)]


def get_enabled_backends():
    return list(_ENABLED_BACKENDS)


def is_backend_enabled(db_type):
    return db_type in _ENABLED_BACKENDS


def maria_db_credentials():
    """Return the MariaDB connection settings from the environment, or None.

    Returns None (rather than raising) when the MariaDB variables are absent so that a
    run without credentials degrades to SQLite-only instead of erroring out.
    """
    load_dotenv()

    host = os.getenv("MARIA_DB_HOST")
    user = os.getenv("MARIA_DB_USER")
    password = os.getenv("MARIA_DB_PASSWORD")
    raw_port = os.getenv("MARIA_DB_PORT")

    if not host or not user:
        return None

    try:
        port = int(raw_port) if raw_port else DEFAULT_MARIA_DB_PORT
    except ValueError:
        raise ValueError(
            f"MARIA_DB_PORT must be an integer, got {raw_port!r}. "
            f"Check your .env file or unset MARIA_DB_HOST to run SQLite-only."
        )

    return {"host": host, "user": user, "password": password, "port": port}


def maria_db_available():
    """True when MariaDB credentials are configured for this run."""
    return maria_db_credentials() is not None


def parametrized_backends():
    """``argvalues`` for a real ``@pytest.mark.parametrize`` over both backends.

    Using this instead of ``_test_for_both`` gives each backend its own test id
    (``[sqlite]`` / ``[mariadb]``), lets ``-k sqlite`` / ``-m "not mariadb"`` select
    them, reports their failures independently and makes ``--durations`` meaningful.
    Coverage is unchanged: both backends still run.
    """
    return [
        pytest.param("mariadb", id="mariadb", marks=pytest.mark.mariadb),
        pytest.param("sqlite", id="sqlite"),
    ]


def prepare_backend(db_name, db_type):
    """Create a clean dataset location (and MariaDB database) for `db_type`.

    Returns ``(db_type, dataset_location, connection_params)`` -- the exact triple
    ``_test_for_both`` passes to a test body, so a parametrized test can be written as
    a drop-in replacement.  Skips (rather than errors) when the requested backend is
    unavailable or has been deselected for this run.
    """
    if not is_backend_enabled(db_type):
        pytest.skip(f"backend {db_type!r} is not enabled for this run")

    if db_type in ('mysql', 'mariadb'):
        credentials = maria_db_credentials()
        if credentials is None:
            pytest.skip("MariaDB is not configured (set MARIA_DB_HOST / MARIA_DB_USER)")

        dataset_path = Path(__file__).parent / "test_datasets" / f"maria_{db_name}"
        shutil.rmtree(dataset_path, ignore_errors=True)
        maria_handler = MariaDBHandler(
            credentials["host"], credentials["user"], credentials["password"], db_name, credentials["port"])
        maria_handler.maria_connect_no_db().cursor().execute(f"DROP DATABASE IF EXISTS `{db_name}`")
        connection_params = {
            'sqltype': 'mariadb',
            'host': credentials["host"],
            'user': credentials["user"],
            'password': credentials["password"],
            'database': db_name,
            'port': credentials["port"]}
        return 'mariadb', dataset_path, connection_params

    dataset_path = Path(__file__).parent / "test_datasets" / f"sqlite_{db_name}"
    shutil.rmtree(dataset_path, ignore_errors=True)
    dataset_path.unlink(missing_ok=True)
    return 'sqlite', dataset_path, None


def _test_for_both(db_name, test_function, *args):
    """Run `test_function` against both backends.

    Retained as a working shim so files can be converted to real parametrization
    (see `parametrized_backends`) incrementally.  Unlike the original it skips the
    MariaDB half cleanly when credentials are absent instead of raising TypeError,
    and it honours the run's backend selection.
    """
    credentials = maria_db_credentials()

    maria_dataset_path = Path(__file__).parent / "test_datasets" / f"maria_{db_name}"
    sqlite_dataset_path = Path(__file__).parent / "test_datasets" / f"sqlite_{db_name}"

    ran_any = False

    if not is_backend_enabled('mariadb'):
        pass
    elif credentials is None:
        warnings.warn(
            "MariaDB credentials are not configured (MARIA_DB_HOST / MARIA_DB_USER); "
            "skipping the MariaDB half of this test and running SQLite only. "
            "Set them in a .env file or the environment to exercise both backends.",
            RuntimeWarning,
        )
    else:
        host = credentials["host"]
        user = credentials["user"]
        password = credentials["password"]
        port = credentials["port"]

        db_type = 'mariadb'
        shutil.rmtree(maria_dataset_path, ignore_errors=True)
        maria_handler = MariaDBHandler(host, user, password, db_name, port)
        connection_params = {
            'sqltype': db_type,
            'host': host,
            'user': user,
            'password': password,
            'database': db_name,
            'port': port}
        maria_handler.maria_connect_no_db().cursor().execute(f"DROP DATABASE IF EXISTS `{db_name}`")
        test_function(db_type, maria_dataset_path, connection_params, *args)
        ran_any = True

    if is_backend_enabled('sqlite'):
        db_type = 'sqlite'
        connection_params = None
        shutil.rmtree(sqlite_dataset_path, ignore_errors=True)
        sqlite_dataset_path.unlink(missing_ok=True)
        test_function(db_type, sqlite_dataset_path, connection_params, *args)
        ran_any = True

    if not ran_any:
        pytest.skip("no database backend is enabled for this run")


def create_sibling_sdk(connection_params, dataset_location, db_type):
    dataset_location = str(dataset_location) + "_2"
    shutil.rmtree(dataset_location, ignore_errors=True)
    if db_type in ['mysql', 'mariadb']:
        connection_params['database'] += "-2"
        host = connection_params['host']
        user = connection_params['user']
        password = connection_params['password']
        db_name = connection_params['database']
        port = connection_params['port']

        maria_handler = MariaDBHandler(host, user, password, db_name, port)
        maria_handler.maria_connect_no_db().cursor().execute(f"DROP DATABASE IF EXISTS `{db_name}`")
    sdk_2 = AtriumSDK.create_dataset(
        dataset_location=dataset_location, database_type=db_type, connection_params=connection_params)
    return sdk_2


def slice_times_values(times, values, start_time, end_time):
    # Ensure times are unique and sorted
    unique_times, unique_indices = np.unique(times, return_index=True)
    unique_values = values[unique_indices]

    # Find the insertion points for start_time and end_time
    start_idx = np.searchsorted(unique_times, start_time, side='left') if start_time is not None else 0
    end_idx = np.searchsorted(unique_times, end_time, side='left') if end_time is not None else unique_times.shape[0]

    # Slice the arrays based on the indices
    sliced_times = unique_times[start_idx:end_idx]
    sliced_values = unique_values[start_idx:end_idx]

    return sliced_times, sliced_values
