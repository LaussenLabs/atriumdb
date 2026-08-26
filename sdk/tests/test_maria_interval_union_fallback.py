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

"""MariaDB merge-mode interval inserts prefer the insert_interval_union stored
procedure but must fall back to the legacy insert_interval procedure on datasets
created before the union procedure existed (create_schema only runs at dataset
creation, so old datasets never get the new procedure)."""

import os
import shutil
from pathlib import Path

import numpy as np
import pytest
from dotenv import load_dotenv

from atriumdb import AtriumSDK
from atriumdb.sql_handler.maria.maria_handler import MariaDBHandler

DB_NAME = 'interval_union_fallback_test'


def test_interval_union_fallback_on_legacy_dataset():
    load_dotenv()
    host = os.getenv("MARIA_DB_HOST")
    if host is None:
        pytest.skip("MariaDB connection not configured (.env)")
    user = os.getenv("MARIA_DB_USER")
    password = os.getenv("MARIA_DB_PASSWORD")
    port = int(os.getenv("MARIA_DB_PORT"))

    dataset_path = Path(__file__).parent / "test_datasets" / f"maria_{DB_NAME}"
    shutil.rmtree(dataset_path, ignore_errors=True)
    maria_handler = MariaDBHandler(host, user, password, DB_NAME, port)
    maria_handler.maria_connect_no_db().cursor().execute(f"DROP DATABASE IF EXISTS `{DB_NAME}`")

    connection_params = {'sqltype': 'mariadb', 'host': host, 'user': user, 'password': password,
                         'database': DB_NAME, 'port': port}
    sdk = AtriumSDK.create_dataset(dataset_location=dataset_path, database_type='mariadb',
                                   connection_params=connection_params)

    # Simulate a dataset created before the union procedure existed.
    with sdk.sql_handler.maria_db_connection(begin=True) as (conn, cursor):
        cursor.execute("DROP PROCEDURE IF EXISTS insert_interval_union")

    measure_id = sdk.insert_measure(measure_tag='sig', freq=1.0, freq_units="Hz")
    device_id = sdk.insert_device(device_tag='dev')

    # Merge-mode writes must silently fall back to the legacy procedure.
    times = np.arange(100, dtype=np.int64) * 1_000_000_000
    values = np.arange(100, dtype=np.int64)
    sdk.write_time_value_pairs(measure_id, device_id, times, values, period=1_000_000_000, time_units="ns")

    assert sdk.sql_handler._interval_union_proc_available is False

    # A second write keeps using the fallback without re-raising.
    times2 = times + 100 * 1_000_000_000
    sdk.write_time_value_pairs(measure_id, device_id, times2, values.copy(), period=1_000_000_000, time_units="ns")

    intervals = sdk.get_interval_array(measure_id, device_id=device_id, gap_tolerance_nano=0)
    assert len(intervals) == 1
    assert intervals[0][0] == 0

    _, rt, rv = sdk.get_data(measure_id, 0, int(times2[-1]) + 1_000_000_001, device_id=device_id)
    assert rt.size == 200

    shutil.rmtree(dataset_path, ignore_errors=True)
