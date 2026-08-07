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

from pathlib import Path
import time

import os
import pytest
from dotenv import load_dotenv

from atriumdb.sql_handler.maria.maria_handler import MariaDBHandler
from atriumdb.sql_handler.sqlite.sqlite_handler import SQLiteHandler
from tests.testing_framework import maria_db_credentials

load_dotenv()

# Read through the shared helper: it returns None instead of raising when the
# MariaDB variables are absent, so a run without them degrades to SQLite-only
# rather than dying at import and taking the whole collection down with it.
_maria_credentials = maria_db_credentials() or {}
host = _maria_credentials.get("host")
user = _maria_credentials.get("user")
password = _maria_credentials.get("password")
port = _maria_credentials.get("port")

# Every test here builds a MariaDBHandler directly, so the file as a whole
# requires MariaDB and is deselected by ``-m "not mariadb"``.
pytestmark = pytest.mark.mariadb

DB_NAME = 'blocks'

SQLITE_FILE = Path(__file__).parent / DB_NAME / 'meta' / 'index.db'


def test_block_select():
    maria_handler = MariaDBHandler(host, user, password, DB_NAME, port)
    maria_handler.maria_connect_no_db().cursor().execute(f"DROP DATABASE IF EXISTS {DB_NAME}")
    maria_handler.create_schema()
    _test_block_select(maria_handler)

    SQLITE_FILE.unlink(missing_ok=True)
    SQLITE_FILE.parent.mkdir(parents=True, exist_ok=True)
    sqlite_handler = SQLiteHandler(SQLITE_FILE)
    sqlite_handler.create_schema()
    _test_block_select(sqlite_handler)


def _test_block_select(sql_handler):
    # Insert a couple measures, devices, patients, encounters, device_encounters, blocks
    t_now = 0
    measure_id = sql_handler.insert_measure("ECG", 1000)
    device_id = sql_handler.insert_device("ECG1")
    # insert institution, unit, bed
    institution_id = sql_handler.insert_institution("Hospital")
    unit_id = sql_handler.insert_unit(institution_id, "ICU", "ICU")
    bed_id = sql_handler.insert_bed(unit_id, "ICU1")

    # patient_id = sql_handler.insert_patient(mrn="12345")
    patient_id = sql_handler.insert_patient(mrn=None)
    encounter_id = sql_handler.insert_encounter(patient_id, bed_id, t_now, t_now + 100)
    device_encounter_id = sql_handler.insert_device_encounter(device_id, encounter_id, t_now, t_now + 100)

    sql_handler.insert_tsc_file_data("/tmp/test.tsc",
                                     [{"measure_id": measure_id, "device_id": device_id, "start_byte": 0,
                                       "num_bytes": 100, "start_time_n": t_now, "end_time_n": t_now + 100,
                                       "num_values": 100}], [{"measure_id": measure_id, "device_id": device_id,
                                                              "start_time_n": t_now, "end_time_n": t_now + 100}], None)
    sql_handler.insert_tsc_file_data("/tmp/test2.tsc",
                                     [{"measure_id": measure_id, "device_id": device_id, "start_byte": 0,
                                       "num_bytes": 100, "start_time_n": t_now + 100,
                                       "end_time_n": t_now + 200, "num_values": 100}],
                                     [{"measure_id": measure_id, "device_id": device_id,
                                       "start_time_n": t_now + 100, "end_time_n": t_now + 200}], None)


    # Get the blocks
    print()
    print("device", sql_handler)
    print("measure_id, start_time_n=None, end_time_n=None, device_id=None, patient_id=None")
    print(measure_id, t_now, t_now + 200, device_id, None)
    blocks = sql_handler.select_blocks(measure_id, t_now, t_now + 200, device_id, None)

    print(blocks)

    print()
    print("patient", sql_handler)
    print("measure_id, start_time_n=None, end_time_n=None, device_id=None, patient_id=None")
    print(measure_id, t_now, t_now + 200, None, patient_id)
    blocks = sql_handler.select_blocks(measure_id, t_now, t_now + 200, None, patient_id)
    print(blocks)

    # Get the intervals
    print()
    print("device", sql_handler)
    print("measure_id, start_time_n=None, end_time_n=None, device_id=None, patient_id=None")
    print(measure_id, t_now, t_now + 200, device_id, None)
    intervals = sql_handler.select_intervals(measure_id, t_now, t_now + 200, device_id, None)

    print(intervals)

    print()
    print("patient", sql_handler)
    print("measure_id, start_time_n=None, end_time_n=None, device_id=None, patient_id=None")
    print(measure_id, t_now, t_now + 200, None, patient_id)
    intervals = sql_handler.select_intervals(measure_id, t_now, t_now + 200, None, patient_id)
    print(intervals)

    print()
    print(sql_handler.select_files([1, 2]))
