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

import os
from dotenv import load_dotenv

from atriumdb.sql_handler.maria.maria_handler import MariaDBHandler
from atriumdb.sql_handler.sqlite.sqlite_handler import SQLiteHandler

load_dotenv()

# Get MariaDB connection details from .env file
host = os.getenv("MARIA_DB_HOST")
user = os.getenv("MARIA_DB_USER")
password = os.getenv("MARIA_DB_PASSWORD")
port = int(os.getenv("MARIA_DB_PORT"))

DB_NAME = 'tsc_insert'

SQLITE_FILE = Path(__file__).parent / DB_NAME / 'meta' / 'index.db'


def test_insert_tsc_file_data():
    maria_handler = MariaDBHandler(host, user, password, DB_NAME)
    maria_handler.maria_connect_no_db().cursor().execute(f"DROP DATABASE IF EXISTS {DB_NAME}")
    maria_handler.create_schema()
    _test_insert_tsc_file_data(maria_handler)

    SQLITE_FILE.unlink(missing_ok=True)
    SQLITE_FILE.parent.mkdir(parents=True, exist_ok=True)
    sqlite_handler = SQLiteHandler(SQLITE_FILE)
    sqlite_handler.create_schema()
    _test_insert_tsc_file_data(sqlite_handler)


def _test_insert_tsc_file_data(sql_handler):
    file_path = "test_file.txt"

    # Insert 2 measures and 2 devices
    for i in range(1, 3):
        sql_handler.insert_measure(f"tag_{i}", 1000)
        sql_handler.insert_device(f"tag_{i}")

    block_data = [{"measure_id": 1, "device_id": 1, "start_byte": 0, "num_bytes": 100, "start_time_n": 0,
                   "end_time_n": 100, "num_values": 10},
                  {"measure_id": 2, "device_id": 2, "start_byte": 100, "num_bytes": 200, "start_time_n": 100,
                   "end_time_n": 300, "num_values": 20}]

    interval_data = [{"measure_id": 1, "device_id": 1, "start_time_n": 0, "end_time_n": 100},
                     {"measure_id": 2, "device_id": 2, "start_time_n": 100, "end_time_n": 300}]
    sql_handler.insert_tsc_file_data(file_path, block_data, interval_data, None)

    # Verify that file path has been inserted into file_index
    result = sql_handler.select_file(file_path=file_path)
    assert result is not None
    file_id = result[0]

    # Verify that block_data has been inserted into block_index
    for block in block_data:
        result = sql_handler.select_block(
            measure_id=block["measure_id"], device_id=block["device_id"], file_id=file_id,
            start_byte=block["start_byte"], num_bytes=block["num_bytes"], start_time_n=block["start_time_n"],
            end_time_n=block["end_time_n"], num_values=block["num_values"])
        assert result is not None

    # Verify that interval_data has been inserted into interval_index
    for interval in interval_data:
        result = sql_handler.select_interval(
            interval["measure_id"], interval["device_id"], interval["start_time_n"], interval["end_time_n"])
        assert result is not None
