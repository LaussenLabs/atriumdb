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

from atriumdb.atrium_sdk import AtriumSDK
from atriumdb.adb_functions import convert_from_nanohz
from pathlib import Path
import shutil
import random
from concurrent.futures import ProcessPoolExecutor, as_completed

import os
from dotenv import load_dotenv

from atriumdb.sql_handler.maria.maria_handler import MariaDBHandler
from tests.testing_framework import _test_for_both

load_dotenv()

# Get MariaDB connection details from .env file
host = os.getenv("MARIA_DB_HOST")
user = os.getenv("MARIA_DB_USER")
password = os.getenv("MARIA_DB_PASSWORD")
port = int(os.getenv("MARIA_DB_PORT"))

DB_NAME = 'source_test'

TSC_DATASET_DIR = Path(__file__).parent / 'test_tsc_data' / 'concurrent_measure_device_inserts'

database_uri = f"mysql+pymysql://{user}:{password}@{host}/{DB_NAME}"

process_sdk = None

NUM_SOURCES = 100


def test_concurrent_source_writing():
    _test_for_both(DB_NAME, _test_concurrent_source_writing)


def _test_concurrent_source_writing(db_type, dataset_location, connection_params):
    if db_type == "sqlite":
        # Sqlite locks instead of allowing parallel writes.
        return

    sdk = AtriumSDK.create_dataset(
        dataset_location=dataset_location, database_type=db_type, connection_params=connection_params)

    test_measure_list = [(str(measure_id), random.randint(1, 2048)) for measure_id in range(NUM_SOURCES)]

    test_device_list = [str(device_id) for device_id in range(NUM_SOURCES)]

    num_processes = 4

    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        futures = []
        for _ in range(num_processes):
            futures.append(executor.submit(
                write_source_info_process, test_measure_list, test_device_list, dataset_location,
                db_type, connection_params))

        for future in as_completed(futures):
            future.result()

    _check_source_equality(sdk, test_device_list, test_measure_list)

    # Try again with new SDK object.
    sdk = AtriumSDK(dataset_location=dataset_location, metadata_connection_type=db_type,
                    connection_params=connection_params)

    _check_source_equality(sdk, test_device_list, test_measure_list)


def _check_source_equality(sdk, test_device_list, test_measure_list):
    for (measure_tag, freq_hz) in test_measure_list:
        measure_id = sdk.get_measure_id(measure_tag, freq_hz, freq_units="Hz")
        assert measure_id is not None
    for device_tag in test_device_list:
        device_id = sdk.get_device_id(device_tag)
        assert device_id is not None
    read_measure_list = []
    for measure_id in sdk.get_all_measures():
        measure_info_dict = sdk.get_measure_info(measure_id)
        read_measure_list.append((measure_info_dict['tag'], convert_from_nanohz(measure_info_dict['freq_nhz'], freq_units="Hz")))
    read_device_list = []
    for device_id in sdk.get_all_devices():
        device_info_dict = sdk.get_device_info(device_id)
        read_device_list.append(device_info_dict['tag'])
    assert sorted(read_measure_list) == sorted(test_measure_list)
    assert sorted(read_device_list) == sorted(test_device_list)


def write_source_info_process(measure_list, device_list, dataset_location, db_type, connection_params):
    global process_sdk
    if process_sdk is None:
        process_sdk = AtriumSDK(dataset_location=dataset_location, metadata_connection_type=db_type,
                                connection_params=connection_params)

    for _ in range(2):
        for (measure_tag, freq_hz) in measure_list:
            process_sdk.insert_measure(measure_tag, freq_hz, freq_units="Hz")

        for device_tag in device_list:
            process_sdk.insert_device(device_tag)
