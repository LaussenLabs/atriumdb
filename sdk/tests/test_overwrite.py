from atriumdb.atrium_sdk import AtriumSDK, convert_to_nanoseconds
import numpy as np
from pathlib import Path
import shutil
import random
from concurrent.futures import ProcessPoolExecutor, as_completed

# from tests.generate_wfdb import get_records
from tests.testing_framework import _test_for_both

DB_NAME = 'overwrite_test'


def test_overwrite():
    _test_for_both(DB_NAME, _test_overwrite)


def _test_overwrite(db_type, dataset_location, connection_params):
    sdk = AtriumSDK.create_dataset(
        dataset_location=dataset_location, database_type=db_type, connection_params=connection_params,
        overwrite='overwrite')

    print()
    measure_tag = 'signal_1'
    freq_hz = 1
    period = 1 / freq_hz

    device_tag = 'dev_1'

    measure_id = sdk.insert_measure(measure_tag, freq_hz, freq_units="Hz")
    device_id = sdk.insert_device(device_tag)

    og_time_data = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int64)
    og_value_data = og_time_data * og_time_data

    print(og_time_data)
    print(og_value_data)
    print()
    sdk.write_data_easy(
        measure_id, device_id, og_time_data, og_value_data, freq_hz, freq_units="Hz", time_units='s')

    _, og_read_times, og_read_values = sdk.get_data(
        measure_id, int(og_time_data[0]), int(og_time_data[-1] + period), device_id=device_id, time_units='s')

    print(og_read_times)
    print(og_read_values)
    print()

    assert np.array_equal(og_time_data, og_read_times)
    assert np.array_equal(og_value_data, og_read_values)

    new_time_data = og_time_data[2:5].copy()
    new_value_data = new_time_data + 5

    print(new_time_data)
    print(new_value_data)
    print()
    sdk.write_data_easy(
        measure_id, device_id, new_time_data, new_value_data, freq_hz, freq_units="Hz", time_units='s')

    _, diff_read_times, diff_read_values = sdk.get_data(
        measure_id, int(og_time_data[0]), int(og_time_data[-1] + period), device_id=device_id, time_units='s')

    print(diff_read_times)
    print(diff_read_values)
    print()
