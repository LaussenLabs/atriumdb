import numpy as np

from atriumdb import AtriumSDK
from atriumdb.windowing.window import CommonWindowFormat
from atriumdb.windowing.window_config import WindowConfig
from tests.test_mit_bih import write_mit_bih_to_dataset
from tests.testing_framework import _test_for_both

DB_NAME = 'atrium-cwf'

MAX_RECORDS = 1
SEED = 42


def test_cwf():
    _test_for_both(DB_NAME, _test_cwf)


def _test_cwf(db_type, dataset_location, connection_params):


    sdk = AtriumSDK.create_dataset(
        dataset_location=dataset_location, database_type=db_type, connection_params=connection_params)

    write_mit_bih_to_dataset(sdk, max_records=MAX_RECORDS, seed=SEED)

    print()
    start_time_s = 0
    end_time_s = 100
    start_time_nano = start_time_s * (10 ** 9)
    end_time_nano = end_time_s * (10 ** 9)
    device_one = "100"
    measure_one = 'MLII'

    measure_id = sdk.get_measure_id(measure_tag=measure_one, freq=500, freq_units="Hz", units="mV")
    device_id = sdk.get_device_id(device_tag=device_one)
    assert measure_id is not None
    assert device_id is not None
    windows_size_sec = 5
    window_slide_sec = 5
    wc = WindowConfig(measures=[measure_one], window_size_sec=windows_size_sec, window_slide_sec=window_slide_sec,
                      allowed_lateness_sec=0)

    window_generator = sdk.get_windows(wc, start_time_nano, end_time_nano, device_tag=device_one)

    _, expected_times, expected_values = sdk.get_data(measure_id, start_time_nano, end_time_nano, device_id=device_id)

    # Add all the windows together
    collected_times = []
    collected_values = []
    for window in window_generator:
        for measure_triplet, signal in window.signals.items():
            collected_times.append(signal.times)
            collected_values.append(signal.data)

    collected_times = np.concatenate(collected_times)
    collected_values = np.concatenate(collected_values)

    # Filter out the NaN values.
    not_nan_index = np.nonzero(~np.isnan(collected_values))

    collected_times = collected_times[not_nan_index]
    collected_values = collected_values[not_nan_index]

    assert np.array_equal(expected_times, collected_times)
    assert np.array_equal(expected_values, collected_values)
