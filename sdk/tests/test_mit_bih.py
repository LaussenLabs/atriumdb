from atriumdb import AtriumSDK
import numpy as np

from tests.generate_wfdb import get_records
from tests.testing_framework import _test_for_both

DB_NAME = 'atrium-mit-bih'


def test_mit_bih():
    _test_for_both(DB_NAME, _test_mit_bih)


def _test_mit_bih(db_type, dataset_location, connection_params):
    sdk = AtriumSDK.create_dataset(
        dataset_location=dataset_location, database_type=db_type, connection_params=connection_params)

    write_mit_bih_to_dataset(sdk)
    assert_mit_bih_to_dataset(sdk)


def assert_mit_bih_to_dataset(sdk):
    for record in get_records(dataset_name='mitdb'):
        device_id = sdk.insert_device(device_tag=record.record_name)
        freq_nano = record.fs * 1_000_000_000
        period_ns = int(10 ** 9 // record.fs)

        time_arr = np.arange(record.sig_len, dtype=np.int64) * period_ns

        # if there are multiple signals in one record split them into two different dataset entries
        if record.n_sig > 1:
            for i in range(len(record.sig_name)):
                measure_id = sdk.insert_measure(measure_tag=record.sig_name[i], freq=freq_nano,
                                                units=record.units[i])

                headers, read_times, read_values = sdk.get_data(
                    measure_id, time_arr[0], time_arr[-1] + period_ns, device_id=device_id)

                assert np.array_equal(record.p_signal.T[i], read_values) and np.array_equal(time_arr, read_times)

        # if there is only one signal in the input file insert it
        else:
            measure_id = sdk.insert_measure(measure_tag=record.sig_name, freq=freq_nano,
                                            units=record.units)

            headers, read_times, read_values = sdk.get_data(
                measure_id, time_arr[0], time_arr[-1] + period_ns, device_id=device_id)

            assert np.array_equal(record.p_signal, read_values) and np.array_equal(time_arr, read_times)


def write_mit_bih_to_dataset(sdk):
    for record in get_records(dataset_name='mitdb'):
        device_id = sdk.insert_device(device_tag=record.record_name)
        freq_nano = record.fs * 1_000_000_000

        time_arr = np.arange(record.sig_len, dtype=np.int64) * int(10 ** 9 // record.fs)

        # if there are multiple signals in one record split them into two different dataset entries
        if record.n_sig > 1:
            for i in range(len(record.sig_name)):
                measure_id = sdk.insert_measure(measure_tag=record.sig_name[i], freq=freq_nano,
                                                units=record.units[i])
                # write data
                sdk.write_data_easy(measure_id, device_id, time_arr, record.p_signal.T[i],
                                    freq_nano, scale_m=None, scale_b=None)

        # if there is only one signal in the input file insert it
        else:
            measure_id = sdk.insert_measure(measure_tag=record.sig_name, freq=freq_nano,
                                            units=record.units)

            sdk.write_data_easy(measure_id, device_id, time_arr, record.p_signal,
                                freq_nano, scale_m=None, scale_b=None)
