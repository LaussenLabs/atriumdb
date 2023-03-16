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


def assert_mit_bih_to_dataset(sdk, max_records=None):
    num_records = 0
    for record in get_records(dataset_name='mitdb'):
        if max_records and num_records >= max_records:
            return
        num_records += 1
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


def assert_partial_mit_bih_to_dataset(sdk, measure_id_list=None, device_id_list=None, start_nano=None, end_nano=None,
                                      max_records=None):
    records_tested = 0
    for record in get_records(dataset_name='mitdb'):
        device_id = sdk.get_device_id(device_tag=record.record_name)
        freq_nano = record.fs * 1_000_000_000
        period_ns = int(10 ** 9 // record.fs)
        time_arr = np.arange(record.sig_len, dtype=np.int64) * period_ns

        if max_records is not None and records_tested >= max_records:
            break

        records_tested += 1

        if device_id not in device_id_list:
            assert device_id is None
            continue

        if record.n_sig > 1:
            for i in range(len(record.sig_name)):
                measure_id = sdk.get_measure_id(measure_tag=record.sig_name[i], freq=freq_nano, units=record.units[i])

                if measure_id not in measure_id_list:
                    assert measure_id is None
                    continue

                # Get the correct slice of data according to start and end nanoseconds
                start_index = 0 if start_nano is None else np.searchsorted(time_arr, start_nano)
                end_index = time_arr.size if end_nano is None else np.searchsorted(time_arr, end_nano, side='left')
                expected_data = record.p_signal.T[i][start_index:end_index]

                # Read data
                start_nano = time_arr[0] if start_nano is None else start_nano
                end_nano = time_arr[-1] + (2 * period_ns) if end_nano is None else end_nano
                _, _, read_data = sdk.get_data(measure_id, start_nano, end_nano, device_id=device_id)

                # Check if the read data is equal to the expected data
                assert np.array_equal(read_data, expected_data)

        else:
            measure_id = sdk.get_measure_id(measure_tag=record.sig_name, freq=freq_nano, units=record.units)

            if measure_id not in measure_id_list:
                assert measure_id is None
                continue

            # Get the correct slice of data according to start and end nanoseconds
            start_index = 0 if start_nano is None else np.searchsorted(time_arr, start_nano)
            end_index = time_arr.size if end_nano is None else np.searchsorted(time_arr, end_nano, side='left')
            expected_data = record.p_signal[start_index:end_index]

            start_nano = time_arr[0] if start_nano is None else start_nano
            end_nano = time_arr[-1] + (2 * period_ns) if end_nano is None else end_nano
            _, _, read_data = sdk.get_data(measure_id, start_nano, end_nano, device_id=device_id)

            # Check if the read data is equal to the expected data
            assert np.array_equal(read_data, expected_data)


def write_mit_bih_to_dataset(sdk, max_records=None):
    num_records = 0
    for record in get_records(dataset_name='mitdb'):
        if max_records and num_records >= max_records:
            return
        num_records += 1
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
