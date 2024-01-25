import numpy as np

from atriumdb import AtriumSDK
from tests.testing_framework import _test_for_both

DB_NAME = 'oversized_block'


def test_write_oversized_block():
    _test_for_both(DB_NAME, _test_write_oversized_block_timestamp)
    _test_for_both(DB_NAME, _test_write_oversized_block_gap)


def _test_write_oversized_block_timestamp(db_type, dataset_location, connection_params):
    sdk = AtriumSDK.create_dataset(
        dataset_location=dataset_location, database_type=db_type, connection_params=connection_params)

    sdk.block.block_size = 32768

    # Create Raw Data
    start_time_s = 1234567890
    start_time_nano = start_time_s * (10 ** 9)

    freq_hz = 500
    freq_nhz = freq_hz * (10 ** 9)

    period_ns = (10 ** 18) // freq_nhz

    num_values = 1_000_000

    gap_data = [10_000, 24_000_000, 12_000, 138_000_000, 54_403, 34_560_000_000, 104_903, 56_530_000_000]
    gap_data = np.array(gap_data, dtype=np.int64)

    timestamp_arr, values, end_time_nano = make_gap_data(gap_data, start_time_nano, num_values, period_ns)

    end_time_nano = int(timestamp_arr[-1]) + period_ns

    measure_id = sdk.insert_measure(measure_tag="sig1", freq=freq_nhz, units="mV")
    device_id = sdk.insert_device(device_tag="dev1")

    sdk.write_data(measure_id, device_id, timestamp_arr, values, freq_nhz, start_time_nano, raw_time_type=1,
                   raw_value_type=1, encoded_time_type=2, encoded_value_type=3, scale_m=None, scale_b=None)

    # Check time type 1
    _, r_times, r_values = sdk.get_data(measure_id, start_time_nano, end_time_nano, device_id=device_id, time_type=1,
                                        analog=False)

    assert np.array_equal(r_times, timestamp_arr)
    assert np.array_equal(r_values, values)

    ### test for case where there is only the oversized block being created ###
    num_values = 60_000
    gap_data = [10_000, 24_000_000, 12_000, 138_000_000, 54_403, 34_560_000_000]
    gap_data = np.array(gap_data, dtype=np.int64)

    timestamp_arr, values, end_time_nano = make_gap_data(gap_data, start_time_nano, num_values, period_ns)

    measure_id = sdk.insert_measure(measure_tag="sig2", freq=freq_nhz, units="mV")
    device_id = sdk.insert_device(device_tag="dev2")

    sdk.write_data(measure_id, device_id, timestamp_arr, values, freq_nhz, start_time_nano, raw_time_type=1,
                   raw_value_type=1, encoded_time_type=2, encoded_value_type=3, scale_m=None, scale_b=None)

    # Check time type 1
    _, r_times, r_values = sdk.get_data(measure_id, start_time_nano, end_time_nano, device_id=device_id, time_type=1,
                                        analog=False)

    assert np.array_equal(r_times, timestamp_arr)
    assert np.array_equal(r_values, values)

    ### test for case where there are only full blocks ###
    num_values = 65_536
    gap_data = [10_000, 24_000_000, 12_000, 138_000_000, 54_403, 34_560_000_000]
    gap_data = np.array(gap_data, dtype=np.int64)

    timestamp_arr, values, end_time_nano = make_gap_data(gap_data, start_time_nano, num_values, period_ns)

    measure_id = sdk.insert_measure(measure_tag="sig3", freq=freq_nhz, units="mV")
    device_id = sdk.insert_device(device_tag="dev3")

    sdk.write_data(measure_id, device_id, timestamp_arr, values, freq_nhz, start_time_nano, raw_time_type=1,
                   raw_value_type=1, encoded_time_type=2, encoded_value_type=3, scale_m=None, scale_b=None)

    # Check time type 1
    _, r_times, r_values = sdk.get_data(measure_id, start_time_nano, end_time_nano, device_id=device_id, time_type=1,
                                        analog=False)

    assert np.array_equal(r_times, timestamp_arr)
    assert np.array_equal(r_values, values)


    ### test for case where there is less than one block worth of values ###
    num_values = 30_000
    gap_data = [10_000, 24_000_000, 12_000, 138_000_000]
    gap_data = np.array(gap_data, dtype=np.int64)

    timestamp_arr, values, end_time_nano = make_gap_data(gap_data, start_time_nano, num_values, period_ns)

    measure_id = sdk.insert_measure(measure_tag="sig4", freq=freq_nhz, units="mV")
    device_id = sdk.insert_device(device_tag="dev4")

    sdk.write_data(measure_id, device_id, timestamp_arr, values, freq_nhz, start_time_nano, raw_time_type=1,
                   raw_value_type=1, encoded_time_type=2, encoded_value_type=3, scale_m=None, scale_b=None)

    # Check time type 1
    _, r_times, r_values = sdk.get_data(measure_id, start_time_nano, end_time_nano, device_id=device_id, time_type=1,
                                        analog=False)

    assert np.array_equal(r_times, timestamp_arr)
    assert np.array_equal(r_values, values)


def _test_write_oversized_block_gap(db_type, dataset_location, connection_params):
    sdk = AtriumSDK.create_dataset(
        dataset_location=dataset_location, database_type=db_type, connection_params=connection_params)

    # set block size so tests on exact gap work
    sdk.block.block_size = 32768
    # Create Raw Data
    start_time_s = 1234567890
    start_time_nano = start_time_s * (10 ** 9)

    freq_hz = 500
    freq_nhz = freq_hz * (10 ** 9)
    period_ns = (10 ** 18) // freq_nhz
    num_values = 1_000_000

    # where values will be split somewhere in the middle of the last gap
    gap_data_split_last_gap = [10_000, 24_000_000, 12_000, 138_000_000, 54_403, 34_560_000_000, 104_903, 56_530_000_000]
    gap_data_split_last_gap = np.array(gap_data_split_last_gap, dtype=np.int64)

    # where values will be split somewhere in the gap just before the last gap
    gap_data_split_just_before_last_gap = [10_000, 24_000_000, 12_000, 138_000_000, 54_403, 34_560_000_000,
                                           104_903, 56_530_000_000, 950_500, 156_000_000]
    gap_data_split_just_before_last_gap = np.array(gap_data_split_just_before_last_gap, dtype=np.int64)

    # there are multiple gaps after where the gap split should happen
    gap_data_split_before_multiple_gaps = [10_000, 24_000_000, 12_000, 138_000_000, 54_403, 34_560_000_000,
                                           104_903, 56_530_000_000, 950_500, 156_000_000, 970_000, 56_000_000,
                                           985_123, 200_000_000]
    gap_data_split_before_multiple_gaps = np.array(gap_data_split_before_multiple_gaps, dtype=np.int64)

    # where the split happens on the last gap exactly
    gap_data_split_on_last_gap_exact = [10_000, 24_000_000, 12_000, 138_000_000, 54_403, 34_560_000_000,
                                        104_903, 56_530_000_000, 950_271, 56_530_000]
    gap_data_split_on_last_gap_exact = np.array(gap_data_split_on_last_gap_exact, dtype=np.int64)

    # where values will be split exactly on the gap just before the last gap
    gap_data_split_just_before_last_gap_exact = [10_000, 24_000_000, 12_000, 138_000_000, 54_403, 34_560_000_000,
                                                 104_903, 56_530_000_000, 950_271, 100_000_000, 950_500, 156_000_000]
    gap_data_split_just_before_last_gap_exact = np.array(gap_data_split_just_before_last_gap_exact, dtype=np.int64)

    # there are multiple gaps after the exact location the gap split will happen
    gap_data_split_before_multiple_gaps_exact = [10_000, 24_000_000, 12_000, 138_000_000, 54_403, 34_560_000_000,
                                                 104_903, 56_530_000_000, 950_271, 100_000_000, 950_500,
                                                 156_000_000, 970_000, 56_000_000, 985_123, 200_000_000]
    gap_data_split_before_multiple_gaps_exact = np.array(gap_data_split_before_multiple_gaps_exact, dtype=np.int64)

    gap_arrays = [gap_data_split_last_gap, gap_data_split_just_before_last_gap, gap_data_split_before_multiple_gaps,
                  gap_data_split_on_last_gap_exact, gap_data_split_just_before_last_gap_exact,
                  gap_data_split_before_multiple_gaps_exact]

    for i, gap_data in enumerate(gap_arrays):
        timestamp_arr, values, end_time_nano = make_gap_data(gap_data, start_time_nano, num_values, period_ns)

        measure_id = sdk.insert_measure(measure_tag="sig"+str(i), freq=freq_nhz, units="mV")
        device_id = sdk.insert_device(device_tag="dev"+str(i))

        sdk.write_data(measure_id, device_id, gap_data, values, freq_nhz, start_time_nano, raw_time_type=2,
                       raw_value_type=1, encoded_time_type=2, encoded_value_type=3, scale_m=None, scale_b=None)

        # Check time type 2
        _, r_times, r_values = sdk.get_data(measure_id, start_time_nano, end_time_nano, device_id=device_id,
                                            time_type=2, analog=False)

        assert np.array_equal(r_values, values)

        # Check time type 1
        _, r_times, r_values = sdk.get_data(measure_id, start_time_nano, end_time_nano, device_id=device_id,
                                            time_type=1, analog=False)

        assert np.array_equal(r_times, timestamp_arr)
        assert np.array_equal(r_values, values)

        print(f"Test {i} passed!")

    # where there is only enough values for the oversized block
    gap_data_split_only_oversized_block = np.array([10_000, 24_000_000, 12_000, 138_000_000], dtype=np.int64)
    num_values = 60_000

    timestamp_arr, values, end_time_nano = make_gap_data(gap_data_split_only_oversized_block, start_time_nano, num_values, period_ns)

    measure_id = sdk.insert_measure(measure_tag="sig"+str(len(gap_arrays)), freq=freq_nhz, units="mV")
    device_id = sdk.insert_device(device_tag="dev"+str(len(gap_arrays)))

    sdk.write_data(measure_id, device_id, gap_data_split_only_oversized_block, values, freq_nhz, start_time_nano,
                   raw_time_type=2, raw_value_type=1, encoded_time_type=2, encoded_value_type=3, scale_m=None,
                   scale_b=None)

    _, r_times, r_values = sdk.get_data(measure_id, start_time_nano, end_time_nano, device_id=device_id, time_type=1,
                                        analog=False)

    assert np.array_equal(r_times, timestamp_arr)
    assert np.array_equal(r_values, values)

    # where there arnt enough values for even one block
    gap_data_split_lessthan_full_block = np.array([10_000, 24_000_000, 12_000, 138_000_000], dtype=np.int64)
    num_values = 30_000

    timestamp_arr, values, end_time_nano = make_gap_data(gap_data_split_lessthan_full_block, start_time_nano, num_values, period_ns)

    measure_id = sdk.insert_measure(measure_tag="sig"+str(len(gap_arrays)+1), freq=freq_nhz, units="mV")
    device_id = sdk.insert_device(device_tag="dev"+str(len(gap_arrays)+1))

    sdk.write_data(measure_id, device_id, gap_data_split_lessthan_full_block, values, freq_nhz, start_time_nano,
                   raw_time_type=2, raw_value_type=1, encoded_time_type=2, encoded_value_type=3, scale_m=None,
                   scale_b=None)

    _, r_times, r_values = sdk.get_data(measure_id, start_time_nano, end_time_nano, device_id=device_id, time_type=1,
                                        analog=False)

    assert np.array_equal(r_times, timestamp_arr)
    assert np.array_equal(r_values, values)


def make_gap_data(gap_data, start_time_nano, num_values, period_ns):
    timestamp_arr = np.arange(start_time_nano, start_time_nano + (num_values * period_ns), period_ns, dtype=np.int64)

    # Add the gaps
    for i in range(gap_data.size // 2):
        index = gap_data[(i * 2)]
        gap = gap_data[(i * 2) + 1]

        timestamp_arr[index:] += gap

    # Create values
    values = (1000 * np.sin(timestamp_arr)).astype(np.int64)

    end_time_nano = int(timestamp_arr[-1]) + period_ns

    return timestamp_arr, values, end_time_nano