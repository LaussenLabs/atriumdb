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

from atriumdb import AtriumSDK, T_TYPE_GAP_ARRAY_INT64_INDEX_DURATION_NANO, V_TYPE_INT64, V_TYPE_DELTA_INT64, \
    V_TYPE_DOUBLE
import numpy as np
import random
from matplotlib import pyplot as plt

from tests.generate_wfdb import get_records
from tests.test_mit_bih import create_gaps
from tests.test_transfer_info import insert_random_patients
from tests.testing_framework import _test_for_both

DB_NAME = 'small-block'

MAX_RECORDS = 1
SEED = 42

global_gap_index = 0
global_d_record, annotation = next(get_records(dataset_name='mitdb', physical=False))


def test_small_block():
    global global_gap_index
    _test_for_both(DB_NAME, _test_small_block)


def _test_small_block_experiment(db_type, dataset_location, connection_params):
    sdk = AtriumSDK.create_dataset(
        dataset_location=dataset_location, database_type=db_type, connection_params=connection_params, no_pool=True)

    one_d_record = global_d_record

    device_id = sdk.insert_device(device_tag=one_d_record.record_name)
    measure_tag = one_d_record.sig_name[0]
    units = one_d_record.units[0]
    freq_nano = 500 * (10 ** 9)
    period_ns = (10 ** 18) // freq_nano
    measure_id = sdk.insert_measure(measure_tag=measure_tag, freq=freq_nano,
                                    units=units)

    value_data = one_d_record.d_signal.T[0].astype(np.int64)[:1000]
    scale_m = (1 / one_d_record.adc_gain[0])
    scale_b = (-one_d_record.adc_zero[0] / one_d_record.adc_gain[0])

    start_time = 0

    # Time type 2
    raw_t_t = 2
    encoded_t_t = 2

    # gap_data = np.array([8001, 10 ** 12], dtype=np.int64)
    # gap_data = create_gaps(value_data.size, period_ns, gap_density=0.001).flatten()
    # gap_data = [[i, 4_000_000] for i in range(value_data.size)]
    # gap_data = np.array(gap_data, dtype=np.int64).flatten()
    gap_data = np.array([global_gap_index, 10_000_000], dtype=np.int64)
    expected_times = np.arange(start_time, start_time + (period_ns * value_data.size), period_ns, dtype=np.int64)

    for gap_i, gap_dur in gap_data.reshape(-1, 2):
        expected_times[gap_i:] += gap_dur

    if np.issubdtype(value_data.dtype, np.integer):
        raw_v_t = V_TYPE_INT64
        encoded_v_t = V_TYPE_DELTA_INT64
    else:
        raw_v_t = V_TYPE_DOUBLE
        encoded_v_t = V_TYPE_DOUBLE

    sdk.block.block_size = 4
    # Call the write_data method with the determined parameters
    sdk.write_data(measure_id, device_id, gap_data, value_data, freq_nano, start_time, raw_time_type=raw_t_t,
                   raw_value_type=raw_v_t, encoded_time_type=encoded_t_t, encoded_value_type=encoded_v_t,
                   scale_m=scale_m, scale_b=scale_b)

    _, times, values = sdk.get_data(measure_id, start_time, start_time + (10 ** 12), device_id=device_id, analog=False)

    if not np.array_equal(expected_times, times):
        print(f"Didn't work for {global_gap_index}")


def _test_small_block(db_type, dataset_location, connection_params):
    np.random.seed(SEED)
    random.seed(SEED)
    sdk = AtriumSDK.create_dataset(
        dataset_location=dataset_location, database_type=db_type, connection_params=connection_params)

    one_p_record, annotation = next(get_records(dataset_name='mitdb'))
    one_d_record, annotation = next(get_records(dataset_name='mitdb', physical=False))

    for key in one_p_record.__dict__.keys():
        value = one_p_record.__dict__[key]
        print(f"{key}: {value}")

    device_id = sdk.insert_device(device_tag=one_p_record.record_name)
    measure_tag = one_p_record.sig_name[0]
    units = one_p_record.units[0]
    freq_nano = 500 * (10 ** 9)
    period_ns = (10 ** 18) // freq_nano
    measure_id = sdk.insert_measure(measure_tag=measure_tag, freq=freq_nano,
                                    units=units)

    value_data = one_d_record.d_signal.T[0].astype(np.int64)[:20]
    scale_m = (1 / one_d_record.adc_gain[0])
    scale_b = (-one_d_record.adc_zero[0] / one_d_record.adc_gain[0])

    start_time = 0

    # Time type 2
    raw_t_t = encoded_t_t = 2

    # gap_data = np.array([8001, 10 ** 12], dtype=np.int64)
    # gap_data = create_gaps(value_data.size, period_ns, gap_density=0.001).flatten()
    # gap_data = [[i, 4_000_000] for i in range(value_data.size)]
    # gap_data = np.array(gap_data, dtype=np.int64).flatten()

    gap_data = np.array([10, 10_000_000], dtype=np.int64)
    expected_times = np.arange(start_time, start_time + (period_ns * value_data.size), period_ns, dtype=np.int64)

    for gap_i, gap_dur in gap_data.reshape(-1, 2):
        expected_times[gap_i:] += gap_dur

    if np.issubdtype(value_data.dtype, np.integer):
        raw_v_t = V_TYPE_INT64
        encoded_v_t = V_TYPE_DELTA_INT64
    else:
        raw_v_t = V_TYPE_DOUBLE
        encoded_v_t = V_TYPE_DOUBLE

    sdk.block.block_size = 10
    # Call the write_data method with the determined parameters
    sdk.write_data(measure_id, device_id, gap_data, value_data, freq_nano, start_time, raw_time_type=raw_t_t,
                   raw_value_type=raw_v_t, encoded_time_type=encoded_t_t, encoded_value_type=encoded_v_t,
                   scale_m=scale_m, scale_b=scale_b)

    headers, times, values = sdk.get_data(measure_id, start_time, start_time + (10 ** 13), device_id=device_id,
                                          analog=False)

    assert compare_arrays(value_data, values)
    assert compare_arrays(expected_times, times)


def compare_arrays(arr1, arr2, sample=5):
    """Compare two numpy arrays, providing detailed differences.

    Parameters:
    arr1, arr2 (numpy.array): The two arrays to compare.
    sample (int): The number of sample differing elements to display.

    Returns:
    bool: True if arrays are identical, False otherwise.
    """

    if arr1.shape != arr2.shape:
        print(f"The two arrays have different shapes: {arr1.shape} vs {arr2.shape}")
        return False

    # Compute a boolean mask of the differences
    differences = arr1 != arr2

    # Count the number of differences
    num_differences = np.count_nonzero(differences)

    if num_differences == 0:
        # print("The two arrays are identical.")
        return True
    else:
        print(f"There are {num_differences} differing elements.")

        # Find the indices of differences
        diff_indices = np.transpose(np.nonzero(differences))

        # Compute the mean and std of differences
        diff_values = arr1[differences] - arr2[differences]
        print(f"The mean difference is {np.mean(diff_values)} and the standard deviation is {np.std(diff_values)}.")

        # Compute the min/max differences
        print(
            f"The minimum difference is {np.min(diff_values)}, and the maximum difference is {np.max(diff_values)}.")

        # Show some samples of differing elements
        sample_indices = diff_indices[:sample]
        for idx in sample_indices:
            idx_tup = tuple(idx)
            print(f"Difference at index {idx_tup}: {arr1[idx_tup]} vs {arr2[idx_tup]}.")

        return False

