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
    V_TYPE_DOUBLE, T_TYPE_TIMESTAMP_ARRAY_INT64_NANO, create_gap_arr, merge_gap_data
import numpy as np
import random

from atriumdb.adb_functions import convert_gap_data_to_timestamps, create_timestamps_from_gap_data
from tests.generate_wfdb import get_records
from tests.test_transfer_info import insert_random_patients
from tests.testing_framework import _test_for_both, create_sibling_sdk

DB_NAME = 'atrium-mit-bih'

MAX_RECORDS = 4
SEED = 42
LABEL_SET_LIST = [
    "Normal Sinus Rhythm",
    "Atrial Fibrillation",
    "Atrial Flutter",
    "Supraventricular Tachycardia",
    "Ventricular Tachycardia",
    "Ventricular Fibrillation",
    "Bradycardia",
    "Tachycardia",
    "Premature Ventricular Contraction",
    "Premature Atrial Contraction",
    "First Degree Heart Block",
    "Second Degree Heart Block",
    "Third Degree Heart Block",
    "Paced Rhythm",
    "Artifact",
    "Asystole",
    "ST Elevation",
    "ST Depression",
    "T-wave Inversion",
    "Bundle Branch Block",
    "Idioventricular Rhythm",
    "Junctional Rhythm",
    "Ectopic Rhythm",
    "Pause",
]


def test_mit_bih():
    _test_for_both(DB_NAME, _test_mit_bih)


def _test_mit_bih(db_type, dataset_location, connection_params):
    sdk = AtriumSDK.create_dataset(
        dataset_location=dataset_location, database_type=db_type, connection_params=connection_params)

    write_mit_bih_to_dataset(sdk, max_records=MAX_RECORDS, seed=SEED)
    assert_mit_bih_to_dataset(sdk, max_records=MAX_RECORDS, seed=SEED)

    # Test using caching
    sdk_cached = AtriumSDK(
        dataset_location=dataset_location, metadata_connection_type=db_type, connection_params=connection_params)

    for device_id in sdk_cached.get_all_devices():
        sdk_cached.load_device(device_id)
    assert_mit_bih_to_dataset(sdk_cached, max_records=MAX_RECORDS, seed=SEED)

    sdk_2 = create_sibling_sdk(connection_params, dataset_location, db_type)

    write_mit_bih_to_dataset(sdk_2, max_records=MAX_RECORDS, seed=SEED, use_numpy=True)
    assert_mit_bih_to_dataset(sdk_2, max_records=MAX_RECORDS, seed=SEED, use_numpy=True)

    # Test using message inserts
    sdk_2 = create_sibling_sdk(connection_params, dataset_location, db_type)

    write_mit_bih_to_dataset(sdk_2, max_records=MAX_RECORDS, seed=SEED, use_messages=True)
    assert_mit_bih_to_dataset(sdk_2, max_records=MAX_RECORDS, seed=SEED)


def assert_mit_bih_to_dataset(sdk, device_patient_map=None, max_records=None, deidentify=False, time_shift=None,
                              use_patient_id=False, seed=None, use_numpy=False):
    print()
    seed = SEED if seed is None else seed
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    num_records = 0
    for (record, annotation) in get_records(dataset_name='mitdb'):
        if max_records and num_records >= max_records:
            return
        num_records += 1
        device_id = sdk.get_device_id(device_tag=record.record_name)

        freq_nano = 500 * 1_000_000_000
        period_ns = int(10 ** 18 // freq_nano)
        if use_numpy:
            device_id = np.int64(device_id)
            freq_nano = np.int64(freq_nano)
            period_ns = np.int64(period_ns)

        time_arr = np.arange(record.sig_len, dtype=np.int64) * period_ns
        gap_data_2d = create_gaps(time_arr.size, period_ns)

        for gap_index, gap_duration in gap_data_2d:
            time_arr[gap_index:] += gap_duration

        if time_shift:
            time_arr -= time_shift

        query_args = {'patient_id': device_patient_map[device_id]} if \
            use_patient_id and device_patient_map is not None else {'device_tag': record.record_name}

        # if there are multiple signals in one record, split them into two different dataset entries
        if record.n_sig > 1:
            for i in range(len(record.sig_name)):
                measure_id = sdk.get_measure_id(measure_tag=record.sig_name[i], freq=freq_nano,
                                                units=record.units[i])

                expected_values = record.p_signal.T[i].astype(np.float64)
                expected_times = time_arr

                start_time_n = int(time_arr[0])
                end_time_n = int(time_arr[-1]) + period_ns

                if use_numpy:
                    start_time_n = np.int64(start_time_n)
                    end_time_n = np.int64(end_time_n)

                headers, read_times, read_values = sdk.get_data(
                    start_time_n=start_time_n,
                    end_time_n=end_time_n,
                    measure_tag=record.sig_name[i],
                    freq=freq_nano,
                    units=record.units[i],
                    **query_args)

                if expected_values.shape != read_values.shape or not np.allclose(expected_values, read_values):
                    print("Wrong Values")
                    print(f"Expected: {expected_values.shape, expected_values.dtype}")
                    print(f"Actual: {read_values.shape, read_values.dtype}")

                    print(expected_values)
                    print(read_values)

                if not np.array_equal(expected_times, read_times):
                    print("Wrong Times")
                    print(f"Expected: {expected_times.shape, expected_times.dtype}")
                    print(f"Actual: {read_times.shape, read_times.dtype}")

                    print(expected_times)
                    print(read_times)

                assert expected_values.shape == read_values.shape
                assert np.allclose(expected_values, read_values)
                assert np.array_equal(expected_times, read_times)

        # if there is only one signal in the input file insert it
        else:
            measure_id = sdk.get_measure_id(measure_tag=record.sig_name, freq=freq_nano,
                                            units=record.units)

            start_time_n = int(time_arr[0])
            end_time_n = int(time_arr[-1]) + period_ns

            if use_numpy:
                start_time_n = np.int64(start_time_n)
                end_time_n = np.int64(end_time_n)

            headers, read_times, read_values = sdk.get_data(measure_id, start_time_n, end_time_n,
                                                            **query_args)

            assert np.array_equal(record.p_signal, read_values) and np.array_equal(time_arr, read_times)


def write_mit_bih_to_dataset(sdk, max_records=None, seed=None, label_set_list=None, use_numpy=False,
                             use_messages=False):
    seed = SEED if seed is None else seed
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    label_set_list = LABEL_SET_LIST if label_set_list is None else label_set_list

    num_records = 0

    device_patient_dict = {}
    for (record, annotation), (d_record, d_annotation) in zip(get_records(dataset_name='mitdb'),
                                                              get_records(dataset_name='mitdb', physical=False)):
        if max_records and num_records >= max_records:
            return
        num_records += 1
        if use_numpy:
            device_id = np.int64(sdk.insert_device(device_tag=record.record_name))
            freq_nano = np.int64(500 * 1_000_000_000)
            period_nano = np.int64(10 ** 18 // freq_nano)
        else:
            device_id = sdk.insert_device(device_tag=record.record_name)
            freq_nano = 500 * 1_000_000_000
            period_nano = int(10 ** 18 // freq_nano)

        time_arr = np.arange(record.sig_len, dtype=np.int64) * period_nano

        gap_data_2d = create_gaps(time_arr.size, period_nano)

        message_starts = [0]
        message_num_values = []

        last_gap = 0
        for gap_index, gap_duration in gap_data_2d:
            time_arr[gap_index:] += gap_duration

            message_num_values.append(gap_index - last_gap)
            message_starts.append(time_arr[gap_index])
            last_gap = gap_index
        message_num_values.append(record.sig_len - last_gap)

        if use_numpy:
            patient_id = np.int64(insert_random_patients(sdk, 1)[0])

            start_time = np.int64(time_arr[0])
            end_time = np.int64(time_arr[-1] + period_nano)
        else:
            patient_id = insert_random_patients(sdk, 1)[0]

            start_time = int(time_arr[0])
            end_time = int(time_arr[-1] + period_nano)

        device_patient_dict[int(device_id)] = int(patient_id)
        sdk.insert_device_patient_data([(device_id, patient_id, start_time, end_time)])

        # Divide the waveform into random segments and assign random labels
        num_segments = random.randint(10, 100)
        segment_duration = (end_time - start_time) // num_segments
        for segment in range(num_segments):
            segment_start = start_time + segment * segment_duration
            segment_end = segment_start + segment_duration

            if use_numpy:
                segment_start = np.int64(start_time + segment * segment_duration)
                segment_end = np.int64(segment_start + segment_duration)

            label = random.choice(label_set_list)
            sdk.insert_label(name=label, device=device_id, start_time=segment_start, end_time=segment_end,
                             time_units='ns')

        if record.n_sig > 1:
            for i in range(len(record.sig_name)):
                write_to_sdk(freq_nano, device_id, gap_data_2d, time_arr, start_time, sdk, record, d_record, i,
                             message_starts, message_num_values, use_messages=use_messages)
        else:
            write_to_sdk(freq_nano, device_id, gap_data_2d, time_arr, start_time, sdk, record, d_record, None,
                         message_starts, message_num_values, use_messages=use_messages)

    return device_patient_dict


def write_to_sdk(freq_nano, device_id, gap_data_2d, time_arr, start_time, sdk, p_record, d_record, signal_i,
                 message_starts, message_num_values, use_messages=False):
    measure_tag, scale_b, scale_m, units, value_data = get_record_data_for_ingest(d_record, p_record, signal_i)

    measure_id = sdk.insert_measure(measure_tag=measure_tag, freq=freq_nano,
                                    units=units)

    # Create random block_size
    sdk.block.block_size = random.choice([2 ** exp for exp in range(11, 21)])
    # sdk.block.block_size = 2 ** 11

    # gap tolerance
    gap_tolerance = 10_000_000_000  # 10 seconds

    # Determine the raw and encoded value types based on the dtype of value_data
    if np.issubdtype(value_data.dtype, np.integer):
        raw_v_t = V_TYPE_INT64
        encoded_v_t = V_TYPE_DELTA_INT64
    else:
        raw_v_t = V_TYPE_DOUBLE
        encoded_v_t = V_TYPE_DOUBLE

    # Write data
    if random.random() < 0.5:
        # Time type 1

        raw_t_t = T_TYPE_TIMESTAMP_ARRAY_INT64_NANO
        if random.random() < 0.5:
            # Time Value pair encoding
            encoded_t_t = T_TYPE_TIMESTAMP_ARRAY_INT64_NANO
            sdk.block.t_compression = 3
            sdk.block.t_compression_level = 12
        else:
            # Gap Array encoding
            encoded_t_t = T_TYPE_GAP_ARRAY_INT64_INDEX_DURATION_NANO

        if use_messages:
            sdk.write_time_value_pairs(measure_id, device_id, time_arr, value_data, freq=freq_nano, time_units="ns",
                                       freq_units="nHz", scale_m=scale_m, scale_b=scale_b)
            return
        # Call the write_data method with the determined parameters
        sdk.write_data(measure_id, device_id, time_arr, value_data, freq_nano, start_time,
                       raw_time_type=raw_t_t, raw_value_type=raw_v_t, encoded_time_type=encoded_t_t,
                       encoded_value_type=encoded_v_t, scale_m=scale_m, scale_b=scale_b, gap_tolerance=gap_tolerance)

        sdk.block.t_compression = 1
        sdk.block.t_compression_level = 0
    else:
        # Time type 2
        if use_messages:
            messages = []
            message_start_index = 0
            for num_values in message_num_values:
                messages.append(value_data[message_start_index:message_start_index + num_values])
                message_start_index += num_values

            sdk.write_segments(measure_id, device_id, messages, message_starts, freq=freq_nano,
                               freq_units='nHz', time_units='ns', scale_m=scale_m, scale_b=scale_b)
            return

        raw_t_t = T_TYPE_GAP_ARRAY_INT64_INDEX_DURATION_NANO
        encoded_t_t = T_TYPE_GAP_ARRAY_INT64_INDEX_DURATION_NANO

        # Call the write_data method with the determined parameters
        sdk.write_data(measure_id, device_id, gap_data_2d.flatten(), value_data, freq_nano, start_time,
                       raw_time_type=raw_t_t, raw_value_type=raw_v_t, encoded_time_type=encoded_t_t,
                       encoded_value_type=encoded_v_t, scale_m=scale_m, scale_b=scale_b, gap_tolerance=gap_tolerance)


def get_record_data_for_ingest(d_record, p_record, signal_i):
    if signal_i is not None:
        measure_tag = p_record.sig_name[signal_i]
        units = p_record.units[signal_i]

        if random.random() < 0.5:
            # Physical
            value_data = p_record.p_signal.T[signal_i].astype(np.float64)
            scale_m = None
            scale_b = None
        else:
            # Digital
            value_data = d_record.d_signal.T[signal_i].astype(np.int64)
            scale_m = (1 / d_record.adc_gain[signal_i])
            scale_b = (-d_record.adc_zero[signal_i] / d_record.adc_gain[signal_i])

    else:
        measure_tag = p_record.sig_name
        units = p_record.units

        if random.random() < 0.5:
            # Physical
            value_data = p_record.p_signal.T.astype(np.float64)
            scale_m = None
            scale_b = None
        else:
            # Digital
            value_data = d_record.d_signal.T.astype(np.int64)
            scale_m = (1 / d_record.adc_gain)
            scale_b = (-d_record.adc_zero / d_record.adc_gain)
    return measure_tag, scale_b, scale_m, units, value_data


def create_gaps(size, period, gap_density=0.001):
    # Determine the total number of gaps based on the gap density
    num_gaps = int(size * gap_density)

    # Generate unique random gap indices
    gap_indices = np.random.choice(size, size=num_gaps, replace=False)

    # Generate multiples of the sample period for each gap
    gap_periods = np.random.randint(1, 10_000, size=num_gaps, dtype=np.int64) * period

    # Create a 2D array with gap indices and gap periods
    gap_data = np.array([gap_indices, gap_periods]).T.astype(np.int64)

    # Sort the array by the gap indices
    gap_data = gap_data[gap_data[:, 0].argsort()]

    return gap_data


def test_create_gaps():
    gap_data = create_gaps(100, 2_000_000)
    print(gap_data)
    print(gap_data.shape, gap_data.dtype)
    print(gap_data.flatten())
