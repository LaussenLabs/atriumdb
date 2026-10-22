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
import numpy as np

from atriumdb import AtriumSDK, DatasetDefinition
import shutil

from atriumdb.sql_handler.maria.maria_handler import MariaDBHandler
from atriumdb.transfer.adb.dataset import transfer_data
from tests.test_mit_bih import write_mit_bih_to_dataset, assert_mit_bih_to_dataset
from tests.testing_framework import _test_for_both, create_sibling_sdk

DB_NAME = 'atrium-def-start-end'


def test_transfer_start_end():
    _test_for_both(DB_NAME, _test_transfer_start_end)


def _test_transfer_start_end(db_type, dataset_location, connection_params):
    # Setup
    sdk_1 = AtriumSDK.create_dataset(
        dataset_location=dataset_location, database_type=db_type, connection_params=connection_params)

    sdk_2 = create_sibling_sdk(connection_params, dataset_location, db_type)

    freq_nano = 1_000_000_000
    period_nano = (10 ** 18) // freq_nano
    device_id = sdk_1.insert_device(device_tag="device")
    measure_id = sdk_1.insert_measure(measure_tag="measure", freq=freq_nano, units="mV")

    start_time = 1_000_000_000
    num_values = 10_000
    end_time = start_time + (num_values * period_nano)
    times = np.arange(start_time, end_time, period_nano)
    values = (np.sin(times) * 1000).astype(np.int64)
    scale_m = 1 / 1000
    scale_b = 0

    sdk_1.write_data_easy(
        measure_id, device_id, times, values, freq_nano, scale_m=scale_m, scale_b=scale_b)

    third_time = int(times[num_values // 3])
    second_third_time = int(times[(2 * num_values) // 3])

    definition = DatasetDefinition(measures=["measure"], device_ids={device_id: "all"})

    window_slide_nano = window_duration_nano = 10 * (10 ** 9)
    # Sanity check to confirm that some windows are outside the middle third of data.
    has_windows_outside_middle_third = False
    for window in sdk_1.get_iterator(definition, window_duration_nano, window_slide_nano, time_units="ns"):
        if window.start_time < third_time or window.start_time >= second_third_time:
            has_windows_outside_middle_third = True
            break

    assert has_windows_outside_middle_third, "Sanity Check Failed"

    # Now use global start-end and confirm that all the windows are inside the bounds
    definition = DatasetDefinition(measures=["measure"], device_ids={device_id: "all"})

    for window in sdk_1.get_iterator(definition, window_duration_nano, window_slide_nano, time_units="ns",
                                     start_time=third_time, end_time=second_third_time):
        if window.start_time < third_time or window.start_time >= second_third_time:
            assert False, f"{window.start_time} < {third_time} or {window.start_time} > {second_third_time}"

    # A partial window at the end bound holds no samples from after it.
    definition = DatasetDefinition(measures=["measure"], device_ids={device_id: "all"})
    for window in sdk_1.get_iterator(definition, window_duration_nano, window_slide_nano, time_units="ns",
                                     start_time=third_time, end_time=second_third_time, iterator_type="lightmapped"):
        signal = window.signals[("measure", freq_nano / 10 ** 9, 'mV')]
        assert np.all(signal['times'][~np.isnan(signal['values'])] < second_third_time)

    # Test transfer
    transfer_data(sdk_1, sdk_2, definition, gap_tolerance=None, deidentify=False, patient_info_to_transfer=None,
                  include_labels=False, start_time=third_time, end_time=second_third_time)

    # Check all transferred data.
    for window in sdk_2.get_iterator(definition, window_duration_nano, window_slide_nano, time_units="ns"):
        signal = window.signals[("measure", freq_nano / 10 ** 9, 'mV')]
        times, values = signal['times'], signal['values']

        # Get indices where values are not NaN
        valid_indices = ~np.isnan(values)
        valid_times = times[valid_indices]

        if valid_times.size == 0:
            continue
        valid_start_time = valid_times[0]
        valid_end_time = valid_times[-1]

        if valid_start_time < third_time or valid_end_time >= second_third_time:
            assert False, f"{valid_start_time} < {third_time} or {valid_end_time} >= {second_third_time}"

    # Test transfer by patient
    sdk_2 = create_sibling_sdk(connection_params, dataset_location, db_type)  # Reset the 2nd db

    patient_id = sdk_1.insert_patient()
    sdk_1.insert_device_patient_data([(device_id, patient_id, start_time, end_time)])
    definition = DatasetDefinition(measures=["measure"], patient_ids={patient_id: "all"})

    transfer_data(sdk_1, sdk_2, definition, gap_tolerance=None, deidentify=False, patient_info_to_transfer=None,
                  include_labels=False, start_time=third_time, end_time=second_third_time)

    # Check all transferred data.
    for window in sdk_2.get_iterator(definition, window_duration_nano, window_slide_nano, time_units="ns"):
        signal = window.signals[("measure", freq_nano / 10 ** 9, 'mV')]
        times, values = signal['times'], signal['values']

        # Get indices where values are not NaN
        valid_indices = ~np.isnan(values)
        valid_times = times[valid_indices]

        if valid_times.size == 0:
            continue
        valid_start_time = valid_times[0]
        valid_end_time = valid_times[-1]

        if valid_start_time < third_time or valid_end_time >= second_third_time:
            assert False, f"{valid_start_time} < {third_time} or {valid_end_time} >= {second_third_time}"

    # Test transfer with specific source intervals
    sdk_2 = create_sibling_sdk(connection_params, dataset_location, db_type)  # Reset the 2nd db

    definition = DatasetDefinition(measures=["measure"],
                                   patient_ids={patient_id: [{'start': start_time, 'end': end_time}]})

    transfer_data(sdk_1, sdk_2, definition, gap_tolerance=None, deidentify=False, patient_info_to_transfer=None,
                  include_labels=False, start_time=third_time, end_time=second_third_time)

    # Check all transferred data.
    for window in sdk_2.get_iterator(definition, window_duration_nano, window_slide_nano, time_units="ns"):
        signal = window.signals[("measure", freq_nano / 10 ** 9, 'mV')]
        times, values = signal['times'], signal['values']

        # Get indices where values are not NaN
        valid_indices = ~np.isnan(values)
        valid_times = times[valid_indices]

        if valid_times.size == 0:
            continue
        valid_start_time = valid_times[0]
        valid_end_time = valid_times[-1]

        if valid_start_time < third_time or valid_end_time >= second_third_time:
            assert False, f"{valid_start_time} < {third_time} or {valid_end_time} >= {second_third_time}"


def test_transfer_between_samples():
    _test_for_both(f"{DB_NAME}-between-samples", _test_transfer_between_samples)


def _test_transfer_between_samples(db_type, dataset_location, connection_params):
    sdk = AtriumSDK.create_dataset(
        dataset_location=dataset_location, database_type=db_type, connection_params=connection_params)
    measure_id = sdk.insert_measure("measure", 1, "mV", freq_units="Hz")
    device_id = sdk.insert_device("device")
    sdk.write_segment(measure_id, device_id, np.arange(100), 0, freq=1, time_units="s", freq_units="Hz",
                      scale_m=1.0, scale_b=0.0)

    # The device moves to the second patient 0.3 s after the sample at 50 s, which belongs to the first patient.
    first_patient_id, second_patient_id = sdk.insert_patient(), sdk.insert_patient()
    transfer_time = 50_300_000_000
    sdk.insert_device_patient_data([(device_id, first_patient_id, 0, transfer_time),
                                    (device_id, second_patient_id, transfer_time, 100 * 10 ** 9)])
    definition = DatasetDefinition(measures=["measure"], patient_ids={second_patient_id: "all"})
    for iterator_type in ["iterator", "lightmapped"]:
        window = next(iter(sdk.get_iterator(definition, 10, 10, time_units="s", iterator_type=iterator_type)))
        values = window.signals[("measure", 1.0, "mV")]["values"]
        assert np.isnan(values[0]) and values[1] == 51
